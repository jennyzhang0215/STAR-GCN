import numpy as np
import mxnet as mx
import numpy.testing as npt
import scipy.sparse as sp
import io
import mxgraph._graph_sampler as _graph_sampler
import pandas as pd
import os
import json
try:
    import cPickle as pickle
except ImportError:
    import pickle

def npy_seg_mul(lhs, ind_ptr, rhs):
    """ ret[ind_ptr[i]:ind_ptr[i+1]] = lhs[ind_ptr[i]:ind_ptr[i+1]] * rhs[i]

    Parameters
    ----------
    lhs : np.ndarray
    ind_ptr : np.ndarray
    rhs : np.ndarray

    Returns
    -------
    ret : np.ndarray
    """
    return _graph_sampler.seg_mul(lhs, ind_ptr, rhs)

def npy_seg_add(lhs, ind_ptr, rhs):
    """ ret[ind_ptr[i]:ind_ptr[i+1]] = lhs[ind_ptr[i]:ind_ptr[i+1]] + rhs[i]

    Parameters
    ----------
    lhs : np.ndarray
    ind_ptr : np.ndarray
    rhs : np.ndarray

    Returns
    -------
    ret : np.ndarray
    """
    return _graph_sampler.seg_add(lhs, ind_ptr, rhs)

def npy_seg_sum(data, ind_ptr):
    """ ret[i] = data[ind_ptr[i]:ind_ptr[i+1]].sum()

    Parameters
    ----------
    data : np.ndarray
    ind_ptr : np.ndarray

    Returns
    -------
    ret : np.ndarray
    """
    return _graph_sampler.seg_sum(data, ind_ptr)


def take1d(data, sel):
    return _graph_sampler.take_1d_omp(np.ascontiguousarray(data),
                                      np.ascontiguousarray(sel, dtype=np.int32))


def unordered_unique(data, return_counts=False, return_inverse=False):
    if return_counts:
        return _graph_sampler.unique_cnt(np.ascontiguousarray(data).astype(np.int32))
    if return_inverse:
        return _graph_sampler.unique_inverse(np.ascontiguousarray(data).astype(np.int32))
    raise NotImplementedError


def set_seed(seed):
    """Set the random seed of the inner sampling handler

    Parameters
    ----------
    seed : int

    Returns
    -------
    ret : bool
    """
    return _graph_sampler.set_seed(seed)

def _gen_edge_row_indices_by_indptr(ind_ptr, nnz):
    """ Generate the row_indices in the COO format based on the indptr

    nnz = 7
    ind_ptr = [0, 2, 5, 7]
    edge_row_indices = [0, 0, 1, 1, 1, 2, 2]

    Parameters
    ----------
    ind_ptr : np.ndarray
    nnz : int

    Returns
    -------
    edge_row_indices : np.ndarray
    """
    return _graph_sampler.gen_row_indices_by_indptr(ind_ptr.astype(np.int32), nnz)


def _shallow_copy_stacked_dict(dic):
    new_dict = {}
    for k1 in dic:
        new_dict[k1] = {}
        for k2 in dic[k1]:
            new_dict[k1][k2] = dic[k1][k2]
    return new_dict


class NodeIDRMap(object):
    def __init__(self, node_ids):
        self._rmap = dict()
        for i, node_id in enumerate(node_ids):
            self._rmap[node_id] = i

    def __getitem__(self, node_ids):
        if isinstance(node_ids, (np.ndarray, list, tuple)):
            return np.array(list(map(lambda ele: self._rmap[ele], node_ids)),
                            dtype=np.int32)
        else:
            return self._rmap[node_ids]


class NodeIDRMapFast(object):
    def __init__(self, node_ids):
        """

        Parameters
        ----------
        node_ids : np.ndarray

        For example:    node_ids = [5, 9, 6, 12]
                        _rmap = [ 0,  2, -1, -1,  1, -1, -1,  3]
                                  5|  6|          9|         12|
        """
        self._node_id_min = node_ids.min()
        self._node_id_max = node_ids.max()
        self._rmap = -1 * np.ones(self._node_id_max - self._node_id_min + 1, dtype=np.int32)
        self._rmap[node_ids - self._node_id_min] = np.arange(node_ids.size, dtype=np.int32)

    def __getitem__(self, node_ids):
        return self._rmap[node_ids - self._node_id_min]


def merge_nodes(node_ids):
    """

    Parameters
    ----------
    node_ids : list of np.ndarray or np.ndarray

    Returns
    -------
    uniq_node_ids : np.ndarray
    indices : list of np.ndarray or np.ndarray
    """
    if isinstance(node_ids, np.ndarray):
        return unordered_unique(node_ids, return_inverse=True)
    else:
        uniq_node_ids, all_indices = unordered_unique(np.concatenate(node_ids, axis=0),
                                                      return_inverse=True)
        indices = []
        begin = 0
        for ele in node_ids:
            indices.append(all_indices[begin:(begin + ele.size)])
            begin += ele.size
        return uniq_node_ids, indices


def merge_node_ids_dict(data):
    """

    Parameters
    ----------
    data : tuple/list of dict
        There are two possible situations:
        1) {key: indices}, the ``indices'' has shape (#nodes,)
        2) {(src_key, dst_key): indices}, the ``indices'' has shape (1 + K, #nodes)

    Returns
    -------
    uniq_node_ids_dict : dict
    new_idx_dict_l : list of dict
    """
    uniq_node_ids_dict = dict()
    new_idx_dict_l = []
    all_ids_dict = dict()
    for ele_dict in data:
        for key, indices in ele_dict.items():
            if isinstance(key, tuple):
                assert ele_dict[key].ndim == 2
                src_key, dst_key = key
                if src_key in all_ids_dict:
                    all_ids_dict[src_key].append(indices[0, :])
                else:
                    all_ids_dict[src_key] = [indices[0, :]]
                if dst_key in all_ids_dict:
                    all_ids_dict[dst_key].append(indices[1:, :].reshape((-1,)))
                else:
                    all_ids_dict[dst_key] = [indices[1:, :].reshape((-1,))]
            else:
                if key in all_ids_dict:
                    all_ids_dict[key].append(indices)
                else:
                    all_ids_dict[key] = [indices]
    counter = {key: 0 for key in all_ids_dict}
    for key, node_ids in all_ids_dict.items():
        uniq_node_ids_dict[key], all_ids_dict[key] = merge_nodes(node_ids)
    for ele_dict in data:
        new_idx_dict = dict()
        for key, indices in ele_dict.items():
            if isinstance(key, tuple):
                src_key, dst_key = key
                src_new_indices = all_ids_dict[src_key][counter[src_key]].reshape(indices[0:1, :].shape)
                dst_new_indices = all_ids_dict[dst_key][counter[dst_key]].reshape(indices[1:, :].shape)
                new_idx_dict[key] = np.concatenate([src_new_indices, dst_new_indices], axis=0)
                counter[src_key] += 1
                counter[dst_key] += 1
            else:
                new_idx_dict[key] = all_ids_dict[key][counter[key]]
                counter[key] += 1
        new_idx_dict_l.append(new_idx_dict)
    return uniq_node_ids_dict, new_idx_dict_l


def empty_as_zero(l, dtype):
    return [ele.astype(dtype) if ele.size > 0 else np.zeros(shape=(1,), dtype=dtype) for ele in l]



class NodeFeatures(object):
    """A simple wrapper for node features/states

    """
    def __init__(self, data, node_ids):
        """Initialize the NodeFeature object

        Parameters
        ----------
        data : mx.nd.ndarray
            Shape (#Node, C)
        node_ids : np.ndarray
            Shape (#Node)
        """
        self._ctx = data.context
        self.data = data
        self.node_ids = node_ids
        self._node_id_rmap = None

    def __repr__(self):
        stream = io.StringIO()
        sprint = lambda *args: print(*args, file=stream)
        sprint('NodeFeatures(')
        sprint('data=')
        sprint(self.data)
        sprint('node_ids=')
        with np.printoptions(precision=3, suppress=True):
            sprint(self.node_ids)
        sprint(')')
        return stream.getvalue()

    def take_by_id(self, sel_node_ids):
        if self._node_id_rmap is None:
            self._node_id_rmap = NodeIDRMapFast(self.node_ids)
        node_inds = mx.nd.array(self._node_id_rmap[sel_node_ids], dtype=np.int32, ctx=self._ctx)
        return NodeFeatures(mx.nd.take(self.data, node_inds, axis=0), sel_node_ids)


class CSRMat(object):
    """A simple wrapper of the CSR Matrix. We can view it as a bipartite graph

    Apart from the traditoinal CSR format, we use two additional arrays: row_ids and col_ids
     to track the original ids of the row/col indices

    We use the C++ API to accelerate the speed if possible
    """
    def __init__(self, end_points, ind_ptr, row_ids, col_ids, values=None, multi_link=None,
                 force_contiguous=True):
        """Initialize the CSRMat

        Parameters
        ----------
        end_points : np.ndarray
            The end_points of the edges. shape (nnz,)
        ind_ptr : np.ndarray
            The starting point in end_points
        row_ids : np.ndarray
        col_ids : np.ndarray
        values : np.ndarray
            Values on the edge
        multi_link : None or list-like object
            The multi-link structure of the csr matrix. This indicates the possible values
             of the edges.
            For example, there are 3 possible ratings, 0.5, 1.0, 1.5 between user and item,
             we can tell CSRMat about this by setting
             ```graph = CSRMat(multi_link=[0.5, 1.0, 1.5])```
        force_contiguous : bool
            Whether to force the end_points, ind_ptr and other elements as contiguous arrays
        """
        assert ind_ptr[0] == 0 and ind_ptr[-1] == end_points.shape[0]
        self.end_points = end_points
        self.ind_ptr = ind_ptr
        self.values = np.ones(shape=self.end_points.shape, dtype=np.float32) if values is None\
            else values.astype(np.float32)
        self.multi_link = np.sort(multi_link) if multi_link is not None else None
        self.row_ids = row_ids
        self.col_ids = col_ids
        assert self.ind_ptr.size == len(self.row_ids) + 1
        if force_contiguous:
            self.end_points = np.ascontiguousarray(self.end_points, dtype=np.int32)
            self.ind_ptr = np.ascontiguousarray(self.ind_ptr, dtype=np.int32)
            if self.values is not None:
                self.values = np.ascontiguousarray(self.values, dtype=np.float32)
            self.row_ids = np.ascontiguousarray(self.row_ids, dtype=np.int32)
            self.col_ids = np.ascontiguousarray(self.col_ids, dtype=np.int32)
        self._node_pair_indices = None
        self._node_pair_ids = None
        self._row_id_rmap = NodeIDRMapFast(self.row_ids)
        self._col_id_rmap = NodeIDRMapFast(self.col_ids)

        self._cached_spy_csr = None
        self._cached_row_degrees = None
        self._cached_col_degrees = None
        self._cached_support = dict()

    def save_edges_txt(self, fname):
        with open(fname, 'w') as f:
            for row_id, col_id, value in zip(self.node_pair_ids[0],
                                             self.node_pair_ids[1],
                                             self.values):
                f.write('{}\t{}\t{:g}\n'.format(row_id, col_id, value))

    def to_spy(self):
        """Convert to the scipy csr matrix

        Returns
        -------
        ret : sp.csr_matrix
        """
        if self._cached_spy_csr is None:
            self._cached_spy_csr = sp.csr_matrix((self.values, self.end_points, self.ind_ptr),
                                                 shape=(self.row_ids.size, self.col_ids.size))
        return self._cached_spy_csr

    @staticmethod
    def from_spy(mat):
        """

        Parameters
        ----------
        mat : sp.csr_matrix

        Returns
        -------
        ret : CSRMat
        """
        return CSRMat(end_points=mat.indices,
                      ind_ptr=mat.indptr,
                      row_ids=np.arange(mat.shape[0], dtype=np.int32),
                      col_ids=np.arange(mat.shape[1], dtype=np.int32),
                      values=mat.data.astype(np.float32),
                      force_contiguous=True)

    @property
    def size(self):
        return self.end_points.size

    @property
    def nnz(self):
        return self.values.size

    @property
    def shape(self):
        return self.row_ids.size, self.col_ids.size

    @property
    def node_pair_indices(self):
        """ Return row & col indices of the edges

        Returns
        -------
        ret : np.ndarray
            Shape (2, TOTAL_EDGE_NUM)
            each has row, col
        """
        if self._node_pair_indices is None:
            self._node_pair_indices =\
                np.stack([_gen_edge_row_indices_by_indptr(self.ind_ptr, self.nnz),
                          self.end_points], axis=0)
        return self._node_pair_indices

    @property
    def node_pair_ids(self):
        """ Return row & col ids of the edges

        Returns
        -------
        ret : np.ndarray
            Shape (2, TOTAL_EDGE_NUM)
            each has row, col
        """
        if self._node_pair_ids is None:
            node_pair_indices = self.node_pair_indices
            self._node_pair_ids = np.stack([self.row_ids[node_pair_indices[0]],
                                            self.col_ids[node_pair_indices[1]]], axis=0)
        return self._node_pair_ids

    @property
    def row_degrees(self):
        if self._cached_row_degrees is None:
            self._cached_row_degrees = np.ascontiguousarray(self.ind_ptr[1:] - self.ind_ptr[:-1])
        return self._cached_row_degrees

    @property
    def col_degrees(self):
        if self._cached_col_degrees is None:
            self._cached_col_degrees = np.zeros(shape=len(self.col_ids), dtype=np.int32)
            uniq_col_indices, cnt = unordered_unique(self.end_points.astype(np.int32), return_counts=True)
            self._cached_col_degrees[uniq_col_indices] = cnt
        return self._cached_col_degrees

    def get_support(self, symm=True):
        key = symm
        if key in self._cached_support:
            return self._cached_support[key]
        else:
            if symm:
                col_degrees = self.col_degrees
            else:
                col_degrees = np.zeros(shape=self.col_ids.shape, dtype=np.int32)
            support = _graph_sampler.get_support(self.row_degrees.astype(np.int32),
                                                 col_degrees,
                                                 self.ind_ptr.astype(np.int32),
                                                 self.end_points.astype(np.int32),
                                                 int(symm))
            self._cached_support[key] = support
            return support

    def row_id_to_ind(self, node_ids):
        """Maps node ids back to row indices in the CSRMat

        Parameters
        ----------
        node_ids : np.ndarray or list or tuple or int

        Returns
        -------
        ret : np.ndarray
        """
        # if isinstance(node_ids, (np.ndarray, list, tuple)):
        #     return np.array(list(map(lambda ele: self._row_id_reverse_mapping[ele], node_ids)),
        #                     dtype=np.int32)
        # else:
        return self._row_id_rmap[node_ids]

    def col_id_to_ind(self, node_ids):
        """Maps node ids back to col indices in the CSRMat

        Parameters
        ----------
        node_ids : np.ndarray or list or tuple or int

        Returns
        -------
        ret : np.ndarray
        """
        # if isinstance(node_ids, (np.ndarray, list, tuple)):
        #     return np.array(list(map(lambda ele: self._col_id_reverse_mapping[ele], node_ids)),
        #                     dtype=np.int32)
        # else:
        return self._col_id_rmap[node_ids]

    def save(self, fname):
        if self.multi_link is None:
            return np.savez_compressed(fname,
                                       row_ids=self.row_ids,
                                       col_ids=self.col_ids,
                                       values=self.values,
                                       end_points=self.end_points,
                                       ind_ptr=self.ind_ptr)
        else:
            return np.savez_compressed(fname,
                                       row_ids=self.row_ids,
                                       col_ids=self.col_ids,
                                       values=self.values,
                                       end_points=self.end_points,
                                       ind_ptr=self.ind_ptr,
                                       multi_link=self.multi_link)

    @staticmethod
    def load(fname):
        data = np.load(fname)
        multi_link = None if 'multi_link' not in data else data['multi_link'][:]
        return CSRMat(row_ids=data['row_ids'][:],
                      col_ids=data['col_ids'][:],
                      values=data['values'][:],
                      multi_link=multi_link,
                      end_points=data['end_points'][:],
                      ind_ptr=data['ind_ptr'][:])

    def submat(self, row_indices=None, col_indices=None):
        """Get the submatrix of the corresponding row/col indices

        Parameters
        ----------
        row_indices : np.ndarray or None
        col_indices : np.ndarray or None

        Returns
        -------
        ret : CSRMat
        """
        if row_indices is None:
            row_indices = None
        else:
            if not isinstance(row_indices, np.ndarray):
                row_indices = np.array([row_indices], dtype=np.int32)
            else:
                row_indices = np.ascontiguousarray(row_indices, dtype=np.int32)
        if col_indices is None:
            col_indices = None
        else:
            if not isinstance(col_indices, np.ndarray):
                col_indices = np.array([col_indices], dtype=np.int32)
            else:
                col_indices = np.ascontiguousarray(col_indices, dtype=np.int32)
        print("col_indices", col_indices.size)
        print("self.col_ids", self.col_ids.size)
        print("ind_ptr", self.ind_ptr.size-1)
        dst_end_points, dst_values, dst_ind_ptr, dst_row_ids, dst_col_ids\
            = _graph_sampler.csr_submat(np.ascontiguousarray(self.end_points.astype(np.int32),
                                                             dtype=np.int32),
                                        np.ascontiguousarray(self.values),
                                        np.ascontiguousarray(self.ind_ptr.astype(np.int32), dtype=np.int32),
                                        np.ascontiguousarray(self.row_ids, dtype=np.int32),
                                        np.ascontiguousarray(self.col_ids, dtype=np.int32),
                                        row_indices,
                                        col_indices)
        return CSRMat(end_points=dst_end_points,
                      ind_ptr=dst_ind_ptr,
                      row_ids=dst_row_ids,
                      col_ids=dst_col_ids,
                      values=dst_values,
                      multi_link=self.multi_link)

    def submat_by_id(self, row_ids=None, col_ids=None):
        row_indices = None if row_ids is None else self.row_id_to_ind(row_ids)
        col_indices = None if col_ids is None else self.col_id_to_ind(col_ids)
        return self.submat(row_indices, col_indices)

    def sample_submat(self, row_indices=None, ncols=5):
        """ Sample a random number of columns WITHOUT replacement for each row and form a new csr_mat

        Parameters
        ----------
        row_indices : np.ndarray or None
        ncols : int or None
            None means to sample all columns

        Returns
        -------
        ret : CSRMat
        """
        if ncols is None:
            return self.submat(row_indices=row_indices, col_indices=None)
        if row_indices is None:
            row_indices = np.arange(self.shape[0], dtype=np.int32)
        sampled_indices, dst_ind_ptr \
            = _graph_sampler.random_sample_fix_neighbor(self.ind_ptr.astype(np.int32),
                                                        row_indices.astype(np.int32),
                                                        ncols)
        dst_end_points = self.end_points[sampled_indices]
        uniq_col_indices, dst_end_points = unordered_unique(dst_end_points, return_inverse=True)
        return CSRMat(end_points=dst_end_points,
                      ind_ptr=dst_ind_ptr,
                      row_ids=self.row_ids[row_indices],
                      col_ids=self.col_ids[uniq_col_indices],
                      values=self.values[sampled_indices],
                      multi_link=self.multi_link)

    def sample_submat_by_id(self, row_ids=None, ncols=5):
        """ Sample a random number of columns WITHOUT replacement for each row and form a new csr_mat. This function
        select the rows by the row_ids

        Parameters
        ----------
        row_ids : np.ndarray or None
        ncols : int or None

        Returns
        -------
        ret : CSRMat
        """
        return self.sample_submat(self.row_id_to_ind(row_ids), ncols)

    @property
    def T(self):
        new_csr_mat = self.to_spy().T.tocsr()
        return CSRMat(end_points=new_csr_mat.indices,
                      ind_ptr=new_csr_mat.indptr,
                      values=new_csr_mat.data,
                      row_ids=self.col_ids,
                      col_ids=self.row_ids,
                      multi_link=self.multi_link)

    def fetch_edges_by_ind(self, node_pair_indices):
        """Select edge values based on the indices of the node pairs

        Parameters
        ----------
        node_pair_ind : np.ndarray
            Shape (2, SEL_EDGE_NUM)

        Returns
        -------
        ret : np.ndarray
            Shape (SEL_EDGE_NUM,)
        """
        ### TODO change .A1? to data
        ret = self.to_spy()[node_pair_indices[0, :], node_pair_indices[1, :]]
        if ret.size == 0:
            return np.ndarray([])
        else:
            return np.array(ret).reshape((-1,))

    def fetch_edges_by_id(self, node_pair_ids):
        """Select edge values based on the ids of node pairs

        Parameters
        ----------
        node_pair_ids : np.ndarray
            Shape (2, SEL_EDGE_NUM)

        Returns
        -------
        ret : np.ndarray
            Shape (SEL_EDGE_NUM,)
        """
        return self.fetch_edges_by_ind(np.stack([self.row_id_to_ind(node_pair_ids[0]),
                                                 self.col_id_to_ind(node_pair_ids[1])]))

    def remove_edges_by_ind(self, node_pair_indices):
        """

        Parameters
        ----------
        node_pair_indices : np.ndarray
            Shape (2, REMOVE_EDGE_NUM)

        Returns
        -------
        ret : CSRMat
            The new CSRMat after removing these edges
        """
        row_indices, col_indices = np.ascontiguousarray(node_pair_indices[0], dtype=np.int32),\
                                   np.ascontiguousarray(node_pair_indices[1], dtype=np.int32)
        dst_end_points, dst_values, dst_indptr =\
            _graph_sampler.remove_edges_by_indices(self.end_points.astype(np.int32),
                                                   self.values,
                                                   self.ind_ptr.astype(np.int32),
                                                   row_indices.astype(np.int32),
                                                   col_indices.astype(np.int32))
        return CSRMat(end_points=dst_end_points,
                      ind_ptr=dst_indptr,
                      values=dst_values,
                      row_ids=self.row_ids,
                      col_ids=self.col_ids,
                      multi_link=self.multi_link,
                      force_contiguous=True)

    def remove_edges_by_id(self, node_pair_ids):
        """

        Parameters
        ----------
        node_pair_ids : np.ndarray
            Shape (2, REMOVE_EDGE_NUM)

        Returns
        -------
        ret : CSRMat
            The new CSRMat after removing these edges
        """
        row_ids, col_ids = node_pair_ids[0], node_pair_ids[1]
        return self.remove_edges_by_ind(np.stack((self.row_id_to_ind(row_ids),
                                                  self.col_id_to_ind(col_ids))))

    def sample_neighbors(self, src_ids=None, symm=True, use_multi_link=True, num_neighbors=None):
        """ Fetch the ids of the columns that are connected to the src_node

        Parameters
        ----------
        src_ids : np.ndarray or None
            None indicates to select all the src_ids. It will have the same value as csr_mat.row_ids
        symm : bool
            Whether to use the symmetric formulation to calculate the support
        use_multi_link : bool
            Whether to sample multiple edge_values
        num_neighbors : int or None
            Number of neighbors to sample.
             None or a negative number indicates to sample all neighborhoods

        Returns
        -------
        end_points_ids : list or np.ndarray
            - use_multi_link is False:
                Ids of the neighboring node that are connected to the source nodes.
            - use_multi_link is True:
                The output will be a list. The i-th element will contain the dst_ids that has the
                 i-th possible edge values with the src_ids
        edge_values : list or np.ndarray
            - use_multi_link is False:
                Edge values between the chosen dst_ids and src_ids
            - use_multi_link is True:
                List of edge values corresponding to dst_ids
        ind_ptr : list or np.ndarray
            - use_multi_link is False:
                dst_ids[dst_ind_ptr[i]:dst_ind_ptr[i+1]] are connected to src_ids[i]
            - use_multi_link is True:
                List of ind_ptrs corresponding to dst_ids
        support : list or np.ndarray
            - use_multi_link is False:
                The support value of the edges.
                If `symm` is True, it's \sqrt(D(src) D(dst))
                Otherwise, it's D(src)
            - use_multi_link is True
                List of support corresponding to dst_ids
        """

        if src_ids is not None:
            src_inds = self.row_id_to_ind(src_ids)
        else:
            src_inds = np.arange(self.shape[0], dtype=np.int32)
        if num_neighbors is None:
            num_neighbors = -1  # The C++ implementation will sample all possible neighbors if num_neighbors is < 0.
        sampled_indices, dst_ind_ptr \
            = _graph_sampler.random_sample_fix_neighbor(self.ind_ptr.astype(np.int32),
                                                        src_inds.astype(np.int32),
                                                        num_neighbors)
        dst_end_points_ids = np.take(self.col_ids, np.take(self.end_points, sampled_indices))
        edge_values = np.take(self.values, sampled_indices)
        support = np.take(self.get_support(symm), sampled_indices)
        if not use_multi_link:
            return dst_end_points_ids, edge_values, dst_ind_ptr, support
        else:
            assert self.multi_link is not None
            split_indices, dst_ind_ptr_l = _graph_sampler.multi_link_split(edge_values, dst_ind_ptr,
                                                                           self.multi_link)
            dst_end_points_ids_l = []
            edge_values_l = []
            support_l = []
            for sel_idx in split_indices:
                ele_dst_end_points_ids = np.take(dst_end_points_ids, sel_idx)
                ele_edge_values = np.take(edge_values, sel_idx)
                ele_support = np.take(support, sel_idx)
                dst_end_points_ids_l.append(ele_dst_end_points_ids)
                edge_values_l.append(ele_edge_values)
                support_l.append(ele_support)
            return dst_end_points_ids_l, edge_values_l, dst_ind_ptr_l, support_l


    def check_consistency(self):
        for i in range(len(self.ind_ptr) - 1):
            ele_end_points = self.end_points[self.ind_ptr[i]:self.ind_ptr[i+1]]
            if np.unique(ele_end_points).shape != ele_end_points.shape:
                raise ValueError('Found duplicates in end_points, i={}'.format(i))

    def issubmat(self, large_mat):
        """ Check whether the matrix is a submatrix of large_mat

        Parameters
        ----------
        large_mat : CSRMat

        Returns
        -------
        ret : bool
            True or False
        """
        for i, row_id in enumerate(self.row_ids):
            lmat_row_idx = large_mat.row_id_to_ind(row_id)
            all_end_points = large_mat.end_points[large_mat.ind_ptr[lmat_row_idx]:large_mat.ind_ptr[lmat_row_idx + 1]]
            all_end_point_ids = large_mat.col_ids[all_end_points]
            all_values = large_mat.values[large_mat.ind_ptr[lmat_row_idx]:large_mat.ind_ptr[lmat_row_idx + 1]]
            all_end_point_ids_value = {eid: val for eid, val in zip(all_end_point_ids, all_values)}
            sub_end_points_ids = self.col_ids[self.end_points[self.ind_ptr[i]:self.ind_ptr[i + 1]]]
            sub_values = self.values[self.ind_ptr[i]:self.ind_ptr[i + 1]]
            if not set(sub_end_points_ids).issubset(set(all_end_point_ids)):
                return False
            for eid, val in zip(sub_end_points_ids, sub_values):
                if all_end_point_ids_value[eid] != val:
                    return False
        if (large_mat.multi_link is None and self.multi_link is not None) or\
                (self.multi_link is None and large_mat.multi_link is not None):
            return False
        if len(large_mat.multi_link) != len(self.multi_link):
            return False
        for lhs, rhs in zip(large_mat.multi_link, self.multi_link):
            if lhs != rhs:
                return False
        return True

    def summary(self):
        print(self)

    def __repr__(self):
        info_str = "CSRMat:" + \
                   "\n   Row={}, Col={}, NNZ={}".format(self.row_ids.size,
                                                        self.col_ids.size,
                                                        self.end_points.size)
        if self.multi_link is not None:
            info_str += '\n   Multi Link={}'.format(self.multi_link)
        return info_str


class HeterGraph(object):
    def __init__(self, features, node_ids=None, csr_mat_dict=None, **kwargs):
        """

        Parameters
        ----------
        features : dict
            {node_key : np.ndarray (#node, fea_dim)}
        node_ids : dict or None
            {node_key : np.ndarray (#node, )}
        csr_mat_dict : dict
            The connection between two types of nodes.
            Contains: {(node_key1, node_key2) : CSRMat}
            IMPORTANT! We allow node_key1 == node_key2, which indicates self-link, e.g., user-user
        """
        self.features = features
        self.node_ids = node_ids
        if self.node_ids is None:
            self.node_ids = {}
            for key, features in self.features.items():
                self.node_ids[key] = np.arange(features.shape[0], dtype=np.int32)
        else:
            assert sorted(self.features.keys()) == sorted(self.node_ids.keys())
            for k, node_ids in self.node_ids.items():
                assert node_ids.shape[0] == self.features[k].shape[0]

        # Generate node_id_to_ind mappings
        self._node_id_rmaps = dict()
        for k, node_ids in self.node_ids.items():
            self._node_id_rmaps[k] = NodeIDRMapFast(node_ids)
        if 'meta_graph' not in kwargs:
            self.meta_graph = dict()
            for key in self.features:
                self.meta_graph[key] = dict()
        else:
            self.meta_graph = kwargs['meta_graph']
        if '_csr_matrices' not in kwargs:
            self.csr_matrices = dict()
            for key in self.features:
                self.csr_matrices[key] = dict()
            for (node_key1, node_key2), mat in csr_mat_dict.items():
                assert node_key1 in self.meta_graph,\
                    '{} not found!, meta_graph_nodes={}'.format(node_key1, self.meta_graph)
                assert node_key2 in self.meta_graph, \
                    '{} not found!, meta_graph_nodes={}'.format(node_key2, self.meta_graph)
                self.meta_graph[node_key1][node_key2] = 1
                self.meta_graph[node_key2][node_key1] = 1
                self.csr_matrices[node_key1][node_key2] = mat
                if node_key2 != node_key1:
                    self.csr_matrices[node_key2][node_key1] = mat.T
                else:
                    assert mat.shape[0] == mat.shape[1],\
                        '{} -> {} must be a square matrix'.format(node_key1, node_key2)
        else:
            self.csr_matrices = kwargs['_csr_matrices']

    def check_continous_node_ids(self):
        for key, ele in self.node_ids.items():
            np.testing.assert_allclose(np.sort(ele), np.arange(len(ele), dtype=np.int32))

    def features_by_id(self, key, node_ids):
        """ Select a subset of the features indexed by the given node_ids

        Parameters
        ----------
        key : str
            Name of the node
        node_ids : np.ndarray
            IDs of the nodes to select

        Returns
        -------
        sub_features : np.ndarray
            Output
        """
        return self.features[key][self._node_id_rmaps[key][node_ids]]

    @property
    def node_names(self):
        return self.features.keys()

    @property
    def node_id_rmaps(self):
        return self._node_id_rmaps

    def get_multi_link_structure(self):
        multi_link_structure = {}
        for src_key in self.csr_matrices:
            for dst_key, mat in self.csr_matrices[src_key].items():
                multi_link_structure[(src_key, dst_key)] = \
                    len(mat.multi_link) if mat.multi_link is not None else None
        return multi_link_structure

    def save(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(os.path.join(dir_name, 'meta_graph.json'), 'w') as f:
            json.dump(self.meta_graph, f)
        for key, features in self.features.items():
            np.savez_compressed(os.path.join(dir_name, '{}.npz'.format(key)),
                                node_ids=self.node_ids[key],
                                features=features.astype(np.float32))
        cached_edge = set()
        for k1 in self.meta_graph:
            for k2 in self.meta_graph[k1]:
                if (k1, k2) in cached_edge:
                    continue
                cached_edge.add((k1, k2))
                cached_edge.add((k2, k1))
                self.csr_matrices[k1][k2].save(os.path.join(dir_name,
                                                            '{}_{}_csr.npz'.format(k1, k2)))

    def node_id_to_ind(self, key, node_ids):
        return self._node_id_rmaps[key][node_ids]

    def fetch_edges_by_id(self, src_key, dst_key, node_pair_ids):
        """

        Parameters
        ----------
        src_key : str
        dst_key : str
        node_pair_ids : np.ndarray
            Shape (2, SEL_EDGE_NUM)

        Returns
        -------
        edge_values : np.ndarray
        """
        return self.csr_matrices[src_key][dst_key].fetch_edges_by_id(node_pair_ids)

    def fetch_edges_by_ind(self, src_key, dst_key, node_pair_indices):
        """

        Parameters
        ----------
        src_key : str
        dst_key : str
        node_pair_indices : np.ndarray
            Shape (2, SEL_EDGE_NUM)

        Returns
        -------
        edge_values : np.ndarray
        """
        return self.csr_matrices[src_key][dst_key].fetch_edges_by_ind(node_pair_indices)

    def remove_edges_by_id(self, src_key, dst_key, node_pair_ids):
        """ Get a new heterogenous graph after the given edges are removed

        Parameters
        ----------
        src_key : str
        dst_key : str
        node_pair_ids : np.ndarray
            Shape (2, REMOVE_EDGE_NUM)

        Returns
        -------
        ret : HeterGraph
        """
        new_csr_matrices = _shallow_copy_stacked_dict(self.csr_matrices)
        new_csr_matrices[src_key][dst_key] =\
            self.csr_matrices[src_key][dst_key].remove_edges_by_id(node_pair_ids)
        new_csr_matrices[dst_key][src_key] = \
            self.csr_matrices[dst_key][src_key].remove_edges_by_id(np.flipud(node_pair_ids))
        return HeterGraph(features=self.features,
                          node_ids=self.node_ids,
                          meta_graph=self.meta_graph,
                          _csr_matrices=new_csr_matrices)

    def remove_edges_by_ind(self, src_key, dst_key, node_pair_indices):
        """

        Parameters
        ----------
        src_key : str
        dst_key : str
        node_pair_indices : np.ndarray

        Returns
        -------
        ret : HeterGraph
        """
        # IMPORTANT! We cannot use copy in the following. Because we have a
        # nested dictionary structure, directly call .copy() will share the reference.
        new_csr_matrices = _shallow_copy_stacked_dict(self.csr_matrices)
        new_csr_matrices[src_key][dst_key] =\
            self.csr_matrices[src_key][dst_key].remove_edges_by_ind(node_pair_indices)
        new_csr_matrices[dst_key][src_key] = \
            self.csr_matrices[dst_key][src_key].remove_edges_by_ind(np.flipud(node_pair_indices))
        return HeterGraph(features=self.features,
                          node_ids=self.node_ids,
                          meta_graph=self.meta_graph,
                          _csr_matrices=new_csr_matrices)

    def sel_subgraph_by_id(self, key, node_ids):
        """ Select the given nodes from the heterogenous graph and return a new graph

        Parameters
        ----------
        key : str
        node_ids : np.ndarray

        Returns
        -------
        ret : HeterGraph
        """
        new_features = self.features.copy()
        new_node_ids = self.node_ids.copy()
        new_csr_matrices = _shallow_copy_stacked_dict(self.csr_matrices)
        new_features[key] = np.take(self.features[key], self.node_id_to_ind(key, node_ids), axis=0)
        new_node_ids[key] = node_ids
        for dst_key, csr_mat in self.csr_matrices[key].items():
            if dst_key != key:
                new_csr_matrices[key][dst_key] = csr_mat.submat_by_id(row_ids=node_ids, col_ids=None)
                new_csr_matrices[dst_key][key] = \
                    self.csr_matrices[dst_key][key].submat_by_id(row_ids=None, col_ids=node_ids)
            else:
                new_csr_matrices[key][dst_key] = csr_mat.submat_by_id(row_ids=node_ids,
                                                                      col_ids=node_ids)

        return HeterGraph(features=new_features,
                          node_ids=new_node_ids,
                          meta_graph=self.meta_graph,
                          _csr_matrices=new_csr_matrices)

    def gen_nd_features(self, ctx):
        """Copy the features to the given mxnet context

        Parameters
        ----------
        ctx : mx.Context

        Returns
        -------
        nd_features : dict
        """
        return {key: NodeFeatures(mx.nd.array(self.features[key], ctx=ctx, dtype=np.float32),
                                  self.node_ids[key]) for key in self.features}

    def check_consistency(self):
        _checked_edges = set()
        for src_key in self.meta_graph:
            for dst_key in self.meta_graph[src_key]:
                if (src_key, dst_key) in _checked_edges:
                    continue
                _checked_edges.add((src_key, dst_key))
                _checked_edges.add((dst_key, src_key))
                src_to_dst = self.csr_matrices[src_key][dst_key]
                dst_to_src = self.csr_matrices[dst_key][src_key]
                assert src_to_dst.shape\
                       == (self.features[src_key].shape[0], self.features[dst_key].shape[0])\
                       == (dst_to_src.shape[1], dst_to_src.shape[0])
                npt.assert_allclose(src_to_dst.fetch_edges_by_id(np.flipud(dst_to_src.node_pair_ids)),
                                    dst_to_src.values)
                npt.assert_allclose(dst_to_src.fetch_edges_by_id(np.flipud(src_to_dst.node_pair_ids)),
                                    src_to_dst.values)
                src_to_dst.check_consistency()
                dst_to_src.check_consistency()

    @staticmethod
    def load(dir_name, fea_normalize=False):
        with open(os.path.join(dir_name, 'meta_graph.json')) as f:
            meta_graph = json.load(f)
        features = {}
        node_ids = {}
        csr_mat_dict = {}
        cached_edge = set()
        for k1 in meta_graph:
            dat = np.load(os.path.join(dir_name, '{}.npz'.format(k1)))
            fea = dat['features'][:]
            if fea_normalize and fea is not None and fea.ndim == 2:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.fit(fea)
                fea = scaler.transform(fea)
            features[k1] = fea
            node_ids[k1] = dat['node_ids'][:]
            for k2 in meta_graph[k1]:
                if (k1, k2) in cached_edge:
                    continue
                cached_edge.add((k1, k2))
                cached_edge.add((k2, k1))
                found = False
                for tk1, tk2 in [(k1, k2), (k2, k1)]:
                    fname = os.path.join(dir_name, '{}_{}_csr.npz'.format(tk1, tk2))
                    if os.path.exists(fname):
                        assert not found
                        csr_mat = CSRMat.load(fname)
                        csr_mat_dict[(tk1, tk2)] = csr_mat
                        found = True
                assert found, "k1={}, k2={} not found!".format(k1, k2)
        return HeterGraph(features=features,
                          node_ids=node_ids,
                          csr_mat_dict=csr_mat_dict)

    def __getitem__(self, pair_keys):
        """

        Parameters
        ----------
        pair_keys : list-like object
            The src_key, dst_key pair

        Returns
        -------
        ret : CSRMat
            The resulting bipartite graph
        """
        assert len(pair_keys) == 2
        return self.csr_matrices[pair_keys[0]][pair_keys[1]]

    def __repr__(self):
        stream = io.StringIO()
        print("  --------------------------", file=stream)
        meta_graph_npy = np.zeros(shape=(len(self.meta_graph), len(self.meta_graph)),
                                  dtype=np.int32)
        node_keys = self.meta_graph.keys()
        for key in node_keys:
            print('{}, num={}, feature dim={}'.format(key, self.features[key].shape[0],
                                                      self.features[key].shape[1]), file=stream)
        node_key_map = {ele: i for i, ele in enumerate(node_keys)}
        for k1 in self.meta_graph:
            for k2 in self.meta_graph[k1]:
                meta_graph_npy[node_key_map[k1]][node_key_map[k2]] = self.csr_matrices[k1][k2].nnz
        mgraph_df = pd.DataFrame(meta_graph_npy, index=node_keys, columns=node_keys)
        print('meta-graph=', file=stream)
        print(mgraph_df, file=stream)
        print('multi-link=', file=stream)
        for k1 in self.meta_graph:
            for k2 in self.meta_graph[k1]:
                if self.csr_matrices[k1][k2].multi_link is not None:
                    print('{} --> {}: {}'.format(k1, k2, self.csr_matrices[k1][k2].multi_link),
                          file=stream)
        return stream.getvalue()

    def summary(self):
        print(self)
