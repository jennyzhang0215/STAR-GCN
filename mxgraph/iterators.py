import io
import numpy as np
from mxgraph.graph import HeterGraph, CSRMat

class NegEdgeGenerator(object):
    def __init__(self, rng, csr_mat):
        """

        Parameters
        ----------
        rng : np.random.RandomState
        csr_mat : CSRMat
        """
        super(NegEdgeGenerator, self).__init__()
        self._rng = rng
        self._csr_mat = csr_mat
        self._nrows, self._ncols = csr_mat.shape
        self._row_degrees = csr_mat.row_degrees
        self._col_degrees = csr_mat.col_degrees
        self._row_edge_remaps = [dict() for _ in range(self._nrows)]
        row_edge_remaps_cnt = [0 for _ in range(self._nrows)]
        self._col_edge_remaps = [dict() for _ in range(self._ncols)]
        col_edge_remaps_cnt = [0 for _ in range(self._ncols)]
        all_edges = csr_mat.node_pair_indices
        for row_ind, col_ind in zip(all_edges[0], all_edges[1]):
            # Build row edge remaps
            cnt = row_edge_remaps_cnt[row_ind]
            val = self._row_edge_remaps[row_ind].get(col_ind, col_ind)
            swap_val = self._row_edge_remaps[row_ind].get(cnt, cnt)
            self._row_edge_remaps[row_ind][cnt] = val
            self._row_edge_remaps[row_ind][col_ind] = swap_val
            row_edge_remaps_cnt[row_ind] += 1
            # Build col edge remaps
            cnt = col_edge_remaps_cnt[col_ind]
            val = self._col_edge_remaps[col_ind].get(row_ind, row_ind)
            swap_val = self._col_edge_remaps[col_ind].get(cnt, cnt)
            self._col_edge_remaps[col_ind][cnt] = val
            self._col_edge_remaps[col_ind][row_ind] = swap_val
            col_edge_remaps_cnt[col_ind] += 1

        for lhs, rhs in zip(row_edge_remaps_cnt, self._row_degrees):
            assert lhs == rhs
        for lhs, rhs in zip(col_edge_remaps_cnt, self._col_degrees):
            assert lhs == rhs
        self._row_probs = self._ncols - self._row_degrees
        self._row_probs = self._row_probs.astype(np.float32) / self._row_probs.sum()

    def rand_neg_col_with_row(self, row_ind):
        val = self._rng.randint(self._row_degrees[row_ind], self._ncols)
        return self._row_edge_remaps[row_ind].get(val, val)

    def rand_neg_row_with_col(self, col_ind):
        val = self._rng.randint(self._col_degrees[col_ind], self._nrows)
        return self._col_edge_remaps[col_ind].get(val, val)

    def rand_neg_edges(self, n):
        row_inds = self._rng.choice(self._nrows, n, replace=True, p=self._row_probs).astype(np.int32)
        col_inds = np.array([self.rand_neg_col_with_row(ele) for ele in row_inds],
                            dtype=np.int32)
        return row_inds, col_inds

    def gen(self, pos_edges, neg_sample_type, neg_ratio):
        """

        Parameters
        ----------
        pos_edges : np.ndarray
            Shape (2, #Positive Edges)
        neg_sample_type : str
            Can be either 'same_node' or 'all'
        neg_ratio : float
            Only used when neg_sample_type is all

        Returns
        -------
        neg_edges : np.ndarray
            Shape (2, #Negative Edges)
        """
        # Conver ID representation to ind representation
        pos_edges = np.stack([self._csr_mat.row_id_to_ind(pos_edges[0, :]),
                              self._csr_mat.col_id_to_ind(pos_edges[1, :])])
        if neg_sample_type == 'same_node':
            neg_rows = []
            neg_cols = []
            for idx in range(pos_edges.shape[1]):
                pos_row, pos_col = pos_edges[:, idx]
                keep_row = self._rng.randint(2)
                if keep_row:
                    if self._row_degrees[pos_row] < self._ncols:
                        neg_rows.append(pos_row)
                        neg_cols.append(self.rand_neg_col_with_row(pos_row))
                    else:
                        neg_row, neg_col = self.rand_neg_edges(1)
                        neg_rows.append(neg_row[0])
                        neg_cols.append(neg_col[0])
                else:
                    if self._col_degrees[pos_col] < self._nrows:
                        neg_rows.append(self.rand_neg_row_with_col(pos_col))
                        neg_cols.append(pos_col)
                    else:
                        neg_row, neg_col = self.rand_neg_edges(1)
                        neg_rows.append(neg_row[0])
                        neg_cols.append(neg_col[0])
            neg_rows = np.array(neg_rows, dtype=np.int32)
            neg_cols = np.array(neg_cols, dtype=np.int32)
        elif neg_sample_type == 'all':
            neg_num = int(np.round(neg_ratio * pos_edges.shape[1]))
            neg_rows, neg_cols = self.rand_neg_edges(neg_num)
        else:
            raise NotImplementedError
        return np.stack([self._csr_mat.row_ids[neg_rows], self._csr_mat.col_ids[neg_cols]])


class DataIterator(object):
    def __init__(self, all_graph, name_user, name_item,
                 is_inductive=False,
                 test_node_pairs=None, valid_node_pairs=None,
                 inductive_key=None,
                 inductive_valid_ids=None, inductive_train_ids=None,
                 embed_P_mask=0.1, embed_p_zero=1.0, embed_p_self=0.0,
                 seed=100):
        """

        Parameters
        ----------
        all_graph : HeterGraph
        name_user : str
        name_item : str
        valid_node_pairs : np.ndarray or None
            The node pairs for validation. It should only be set in the transductive setting.
        test_node_pairs : np.ndarray or None
            The node pairs for validation. It should only be set in the transductive setting.
        inductive_valid_ids : np.ndarray or None
            The node ids of the validation users. It should only be set in the inductive learning setting.
        inductive_train_ids : np.ndarray or None
            The node ids of the train users. It should only be set in the inductive learning setting.
        inductive_key : str (name_item, name_user)
        inductive_test_pairs : np.ndarray or None
            (2, #edges)
        inductive_valid_pairs : np.ndarray or None
            (2, #edges)
        linkpred_valid_ratio : float
            Ratio of the links that are used for validation
        recon_valid_ratio : float
            Ratio of the nodes that are used for
        embed_P_mask : dict or float
            Probability of masking the embeddings.
        embed_p_zero : dict or float
            Probability of setting the embedding to zero.
        embed_p_self : dict or float
            Probability of keep the original embedding
        seed : int or None
            The seed of the random number generator
        """
        self._rng = np.random.RandomState(seed=seed)
        self._all_graph = all_graph
        self._is_inductive = False
        self._name_user = name_user
        self._name_item = name_item

        ### Generate graphs
        ### test_graph is for testing data to aggregate neighbors
        ### val_graph is for validation data to aggregate neighbors
        ### train_graph is for training data to aggregate neighbors, require to remove batch pairs first
        self._test_graph = all_graph.remove_edges_by_id(name_user, name_item, test_node_pairs)
        self._is_inductive = is_inductive
        if not is_inductive:
            self._val_graph = self._test_graph.remove_edges_by_id(name_user, name_item, valid_node_pairs)
            self._train_graph = self._val_graph
        else:
            assert inductive_key is not None and \
                   inductive_train_ids is not None and inductive_valid_ids is not None
            train_val_ids = np.concatenate((inductive_train_ids, inductive_valid_ids)).astype(np.int32)
            self._val_graph = all_graph.sel_subgraph_by_id(inductive_key, train_val_ids). \
                remove_edges_by_id(name_user, name_item, valid_node_pairs)
            self._train_graph = all_graph.sel_subgraph_by_id(inductive_key, inductive_train_ids)

        self._test_node_pairs = test_node_pairs
        self._valid_node_pairs = valid_node_pairs
        self._train_node_pairs = self._train_graph[name_user, name_item].node_pair_ids
        self._train_ratings = self._train_graph[name_user, name_item].values
        self._valid_ratings = self._all_graph.fetch_edges_by_id(src_key=name_user,
                                                                dst_key=name_item,
                                                                node_pair_ids=self._valid_node_pairs)
        self._test_ratings = self._all_graph.fetch_edges_by_id(src_key=name_user,
                                                               dst_key=name_item,
                                                               node_pair_ids=self._test_node_pairs)
        ##############################################################################
        ### split train/val pos edges and sample neg edges for link prediction
        self._train_pos_edges_dict = dict()
        self._neg_edge_generators = dict()
        for src_key in self._train_graph.meta_graph:
            for dst_key in self._train_graph.meta_graph[src_key]:
                if ((src_key, dst_key) == (name_user, name_item)) or ((src_key, dst_key) == (name_item, name_user))\
                        or (dst_key, src_key) in self._train_pos_edges_dict:
                    continue
                csr_mat = self._train_graph[src_key, dst_key]
                all_edges = csr_mat.node_pair_ids
                # permutation_idx = self._rng.permutation(all_edges.shape[1])
                # valid_num = int(np.ceil(all_edges.shape[1] * linkpred_valid_ratio))
                # self._valid_pos_edges_dict[(src_key, dst_key)] = all_edges[:, permutation_idx[:valid_num]]
                # self._valid_neg_edges_dict[(src_key, dst_key)] =\
                #     gen_linkpred_neg_edges(self._rng, csr_mat,
                #                            self._valid_pos_edges_dict[(src_key, dst_key)])
                ### TODO set all training edges as the link prediction candidate
                #self._train_pos_edges_dict[(src_key, dst_key)] = all_edges[:, permutation_idx[valid_num:]]
                self._train_pos_edges_dict[(src_key, dst_key)] = all_edges
                self._neg_edge_generators[(src_key, dst_key)] = NegEdgeGenerator(self._rng, csr_mat)

        ##############################################################################
        ### split train/val node ids for reconstruction loss
        self._recon_train_candidates = dict()
        #self._recon_valid_candidates = dict()
        if isinstance(embed_P_mask, dict):
            self._embed_P_mask = embed_P_mask
        else:
            self._embed_P_mask = {key: embed_P_mask for key in all_graph.meta_graph}
        if isinstance(embed_p_zero, dict):
            self._embed_p_zero = embed_p_zero
        else:
            self._embed_p_zero = {key: embed_p_zero for key in all_graph.meta_graph}
        if isinstance(embed_p_self, dict):
            self._embed_p_self = embed_p_self
        else:
            self._embed_p_self = {key: embed_p_self for key in all_graph.meta_graph}
        for key in self._embed_P_mask:
            assert self._embed_p_zero[key] + self._embed_p_self[key] == 1.0
        self._evaluate_embed_noise_dict = dict()
        for key in self._train_graph.meta_graph:
            self._recon_train_candidates[key] = self._train_graph.node_ids[key]
            self._evaluate_embed_noise_dict[key] = -np.ones(self._all_graph.node_ids[key].shape,
                                                            dtype=np.int32)
            train_node_ids = self._train_graph.node_ids[key]
            print("{}: all_node_num {} v.s. train_node_num {}".format(key, self._all_graph.node_ids[key].size,
                                                                      train_node_ids.size,))
            self._evaluate_embed_noise_dict[key][train_node_ids] = train_node_ids

    @property
    def possible_rating_values(self):
        return self.all_graph[self._name_user, self._name_item].multi_link
    @property
    def evaluate_embed_noise_dict(self):
        return self._evaluate_embed_noise_dict
    @property
    def is_inductive(self):
        return self._is_inductive

    @property
    def all_graph(self):
        return self._all_graph

    @property
    def test_graph(self):
        return self._test_graph

    @property
    def val_graph(self):
        return self._val_graph

    @property
    def train_graph(self):
        return self._train_graph

    def rating_sampler(self, batch_size, segment='train', sequential=None):
        """ Return the sampler for ratings

        Parameters
        ----------
        batch_size : int, -1 means the whole data samples
        segment : str
        sequential : bool or None
            Whether to sample in a sequential manner. If it's set to None, it will be
            automatically determined based on the sampling segment.

        Returns
        -------
        node_pairs : np.ndarray
            Shape (2, #Edges)
        ratings : np.ndarray
            Shape (#Edges,)
        """
        if segment == 'train':
            sequential = False if sequential is None else sequential
            node_pairs, ratings = self._train_node_pairs, self._train_ratings
        elif segment == 'valid':
            sequential = True if sequential is None else sequential
            node_pairs, ratings = self._valid_node_pairs, self._valid_ratings
        elif segment == 'test':
            sequential = True if sequential is None else sequential
            node_pairs, ratings = self._test_node_pairs, self._test_ratings
        else:
            raise NotImplementedError('segment must be in {}, received {}'.format(['train', 'valid', 'test'], segment))
        if batch_size < 0:
            batch_size = node_pairs.shape[1]
        else:
            batch_size = min(batch_size, node_pairs.shape[1])
        if sequential:
            for start in range(0, node_pairs.shape[1], batch_size):
                end = min(start + batch_size, node_pairs.shape[1])
                yield node_pairs[:, start:end], ratings[start:end]
        else:
            while True:
                if batch_size == node_pairs.shape[1]:
                    yield node_pairs, ratings
                else:
                    sel = self._rng.choice(node_pairs.shape[1], batch_size, replace=False)
                    yield node_pairs[:, sel], ratings[sel]

    def recon_nodes_sampler(self, batch_size, segment='train', sequential=False):
        """

        Parameters
        ----------
        batch_size
        segment
        sequential

        Returns
        -------
        embed_noise_dict : dict {node_key: np.ndarray <Shape (#recon_candidates, )>, ..}
        batch_recon_node_ids_dict : dict
        all_masked_node_ids_dict : dict
        """
        assert segment is 'train'
        assert sequential is False
        while True:
            if segment == 'train':
                sequential = False if sequential is None else sequential
                embed_noise_dict = dict()
                recon_node_ids_dict = dict()
                for key, node_ids in self._recon_train_candidates.items():
                    recon_node_num = int(np.ceil(self._embed_P_mask[key] * node_ids.size))
                    perm_node_ids = self._rng.permutation(node_ids)
                    recon_node_ids = perm_node_ids[:recon_node_num]
                    remain_node_ids = perm_node_ids[recon_node_num:]
                    if recon_node_ids.size > 0:
                        recon_node_ids_dict[key] = recon_node_ids
                        mask_type = self._rng.multinomial(1, [self._embed_p_zero[key],
                                                              self._embed_p_self[key]],
                                                          size=recon_node_ids.size)
                        ### nodes unseen in the training graph are masked as -1.0
                        embed_noise = -np.ones(self._all_graph.node_ids[key].shape, dtype=np.int32)
                        embed_noise[remain_node_ids] = remain_node_ids
                        embed_noise[recon_node_ids] =\
                            (mask_type * np.stack([-np.ones(recon_node_ids.shape), recon_node_ids],
                                                  axis=1)).sum(axis=1).astype(np.int32)
                        embed_noise_dict[key] = embed_noise
                    else:
                        embed_noise_dict[key] = -np.ones(self._all_graph.node_ids[key].shape,
                                                         dtype=np.int32)
                        embed_noise_dict[key][node_ids] = node_ids
            else:
                raise NotImplementedError

            curr_dict = {key: 0 for key in recon_node_ids_dict}
            while True:
                batch_recon_node_ids_dict = dict()
                for key, recon_node_ids in recon_node_ids_dict.items():
                    curr = curr_dict[key]
                    if curr > recon_node_ids.size:
                        continue
                    batch_recon_node_ids_dict[key] = recon_node_ids[curr:(curr + batch_size)]
                    curr_dict[key] += batch_size
                if len(batch_recon_node_ids_dict) == 0:
                    break
                if not sequential and (len(batch_recon_node_ids_dict) != len(recon_node_ids_dict)):
                    break
                yield embed_noise_dict, batch_recon_node_ids_dict, recon_node_ids_dict
            if sequential:
                break

    def __repr__(self):
        stream = io.StringIO()
        print('All Graph=', file=stream)
        print(self.all_graph, file=stream)
        print('Test Graph=', file=stream)
        print(self.test_graph, file=stream)
        print('Val Graph=', file=stream)
        print(self.val_graph, file=stream)
        print('Train Graph=', file=stream)
        print(self.train_graph, file=stream)
        return stream.getvalue()

