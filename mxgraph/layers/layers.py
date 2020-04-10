import warnings
from ..graph import HeterGraph, empty_as_zero, merge_nodes, unordered_unique
from .aggregators import *
from ..utils import copy_to_ctx
from mxnet.gluon import nn
import mxnet as mx

class LayerDictionary(nn.Block):
    def __init__(self, **kwargs):
        """
        
        Parameters
        ----------
        input_dims : dict
        output_dims : dict
        """
        super(LayerDictionary, self).__init__(**kwargs)
        self._key2idx = dict()
        with self.name_scope():
            self._layers = nn.HybridSequential()
        self._nlayers = 0

    def __len__(self):
        return len(self._layers)

    def __setitem__(self, key, layer):
        if key in self._key2idx:
            warnings.warn('Duplicate Key. Need to test the code!')
            self._layers[self._key2idx[key]] = layer
        else:
            self._layers.add(layer)
            self._key2idx[key] = self._nlayers
            self._nlayers += 1

    def __getitem__(self, key):
        return self._layers[self._key2idx[key]]

    def __contains__(self, key):
        return key in self._key2idx


class HeterGCNLayer(nn.Block):
    def __init__(self, meta_graph, multi_link_structure, agg_units, out_units,
                 source_keys=None, dropout_rate=0.0,
                 agg_ordinal_sharing=False,
                 agg_accum='stack', agg_act='relu',
                 layer_accum='stack', accum_self=False, out_act=None,
                 prefix=None, params=None):
        """

        Parameters
        ----------
        meta_graph : dict
        multi_link_structure : dict
            {(src_key, dst_key): len(multi_link)}
        agg_units : int or dict or None
            If it's a dictionary, it will be node_key : #units
        out_units : int or dict or None
            If it's a dictionary, it will be node_key : #units
        source_keys : dict or None
            The
        dropout_rate : float
        agg_ordinal_sharing : bool
        agg_accum : str
        agg_act : None or nn.Activation or str
            Activation of the output layer
        act : None or nn.Activation or str
            Activation of the output layer
        layer_accum : str
            Can be 'stack' or 'sum'
        accum_self : bool
            Whether to accumulate the features of yourself
        out_act : None or nn.Activation or str
        prefix : str
        params : None
        """
        super(HeterGCNLayer, self).__init__(prefix=prefix, params=params)
        self._meta_graph = meta_graph
        if source_keys is None:
            source_keys = meta_graph.keys()
        self._source_keys = source_keys
        if not isinstance(out_units, dict):
            out_units = {k: out_units for k in source_keys}
        if not isinstance(agg_units, dict):
            agg_units = {k: agg_units for k in meta_graph}

        self._layer_accum = layer_accum
        self._accum_self = accum_self
        self._out_act = get_activation(out_act)
        with self.name_scope():
            self.dropout = nn.Dropout(dropout_rate) ### dropout before feeding the out layer
            # Build the aggregators
            self._aggregators = LayerDictionary(prefix='agg_')
            with self._aggregators.name_scope():
                for src_key in source_keys:
                    for dst_key in meta_graph[src_key]:
                        if multi_link_structure[(src_key, dst_key)] is None:
                            self._aggregators[(src_key, dst_key)] =\
                                    GCNAggregator(units=agg_units[src_key],
                                                  act=agg_act,
                                                  dropout_rate=dropout_rate,
                                                  prefix='{}_{}_'.format(src_key, dst_key))
                        else:
                            self._aggregators[(src_key, dst_key)] =\
                                    MultiLinkGCNAggregator(
                                        units=agg_units[src_key],
                                        num_links=multi_link_structure[(src_key, dst_key)],
                                        act=agg_act,
                                        dropout_rate=dropout_rate,
                                        ordinal_sharing=agg_ordinal_sharing,
                                        accum=agg_accum,
                                        prefix='{}_{}_'.format(src_key, dst_key))


            # Build the output FC layers
            self._out_fcs = LayerDictionary(prefix='out_fc_')
            with self._out_fcs.name_scope():
                for key, ele_units in out_units.items():
                    if ele_units is not None:
                        self._out_fcs[key] = nn.Dense(ele_units, flatten=False,
                                                      prefix='{}_'.format(key))

            # Build the layer norm layers
            # if self._use_layer_norm:
            #     self._layer_norms = LayerDictionary(prefix='layer_norm_')
            #     with self._layer_norms.name_scope():
            #         for key in source_keys:
            #             self._layer_norms[key] = nn.LayerNorm(prefix='{}_'.format(key))

            if self._accum_self:
                self._self_fcs = LayerDictionary(prefix='self_fc_')
                with self._self_fcs.name_scope():
                    for key, ele_units in out_units.items():
                        if ele_units is not None:
                            self._self_fcs[key] = nn.HybridSequential(prefix='{}_'.format(key))
                            with self._self_fcs[key].name_scope():
                                self._self_fcs[key].add(nn.Dropout(dropout_rate))
                                self._self_fcs[key].add(nn.Dense(ele_units, flatten=False,
                                                                 params=None,
                                                                 prefix='{}_'.format(key)))
                                self._self_fcs[key].add(nn.Dropout(dropout_rate))

    @property
    def aggregators(self):
        return self._aggregators

    def forward_single(self, key, base_feas, neighbor_data):
        """

        Parameters
        ----------
        key : str
            The key of the target
        base_feas : mx.nd.ndarray
            Features of the base node. Shape (#DST Nodes, C)
        neighbor_data : dict
            {neighbor_key: (feas, end_points, edge_values, indptr, support)}
        Returns
        -------
        dst_out : mx.nd.ndarray
        """
        out_l = []
        for dst_key in self._meta_graph[key]:
            neighbor_feas, end_points, edge_values, indptr, support = neighbor_data[dst_key]
            if self._aggregators[(key, dst_key)].use_support:
                out = self._aggregators[(key, dst_key)](neighbor_feas, end_points, indptr, support)
            else:
                out = self._aggregators[(key, dst_key)](neighbor_feas, end_points, indptr)
            out = self.dropout(out)
            out_l.append(out)

        if self._accum_self:
            out_l.append(self._self_fcs[key](base_feas))
        if len(out_l) == 1:
            out = out_l[0]
        else:
            if self._layer_accum == 'stack':
                out = mx.nd.concat(*out_l, dim=1)
            elif self._layer_accum == 'sum':
                out = mx.nd.add_n(*out_l)
            else:
                raise NotImplementedError
        out = self._out_fcs[key](out)
        out = self._out_act(out)
        # if self._use_layer_norm:
        #     out = self._layer_norms[key](out)
        return out

    def forward(self, base_feas, neighbor_data):
        """Aggregate the information from the src data to the dst

        Parameters
        ----------
        base_feas : dict
            {key : mx.nd.ndarray}
        neighbor_data : dict
            {key : {src_key : (feas, end_points, indptr, support)}}

        Returns
        -------
        out : dict
            {key: mx.nd.ndarray}
        """
        out = {}
        for key, ele_feas in base_feas.items():
            assert key in neighbor_data
            out[key] = self.forward_single(key, ele_feas, neighbor_data[key])
        return out

class InnerProductLayer(nn.HybridBlock):
    def __init__(self, mid_units=None, **kwargs):
        super(InnerProductLayer, self).__init__(**kwargs)
        self._mid_units = mid_units
        if self._mid_units is not None:
            self._mid_map = nn.Dense(mid_units, flatten=False)

    def hybrid_forward(self, F, data1, data2):
        if self._mid_units is not None:
            data1 = self._mid_map(data1)
            data2 = self._mid_map(data2)
        score = F.sum(data1 * data2, axis=1, keepdims=True)
        return score

class StackedHeterGCNLayers(nn.Sequential):
    """Stack multiple HeterGCNLayers
    """
    def __init__(self, recurrent_layer_num=None, **kwargs):
        super(StackedHeterGCNLayers, self).__init__(**kwargs)
        self._recurrent_layer_num = recurrent_layer_num

    def __len__(self):
        if self._recurrent_layer_num is None:
            return super(StackedHeterGCNLayers, self).__len__()
        else:
            if super(StackedHeterGCNLayers, self).__len__() == 0:
                return 0
            else:
                return self._recurrent_layer_num

    def __getitem__(self, key):
        if self._recurrent_layer_num is not None:
            if key < self._recurrent_layer_num:
                return super(StackedHeterGCNLayers, self).__getitem__(0)
            else:
                raise KeyError('{} is out of range. Layer number={}'.format(key, len(self)))
        else:
            return super(StackedHeterGCNLayers, self).__getitem__(key)

    def add(self, *blocks):
        if self._recurrent_layer_num is not None:
            if len(self) == 1:
                raise ValueError('Cannot add more blocks if `use_recurrent` flag is turned on!')
            if len(blocks) > 1:
                raise ValueError('Can only add a single block if `use_recurrent` flag'
                                 ' is turned on!')
        for block in blocks:
            assert isinstance(block, HeterGCNLayer)
        super(StackedHeterGCNLayers, self).add(*blocks)

    def gen_plan(self, graph, sel_node_ids_dict, graph_sampler_args=None, symm=True):
        """

        Parameters
        ----------
        graph : HeterGraph
        sel_node_ids_dict : dict
            The selected node ids
        graph_sampler_args : dict
            Arguments for the graph sampler
            {(src_key, dst_key) : num_neighbors}
        symm : bool
            Whether to calculate the support in a symmetric way

        Returns
        -------
        req_node_ids_dict : dict
            Dictionary that contains the required node ids
        computing_plan : list
        """
        computing_plan = [None for _ in range(len(self))]
        for depth in range(len(self) - 1, -1, -1):
            prev_level_ids_dict = dict()
            agg_args_dict = dict()
            all_neighbor_ids_dict = dict()
            all_src_ids_dict = dict()
            for src_key, sel_node_ids in sel_node_ids_dict.items():
                print(src_key, sel_node_ids)
                if depth == len(self) - 1:
                    uniq_sel_node_ids, sel_node_idx = unordered_unique(sel_node_ids, return_inverse=True)
                else:
                    uniq_sel_node_ids, sel_node_idx = sel_node_ids, None
                agg_args_dict[src_key] = [uniq_sel_node_ids, sel_node_idx, dict()]
                all_src_ids_dict[src_key] = uniq_sel_node_ids
                for dst_key in graph.meta_graph[src_key]:
                    use_multi_link = self[depth].aggregators[(src_key, dst_key)].use_multi_link
                    end_points_ids, edge_values, ind_ptr, support = \
                        graph[src_key, dst_key].sample_neighbors(src_ids=uniq_sel_node_ids,
                                                                 symm=symm,
                                                                 use_multi_link=use_multi_link,
                                                                 num_neighbors=graph_sampler_args[(src_key, dst_key)])
                    # The aggregation parameters should be [end_points, edge_values ind_ptr, support]
                    #    the previous two will be filled later.
                    agg_args_dict[src_key][2][dst_key] = [None, edge_values, ind_ptr, support]
                    if dst_key not in all_neighbor_ids_dict:
                        all_neighbor_ids_dict[dst_key] = dict()
                    all_neighbor_ids_dict[dst_key][src_key] = end_points_ids

            # Map the end_points_ids to end_points_inds
            for key in set(all_neighbor_ids_dict.keys()) | set(all_src_ids_dict.keys()):
                node_ids_l = []
                if key in all_neighbor_ids_dict:
                    for _, end_points in all_neighbor_ids_dict[key].items():
                        if isinstance(end_points, np.ndarray):
                            node_ids_l.append(end_points)
                        else:
                            node_ids_l.extend(end_points)
                if key in all_src_ids_dict:
                    node_ids_l.append(all_src_ids_dict[key])
                uniq_node_ids, node_inds_l = merge_nodes(node_ids_l)
                prev_level_ids_dict[key] = uniq_node_ids
                curr = 0
                if key in all_neighbor_ids_dict:
                    for src_key, end_points in all_neighbor_ids_dict[key].items():
                        if isinstance(end_points, np.ndarray):
                            agg_args_dict[src_key][2][key][0] = node_inds_l[curr]
                            curr += 1
                        else:
                            assert isinstance(end_points, list)
                            agg_args_dict[src_key][2][key][0] = \
                                node_inds_l[curr:(curr + len(end_points))]
                            curr += len(end_points)
                if key in all_src_ids_dict:
                    agg_args_dict[key][0] = node_inds_l[curr]
            computing_plan[depth] = [prev_level_ids_dict, agg_args_dict]
            sel_node_ids_dict = prev_level_ids_dict
        return computing_plan[0][0], computing_plan

    def heter_sage(self, input_dict, computing_plan):
        """

        Parameters
        ----------
        input_dict : dict of mx.nd.ndarray
        computing_plan : list
            The plan generated by `gen_plan`

        Returns
        -------
        ret : dict of mx.nd.ndarray
            NDArrays that contains the forward result
        """
        ctx = next(iter(input_dict.values())).context
        ret = dict()
        for depth in range(len(self)):
            ret = dict()
            prev_level_ids_dict, agg_args_dict = computing_plan[depth]
            for src_key in agg_args_dict:
                uniq_sel_node_inds, sel_node_idx, agg_info_dict = agg_args_dict[src_key]
                nd_src_feas = mx.nd.take(input_dict[src_key],
                                         mx.nd.array(uniq_sel_node_inds,
                                                     ctx=ctx, dtype=np.int32), axis=0)
                neighbor_data = {}
                for dst_key in agg_info_dict:
                    nd_neighbor_feas = input_dict[dst_key]
                    end_points, edge_values, ind_ptr, support = agg_info_dict[dst_key]
                    if isinstance(end_points, list):
                        nd_end_points = copy_to_ctx(empty_as_zero(end_points, np.int32), ctx, np.int32)
                        nd_edge_values = copy_to_ctx(empty_as_zero(edge_values, np.float32), ctx, np.float32)
                        nd_ind_ptr = copy_to_ctx(ind_ptr, ctx, np.int32)
                        nd_support = copy_to_ctx(empty_as_zero(support, np.float32), ctx, np.float32)
                    else:
                        nd_end_points = copy_to_ctx(end_points, ctx, np.int32)
                        nd_edge_values = copy_to_ctx(edge_values, ctx, np.float32)
                        nd_ind_ptr = copy_to_ctx(ind_ptr, ctx, np.int32)
                        nd_support = copy_to_ctx(support, ctx, np.float32)
                    neighbor_data[dst_key] = (nd_neighbor_feas, nd_end_points, nd_edge_values, nd_ind_ptr, nd_support)
                ret[src_key] = self[depth].forward_single(key=src_key,
                                                          base_feas=nd_src_feas,
                                                          neighbor_data=neighbor_data)
                if depth == len(self) - 1:
                    ret[src_key] = mx.nd.take(ret[src_key],
                                              mx.nd.array(sel_node_idx, dtype=np.int32, ctx=ctx))
            input_dict = ret
        return ret
