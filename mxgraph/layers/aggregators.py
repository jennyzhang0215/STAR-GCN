from mxnet.gluon import nn, HybridBlock
#from mxgraph.layers.common import *
from mxgraph.layers.common import get_activation

class BaseAggregator(HybridBlock):
    @property
    def use_mulit_link(self):
        raise NotImplementedError

    @property
    def use_support(self):
        raise NotImplementedError

    @property
    def use_edge_type(self):
        raise NotImplementedError


class GCNAggregator(BaseAggregator):
    def __init__(self, units, act=None,
                 dropout_rate=0.0, input_dropout=True, use_layer_norm=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)
        with self.name_scope():
            self._agg = MultiLinkGCNAggregator(units=units, num_links=1, act=act,
                                               dropout_rate=dropout_rate,
                                               input_dropout=input_dropout,
                                               use_layer_norm=use_layer_norm)

    @property
    def use_multi_link(self):
        return False

    @property
    def use_support(self):
        return True

    @property
    def use_edge_type(self):
        return False

    def hybrid_forward(self, F, neighbor_data, end_points, indptr, support):
        """

        Parameters
        ----------
        F
        neighbor_data
        end_points
        indptr
        support

        Returns
        -------
        out
        """
        return self._agg(neighbor_data, [end_points], [indptr], [support])


class MultiLinkGCNAggregator(BaseAggregator):
    def __init__(self, units, num_links, act=None, dropout_rate=0.0,
                 input_dropout=True, ordinal_sharing=True, use_layer_norm=False,
                 accum='stack', **kwargs):
        """

        Parameters
        ----------
        units : int
        num_links : int
        act : str
        dropout_rate : float
        input_dropout : bool
        ordinal_sharing : bool
        accum : str
        kwargs : dict
        """
        super(MultiLinkGCNAggregator, self).__init__(**kwargs)
        self._units = units
        self._num_links = num_links
        self._use_layer_norm = use_layer_norm
        self._act = get_activation(act)
        self._ordinal_sharing = ordinal_sharing
        self._accum = accum
        self._input_dropout = input_dropout
        if self._accum == 'stack':
            assert units % num_links == 0, 'units should be divisible by the num_links '
            self._units = self._units // num_links
        with self.name_scope():
            if self._use_layer_norm:
                self.layer_norm = nn.LayerNorm(epsilon=1E-3)
            self.dropout = nn.Dropout(dropout_rate)
            for i in range(num_links):
                self.__setattr__('weight{}'.format(i),
                                 self.params.get('weight{}'.format(i),
                                                 shape=(self._units, 0),
                                                 dtype=np.float32,
                                                 allow_deferred_init=True))
                self.__setattr__('bias{}'.format(i),
                                 self.params.get('bias{}'.format(i),
                                                 shape=(self._units,),
                                                 dtype=np.float32,
                                                 init='zeros',
                                                 allow_deferred_init=True))

    @property
    def use_multi_link(self):
        return True

    @property
    def use_support(self):
        return True

    @property
    def use_edge_type(self):
        return False

    def hybrid_forward(self, F, neighbor_data, end_points_l, indptr_l, support_l=None, **kwargs):
        """The basic aggregator

        Parameters
        ----------
        F
        neighbor_data: Symbol or NDArray
            Data related to the input: element Shape (neighbor_node_num, feat_dim2)
        end_points_l: Symbol or NDArray
            Contain ids of the neighboring nodes: element Shape (nnz,)
        indptr_l: Symbol or NDArray
            element Shape (node_num + 1, )
        support_l: Symbol or NDArray or None
            The edge support: element Shape (nnz,)
        Returns
        -------
        out : Symbol or NDArray
            The output features
        """
        out_l = []
        if self._input_dropout:
            neighbor_data = self.dropout(neighbor_data)
        weight, bias = kwargs['weight0'], kwargs['bias0']
        for i in range(self._num_links):
            if i > 0 and self._ordinal_sharing:
                # Implement ordinal weight sharing
                weight = weight + kwargs['weight{}'.format(i)]
                bias = bias + kwargs['bias{}'.format(i)]
            else:
                weight = kwargs['weight{}'.format(i)]
                bias = kwargs['bias{}'.format(i)]
            neighbor_feat = F.FullyConnected(neighbor_data,
                                             weight=weight, bias=bias,
                                             no_bias=False,
                                             num_hidden=self._units,
                                             flatten=False)
            out = F.contrib.seg_weighted_pool(data=F.expand_dims(neighbor_feat, axis=0),
                                              weights=F.expand_dims(support_l[i], axis=0),
                                              indices=end_points_l[i],
                                              indptr=indptr_l[i])
            out_l.append(F.reshape(out, shape=(-3, 0)))
        if len(out_l) == 1:
            out = out_l[0]
        else:
            if self._accum == 'stack':
                out = F.concat(*out_l, dim=1)
            elif self._accum == 'sum':
                out = F.add_n(*out_l)
            else:
                raise NotImplementedError
        out = self._act(out)
        if self._use_layer_norm:
            out = self.layer_norm(out)
        return out
