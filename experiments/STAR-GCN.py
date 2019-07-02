import os
import argparse
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxgraph.datasets import LoadData
from mxgraph.graph import HeterGraph, merge_node_ids_dict, set_seed
from mxgraph.layers import HeterGCNLayer, StackedHeterGCNLayers, LayerDictionary, InnerProductLayer
from mxgraph.layers.common import get_activation
from mxgraph.iterators import DataIterator
from mxgraph.utils import gluon_net_info, parse_ctx, logging_config, gluon_total_param_num, params_clip_global_norm
from mxgraph.helpers.ordered_easydict import OrderedEasyDict as edict
from mxgraph.helpers.metric_logger import MetricLogger
from mxgraph.config import cfg_from_file, save_cfg_dir

def config():
    parser = argparse.ArgumentParser(description='Run the baseline method.')
    parser.add_argument('--cfg', dest='cfg_file', help='Optional configuration file',
                        default=None, type=str)
    parser.add_argument('--ctx', dest='ctx', default='gpu',
                        help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`', type=str)
    parser.add_argument('--save_dir', help='The saving directory', type=str)
    parser.add_argument('--dataset', help='The dataset name: ml-100k, ml-1m, ml-10m', type=str,
                        default='ml-100k')
    parser.add_argument('--inductive', dest='inductive', help='Whether to train the model in the inductive setting.',
                        action='store_true')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--silent', action='store_true')
    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)[0]

    cfg = edict()
    cfg.SEED = args.seed
    cfg.DATASET = edict()
    cfg.DATASET.NAME = args.dataset  # e.g. ml-100k
    cfg.DATASET.VALID_RATIO = 0.1
    cfg.DATASET.TEST_RATIO = 0.2
    cfg.DATASET.IS_INDUCTIVE = args.inductive
    cfg.DATASET.INDUCTIVE_KEY = "item" ## ["item" / "name"]
    cfg.DATASET.INDUCTIVE_NODE_FRAC = 20 ### choice: 20
    cfg.DATASET.INDUCTIVE_EDGE_FRAC = 90 ### choice: 50, 70, 90

    cfg.MODEL = edict()
    cfg.MODEL.USE_EMBED = True
    cfg.MODEL.USE_FEA_PROJ = False
    cfg.MODEL.RECON_FEA = False
    cfg.MODEL.REMOVE_RATING = True
    cfg.MODEL.USE_DAE = True  # Use the denoising autoencoder structure.
    cfg.MODEL.NBLOCKS = 2  # Number of AE blocks. NBLOCK = 3 ==> AE1 --> AE2 --> AE3 Like the hourglass structure
    cfg.MODEL.USE_RECURRENT = False  # Whether to share the weights between different AE blocks
    cfg.MODEL.RECON_LAMBDA = 0.1  # Weight for the reconstruction loss
    cfg.MODEL.ACTIVATION = "leaky"

    cfg.GRAPH_SAMPLER = edict()  # Sample a random number of neighborhoods for mini-batch training
    cfg.GRAPH_SAMPLER.NUM_NEIGHBORS = -1

    cfg.FEA = edict()
    cfg.FEA.MID_MAP = 16
    cfg.FEA.UNITS = 16

    cfg.EMBED = edict()
    cfg.EMBED.UNITS = 64
    cfg.EMBED.MASK_PROP = 0.1
    cfg.EMBED.P_ZERO = 0.0

    cfg.GCN = edict()
    cfg.GCN.TYPE = 'gcn'
    cfg.GCN.DROPOUT = 0.7
    cfg.GCN.USE_RECURRENT = False  # Whether to use recurrent connections
    cfg.GCN.AGG = edict()
    cfg.GCN.AGG.NORM_SYMM = True
    cfg.GCN.AGG.UNITS = [500]  # Number of aggregator units
    cfg.GCN.AGG.ACCUM = "stack"
    cfg.GCN.OUT = edict()
    cfg.GCN.OUT.UNITS = [75]  # [50, 100] ### the hidden state of FC

    cfg.GEN_RATING = edict()
    cfg.GEN_RATING.MID_MAP = 64

    cfg.TRAIN = edict()
    cfg.TRAIN.RATING_BATCH_SIZE = 10000
    cfg.TRAIN.RECON_BATCH_SIZE = 1000000
    cfg.TRAIN.MAX_ITER = 1000000  ### Need to tune
    cfg.TRAIN.LOG_INTERVAL = 10
    cfg.TRAIN.VALID_INTERVAL = 10
    cfg.TRAIN.OPTIMIZER = "adam"
    cfg.TRAIN.LR = 1E-2  # initial learning rate
    cfg.TRAIN.WD = 0.0
    cfg.TRAIN.DECAY_PATIENCE = 100
    cfg.TRAIN.MIN_LR = 5E-4
    cfg.TRAIN.LR_DECAY_FACTOR = 0.5
    cfg.TRAIN.EARLY_STOPPING_PATIENCE = 150
    cfg.TRAIN.GRAD_CLIP = 10.0

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file, target=cfg)
    ### manually set some configurations
    assert cfg.MODEL.USE_EMBED or cfg.MODEL.USE_FEA_PROJ

    if cfg.MODEL.NBLOCKS > 1:
        assert cfg.MODEL.USE_DAE
    ### configure save_fir to save all the info
    if args.save_dir is None:
        if args.cfg_file is None:
            raise ValueError("Must set --cfg if not set --save_dir")
        args.save_dir = os.path.splitext(args.cfg_file)[0]
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_id = save_cfg_dir(args.save_dir, source=cfg)

    return cfg, args


cfg, args = config()
logging_config(folder=args.save_dir, name='log{:d}'.format(args.save_id), no_console=args.silent)
logging.info(cfg)
_DATA = cfg.DATASET
_MODEL = cfg.MODEL
_GRAPH_SAMPLER = cfg.GRAPH_SAMPLER
_FEA = cfg.FEA
_EMBED = cfg.EMBED
_GCN = cfg.GCN
_GEN_RATING = cfg.GEN_RATING
_TRAIN = cfg.TRAIN


def load_dataset(seed):
    dataset = LoadData(_DATA.NAME,
                       use_inductive=_DATA.IS_INDUCTIVE,
                       test_ratio=_DATA.TEST_RATIO, val_ratio=_DATA.VALID_RATIO,
                       inductive_key=_DATA.INDUCTIVE_KEY,
                       inductive_node_frac=_DATA.INDUCTIVE_NODE_FRAC,
                       inductive_edge_frac=_DATA.INDUCTIVE_EDGE_FRAC,
                       seed=seed)
    all_graph = dataset.graph
    logging.info(dataset)
    # !IMPORTANT. We need to check that ids in all_graph are continuous from 0 to #Node - 1.
    # We will later use these ids to take the embedding vectors
    all_graph.check_continous_node_ids()
    feature_dict = dict()
    info_line = "Feature dim: "
    for key in all_graph.meta_graph:
        features = mx.nd.array(all_graph.features[key], ctx=args.ctx, dtype=np.float32)
        feature_dict[key] = features
        info_line += "\n" + key + ": {}".format(features.shape)
    logging.info(info_line)
    return dataset, all_graph, feature_dict


def gen_graph_sampler_args(meta_graph):
    ret = dict()
    for src_key in meta_graph:
        for dst_key in meta_graph[src_key]:
            ret[(src_key, dst_key)] = _GRAPH_SAMPLER.NUM_NEIGHBORS
    return ret


def gen_pair_key(src_key, dst_key):
    if src_key < dst_key:
        return src_key, dst_key
    else:
        return dst_key, src_key


class Net(nn.Block):
    def __init__(self, all_graph, nratings, name_user, name_item, **kwargs):
        super(Net, self).__init__(**kwargs)
        self._nratings = nratings
        self._name_user = name_user
        self._name_item = name_item
        self._act = get_activation(_MODEL.ACTIVATION)
        with self.name_scope():
            if _MODEL.USE_EMBED:
                self.embed_layers = LayerDictionary(prefix='embed_')
                with self.embed_layers.name_scope():
                    for key, fea in all_graph.features.items():
                        self.embed_layers[key] = nn.Embedding(input_dim=fea.shape[0],
                                                              output_dim=_EMBED.UNITS,
                                                              weight_initializer=mx.init.Uniform(0.1),
                                                              prefix='{}_'.format(key))
            if _MODEL.USE_FEA_PROJ:
                self.fea_mappings = LayerDictionary(prefix='fea_map_')
                with self.fea_mappings.name_scope():
                    for key in all_graph.features:
                        self.fea_mappings[key] = nn.HybridSequential()
                        self.fea_mappings[key].add(nn.Dense(units=_FEA.MID_MAP,
                                                          flatten=False, prefix='{}_l0_'.format(key)))
                        self.fea_mappings[key].add(get_activation(_MODEL.ACTIVATION))
                        self.fea_mappings[key].add(nn.Dense(units=_FEA.UNITS,
                                                          flatten=False, prefix='{}_l1_'.format(key)))


            # Construct Encoder
            self.encoders = nn.Sequential(prefix='enc_')
            with self.encoders.name_scope():
                num_enc_blocks = 1 if _MODEL.USE_RECURRENT else _MODEL.NBLOCKS
                for block_id in range(num_enc_blocks):
                    recurrent_layer_num = len(_GCN.AGG.UNITS) if _GCN.USE_RECURRENT else None
                    encoder = StackedHeterGCNLayers(recurrent_layer_num=recurrent_layer_num,
                                                    prefix='b{}_'.format(block_id))
                    with encoder.name_scope():
                        for i, (agg_units, out_units) in enumerate(zip(_GCN.AGG.UNITS, _GCN.OUT.UNITS)):
                            if not _GCN.USE_RECURRENT and not _MODEL.USE_DAE\
                                    and (i == len(_GCN.AGG.UNITS) - 1):
                                source_keys = [name_user, name_item] ### For HeterGCN without link prediction training
                            else:
                                source_keys = all_graph.meta_graph.keys()

                            encoder.add(HeterGCNLayer(meta_graph=all_graph.meta_graph,
                                                      multi_link_structure=all_graph.get_multi_link_structure(),
                                                      dropout_rate=_GCN.DROPOUT,
                                                      agg_units=agg_units,
                                                      out_units=out_units,
                                                      source_keys=source_keys,
                                                      agg_accum=_GCN.AGG.ACCUM,
                                                      agg_act=_MODEL.ACTIVATION,
                                                      out_act=_MODEL.ACTIVATION,
                                                      prefix='l{}_'.format(i)))
                            if _GCN.USE_RECURRENT:
                                # In the recurrent formula, we will only create one layer
                                break
                    self.encoders.add(encoder)

            # Construct Decoder
            if _MODEL.USE_DAE:
                num_dec_blocks = 1 if _MODEL.USE_RECURRENT else _MODEL.NBLOCKS
                # Generate the embed_map
                self.embed_maps = nn.Sequential(prefix='embed_maps_')
                with self.embed_maps.name_scope():
                    if _MODEL.USE_FEA_PROJ and _MODEL.RECON_FEA:
                        out_emb_units = _EMBED.UNITS + _FEA.UNITS
                    else:
                        out_emb_units = _EMBED.UNITS
                    for block_id in range(num_dec_blocks):
                        embed_map = LayerDictionary(prefix='b{}_'.format(block_id))
                        with embed_map.name_scope():
                            for key in all_graph.meta_graph:
                                embed_map[key] = nn.HybridSequential(prefix='{}_'.format(key))
                                with embed_map[key].name_scope():
                                    embed_map[key].add(nn.Dense(units=out_emb_units, flatten=False,
                                                                prefix='{}_l0_'.format(key)))
                                    embed_map[key].add(get_activation(_MODEL.ACTIVATION))
                                    embed_map[key].add(nn.Dense(units=out_emb_units, flatten=False,
                                                                prefix='{}_l1_'.format(key)))
                        self.embed_maps.add(embed_map)

            self.gen_ratings = InnerProductLayer(prefix='gen_rating')

            self.rating_user_projs = nn.Sequential(prefix='rating_user_proj_')
            self.rating_item_projs = nn.Sequential(prefix='rating_item_proj_')
            for rating_proj in [self.rating_user_projs, self.rating_item_projs]:
                with rating_proj.name_scope():
                    num_blocks = 1 if _MODEL.USE_RECURRENT else _MODEL.NBLOCKS
                    for block_id in range(num_blocks):
                        ele_proj = nn.HybridSequential(prefix='b{}_'.format(block_id))
                        with ele_proj.name_scope():
                            ele_proj.add(nn.Dense(units=_GEN_RATING.MID_MAP,
                                                  flatten=False))
                        rating_proj.add(ele_proj)

    def get_embed(self, ctx, node_ids_dict, embed_noise_dict=None, use_mask=True):
        """ Generate the embedding of the nodes

        Parameters
        ----------
        node_ids_dict : dict
            Dictionary that contains the ids of the nodes that need to be embedded
            Inner values should be mx.nd.ndarrays
        feature_dict : dict
            Dictionary that contains the features of the nodes
        embed_noise_dict : dict
            Dictionary that contains the noise information.
            There are two possible values:
                -1 --> mask to zero
                i --> use the embedding vector as in the ith-node
            Inner values should be mx.nd.ndarrays
        use_mask : bool
            Whether to mask the embeddings

        Returns
        -------
        embedding_dict : dict
            Dictionary that contains the node embeddings
        """
        assert _MODEL.USE_EMBED
        embedding_dict = dict()
        for key, node_ids in node_ids_dict.items():
            node_ids = mx.nd.array(node_ids, ctx=ctx, dtype=np.int32)
            if use_mask:
                node_ids = mx.nd.take(embed_noise_dict[key], node_ids)
                mask = (node_ids != -1)
                node_ids = node_ids * mask
            embedding = self.embed_layers[key](node_ids)
            if use_mask:
                embedding = embedding * mx.nd.reshape(mask, shape=(-1, 1)).astype(np.float32)
            embedding_dict[key] = embedding
        return embedding_dict

    def get_feature(self, ctx, node_ids_dict, feature_dict):
        assert _MODEL.USE_FEA_PROJ
        out_fea_dict = dict()
        for key, node_ids in node_ids_dict.items():
            out_fea_dict[key] = self.fea_mappings[key](mx.nd.take(feature_dict[key],
                                                                  mx.nd.array(node_ids, ctx=ctx, dtype=np.int32)))
        return out_fea_dict


    def forward(self, graph, feature_dict, rating_node_pairs=None,
                embed_noise_dict=None, recon_node_ids_dict=None, graph_sampler_args=None, symm=None):
        """

        Parameters
        ----------
        graph : HeterGraph
        feature_dict : dict
            Dictionary contains the base features of all nodes
        rating_node_pairs : np.ndarray or None
            Shape: (2, #Edges), First row is user and the second row is item
        embed_noise_dict : dict or None
            Dictionary that contains the noises of all nodes that is used to replace the node ids for masked embedding
            {key: (#all node ids, ) the shape and order is the same as the node ids in the whole graph}
        recon_node_ids_dict: dict or None
            Dictionary that contains the nodes ids that we need to reconstruct the embedding
        all_masked_node_ids_dict : dict or None
            Dictionary that contains the node ids of all masked nodes
        graph_sampler_args : dict or None
            Arguments for graph sampler
        symm : bool
            Whether to calculate the support in the symmetric formula

        Returns
        -------
        pred_ratings : list of mx.nd.ndarray
            The predicted ratings. If we use the stacked hourglass AE structure.
             it will return a list with multiple predicted ratings
        pred_embeddings : list of dict
            The predicted embeddings. Return a list of predicted embeddings
             if we use the stacked hourglass AE structure.
        gt_embeddings : dict
            The ground-truth embedding of the target node ids.
        """
        if symm is None:
            symm = _GCN.AGG.NORM_SYMM
        ctx = next(iter(feature_dict.values())).context
        req_node_ids_dict = dict()
        encoder_fwd_plan = [None for _ in range(_MODEL.NBLOCKS)]
        encoder_fwd_indices = [None for _ in range(_MODEL.NBLOCKS)]
        pred_ratings = []
        pred_embeddings = []
        block_req_node_ids_dict = [None for _ in range(_MODEL.NBLOCKS)]
        if embed_noise_dict is not None:
            nd_embed_noise_dict = {key: mx.nd.array(ele, ctx=ctx, dtype=np.int32)
                                   for key, ele in embed_noise_dict.items()}
        else:
            nd_embed_noise_dict = None
        if recon_node_ids_dict is not None:
            gt_embeddings = self.get_embed(ctx=ctx,
                                           node_ids_dict=recon_node_ids_dict,
                                           embed_noise_dict=nd_embed_noise_dict,
                                           use_mask=False)
            if _MODEL.USE_FEA_PROJ and _MODEL.RECON_FEA:
                gt_fea = self.get_feature(ctx=ctx,
                                          node_ids_dict=recon_node_ids_dict,
                                          feature_dict=feature_dict)
                for key in gt_embeddings:
                    gt_embeddings[key] = mx.nd.concat(gt_embeddings[key], gt_fea[key])
        else:
            gt_embeddings = dict()
        # From top to bottom, generate the forwarding plan
        for block_id in range(_MODEL.NBLOCKS - 1, -1, -1):
            # Backtrack the encoders
            encoder = self.encoders[0] if _MODEL.USE_RECURRENT else self.encoders[block_id]
            if rating_node_pairs is not None and recon_node_ids_dict is not None:
                uniq_node_ids_dict, encoder_fwd_indices[block_id] = \
                    merge_node_ids_dict([{self._name_user : rating_node_pairs[0],
                                          self._name_item: rating_node_pairs[1]},
                                         recon_node_ids_dict,
                                         req_node_ids_dict])
            elif rating_node_pairs is not None and recon_node_ids_dict is None:
                uniq_node_ids_dict, encoder_fwd_indices[block_id] = \
                    merge_node_ids_dict([{self._name_user: rating_node_pairs[0],
                                          self._name_item: rating_node_pairs[1]},
                                         req_node_ids_dict])
            elif rating_node_pairs is None and recon_node_ids_dict is not None:
                uniq_node_ids_dict, encoder_fwd_indices[block_id] = \
                    merge_node_ids_dict([recon_node_ids_dict, req_node_ids_dict])
            else:
                raise NotImplementedError
            block_req_node_ids_dict[block_id] = req_node_ids_dict
            req_node_ids_dict, encoder_fwd_plan[block_id]\
                = encoder.gen_plan(graph=graph,
                                   sel_node_ids_dict=uniq_node_ids_dict,
                                   graph_sampler_args=graph_sampler_args,
                                   symm=symm)

        # From bottom to top, calculate the forwarding results
        if _MODEL.USE_EMBED:
            input_dict = self.get_embed(ctx=ctx,
                                        node_ids_dict=req_node_ids_dict,
                                        embed_noise_dict=nd_embed_noise_dict,
                                        use_mask=embed_noise_dict is not None)
        if _MODEL.USE_FEA_PROJ:
            fea_dict = self.get_feature(ctx=ctx,
                                        node_ids_dict={key: mx.nd.array(req_node_ids, ctx=ctx, dtype=np.int32)
                                                       for key, req_node_ids in req_node_ids_dict.items()},
                                        feature_dict=feature_dict)
            if _MODEL.USE_EMBED:
                for key in input_dict:
                    input_dict[key] = mx.nd.concat(input_dict[key], fea_dict[key], dim=-1)
            else:
                input_dict = fea_dict

        for block_id in range(_MODEL.NBLOCKS):
            encoder = self.encoders[0] if _MODEL.USE_RECURRENT else self.encoders[block_id]
            output_dict = encoder.heter_sage(input_dict, encoder_fwd_plan[block_id])
            if rating_node_pairs is not None and recon_node_ids_dict is not None:
                rating_idx_dict, recon_idx_dict, req_idx_dict = encoder_fwd_indices[block_id]
            elif rating_node_pairs is not None and recon_node_ids_dict is None:
                rating_idx_dict, req_idx_dict = encoder_fwd_indices[block_id]
            elif rating_node_pairs is None and recon_node_ids_dict is not None:
                recon_idx_dict, req_idx_dict = encoder_fwd_indices[block_id]
            else:
                raise NotImplementedError

            # Generate the predicted ratings
            if rating_node_pairs is not None:
                rating_user_fea = mx.nd.take(output_dict[self._name_user],
                                             mx.nd.array(rating_idx_dict[self._name_user], ctx=ctx, dtype=np.int32))
                rating_item_fea = mx.nd.take(output_dict[self._name_item],
                                             mx.nd.array(rating_idx_dict[self._name_item], ctx=ctx, dtype=np.int32))
                user_proj = self.rating_user_projs[0] if _MODEL.USE_RECURRENT else self.rating_user_projs[block_id]
                item_proj = self.rating_item_projs[0] if _MODEL.USE_RECURRENT else self.rating_item_projs[block_id]
                rating_user_fea = user_proj(rating_user_fea)
                rating_item_fea = item_proj(rating_item_fea)
                block_pred_ratings = self.gen_ratings(rating_user_fea, rating_item_fea)
                pred_ratings.append(block_pred_ratings)

            # Decoder
            if recon_node_ids_dict is not None:
                embed_map = self.embed_maps[0] if _MODEL.USE_RECURRENT else self.embed_maps[block_id]
                # Generate the predicted embeddings
                block_pred_embeddings = dict()
                for key, idx in recon_idx_dict.items():
                    block_pred_embeddings[key] =\
                        embed_map[key](mx.nd.take(output_dict[key], mx.nd.array(idx, ctx=ctx, dtype=np.int32)))
                pred_embeddings.append(block_pred_embeddings)
            if block_id < _MODEL.NBLOCKS - 1 and _MODEL.USE_DAE:
                # Generate the Input Embeddings of the next layer
                embed_map = self.embed_maps[0] if _MODEL.USE_RECURRENT else self.embed_maps[block_id]
                input_dict = dict()
                for key, idx in req_idx_dict.items():
                    input_dict[key] = embed_map[key](mx.nd.take(output_dict[key], mx.nd.array(idx, ctx=ctx, dtype=np.int32)))
                if _MODEL.USE_FEA_PROJ and not _MODEL.RECON_FEA:
                    fea_dict = self.get_feature(ctx=ctx, node_ids_dict=block_req_node_ids_dict[block_id],
                                                feature_dict=feature_dict)
                    for key in input_dict:
                        input_dict[key] = mx.nd.concat(input_dict[key], fea_dict[key], dim=-1)

        return pred_ratings, pred_embeddings, gt_embeddings



def evaluate(net, feature_dict, ctx, data_iter, segment='valid'):
    rating_mean = data_iter._train_ratings.mean()
    rating_std = data_iter._train_ratings.std()
    rating_sampler = data_iter.rating_sampler(batch_size=_TRAIN.RATING_BATCH_SIZE, segment=segment,
                                              sequential=True)
    possible_rating_values = data_iter.possible_rating_values
    nd_possible_rating_values = mx.nd.array(possible_rating_values, ctx=args.ctx, dtype=np.float32)
    eval_graph = data_iter.val_graph if segment == 'valid' else data_iter.test_graph
    graph_sampler_args = gen_graph_sampler_args(data_iter.all_graph.meta_graph)
    # Evaluate RMSE
    rmse_l = [0 for _ in range(_MODEL.NBLOCKS)]
    cnt = 0

    for rating_node_pairs, gt_ratings in rating_sampler:
        nd_gt_ratings = mx.nd.array(gt_ratings, dtype=np.float32, ctx=ctx)
        cnt += rating_node_pairs.shape[1]

        pred_ratings, _, _ \
            = net.forward(graph=eval_graph,
                          feature_dict=feature_dict,
                          rating_node_pairs=rating_node_pairs,
                          embed_noise_dict=data_iter.evaluate_embed_noise_dict,
                          recon_node_ids_dict=None,
                          graph_sampler_args=graph_sampler_args,
                          symm=_GCN.AGG.NORM_SYMM)
        for i in range(_MODEL.NBLOCKS):
            rmse_l[i] +=\
                    mx.nd.square(mx.nd.clip(pred_ratings[i].reshape((-1,)) * rating_std + rating_mean,
                                            possible_rating_values.min(),
                                            possible_rating_values.max()) - nd_gt_ratings).sum().asscalar()
    for i in range(_MODEL.NBLOCKS):
        rmse_l[i] = np.sqrt(rmse_l[i] / cnt)
    return rmse_l


def log_str(loss_l, loss_name):
    return ', ' + \
           ', '.join(
               ["{}{}={:.3f}".format(loss_name, i, loss_l[i][0] / loss_l[i][1])
                for i in range(len(loss_l))])

def train(seed):
    dataset, all_graph, feature_dict = load_dataset(seed)
    valid_node_pairs, _ = dataset.valid_data
    test_node_pairs, _ = dataset.test_data
    if not _DATA.IS_INDUCTIVE:
        data_iter = DataIterator(all_graph=all_graph,
                                 name_user=dataset.name_user,
                                 name_item=dataset.name_item,
                                 test_node_pairs=test_node_pairs,
                                 valid_node_pairs=valid_node_pairs,
                                 is_inductive=_DATA.IS_INDUCTIVE,
                                 embed_P_mask=_EMBED.MASK_PROP,
                                 embed_p_zero=_EMBED.P_ZERO,
                                 embed_p_self=1.0-_EMBED.P_ZERO,
                                 seed=seed)
    else:
        if _DATA.INDUCTIVE_KEY == "item":
            inductive_key = dataset.name_item
            other_key = dataset.name_user
        elif _DATA.INDUCTIVE_KEY == "user":
            inductive_key = dataset.name_user
            other_key = dataset.name_item
        else:
            raise NotImplementedError
        print("DataIterator")
        data_iter = DataIterator(all_graph=all_graph,
                                 name_user=dataset.name_user,
                                 name_item=dataset.name_item,
                                 test_node_pairs=test_node_pairs,
                                 valid_node_pairs=valid_node_pairs,
                                 is_inductive=_DATA.IS_INDUCTIVE,
                                 inductive_key=inductive_key,
                                 inductive_valid_ids=dataset.inductive_valid_ids,
                                 inductive_train_ids=dataset.inductive_train_ids,
                                 embed_P_mask=_EMBED.MASK_PROP,
                                 embed_p_zero={inductive_key: _EMBED.P_ZERO, other_key: 0.0},
                                 embed_p_self={inductive_key: 1.0-_EMBED.P_ZERO, other_key: 1.0},
                                 seed=seed)
    logging.info(data_iter)
    ### build the net
    possible_rating_values = data_iter.possible_rating_values
    print("Net ...")
    net = Net(all_graph=all_graph, nratings=possible_rating_values.size,
              name_user=dataset.name_user, name_item=dataset.name_item)
    net.initialize(init=mx.init.Xavier(factor_type='in'), ctx=args.ctx)
    net.hybridize()
    rating_loss_net = gluon.loss.L2Loss()
    rating_loss_net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), _TRAIN.OPTIMIZER,
                            {'learning_rate': _TRAIN.LR, 'wd': _TRAIN.WD})
    ### prepare the logger
    train_loss_logger = MetricLogger(['iter', 'loss'] + sum([['rmse{}'.format(i),
                                                              'rating_loss{}'.format(i),
                                                              'recon_loss{}'.format(i)]
                                                             for i in range(_MODEL.NBLOCKS)], []),
                                     ['%d', '%.4f'] + ['%.4f', '%.4f', '%.4f'] * _MODEL.NBLOCKS,
                                     os.path.join(args.save_dir, 'train_loss%d.csv' % args.save_id))
    valid_loss_logger = MetricLogger(['iter'] + ['rmse{}'.format(i) for i in range(_MODEL.NBLOCKS)],
                                     ['%d'] + ['%.4f'] * _MODEL.NBLOCKS,
                                     os.path.join(args.save_dir, 'valid_loss%d.csv' % args.save_id))
    test_loss_logger = MetricLogger(['iter'] + ['rmse{}'.format(i) for i in range(_MODEL.NBLOCKS)],
                                    ['%d'] + ['%.4f'] * _MODEL.NBLOCKS,
                                    os.path.join(args.save_dir, 'test_loss%d.csv' % args.save_id))
    ### initialize the iterator
    print("sampler ...")
    rating_sampler = data_iter.rating_sampler(batch_size=_TRAIN.RATING_BATCH_SIZE, segment='train')
    print("recon_nodes_sampler ...")
    recon_sampler = data_iter.recon_nodes_sampler(batch_size=_TRAIN.RECON_BATCH_SIZE, segment='train')
    print("gen_graph_sampler_args")
    graph_sampler_args = gen_graph_sampler_args(all_graph.meta_graph)
    rating_mean = data_iter._train_ratings.mean()
    rating_std = data_iter._train_ratings.std()

    best_valid_rmse = np.inf
    best_test_rmse_l = None
    no_better_valid = 0
    best_iter = -1
    avg_gnorm = 0
    avg_rmse_l = [[0, 0] for _ in range(_MODEL.NBLOCKS)]
    avg_rating_loss_l = [[0, 0] for _ in range(_MODEL.NBLOCKS)]
    avg_recon_loss_l = [[0, 0] for _ in range(_MODEL.NBLOCKS)]

    for iter_idx in range(1, _TRAIN.MAX_ITER):
        rating_node_pairs, gt_ratings = next(rating_sampler)

        if _MODEL.USE_DAE:
            embed_noise_dict, recon_node_ids_dict, all_masked_node_ids_dict = next(recon_sampler)
        else:
            embed_noise_dict, recon_node_ids_dict, all_masked_node_ids_dict = None, None, None

        nd_gt_ratings = mx.nd.array(gt_ratings, ctx=args.ctx, dtype=np.float32)

        iter_graph = data_iter.train_graph
        ## remove the batch rating pair (optional)
        if rating_node_pairs.shape[1] < data_iter._train_node_pairs.shape[1] and _MODEL.REMOVE_RATING:
            if iter_idx == 1:
                logging.info("Removing training edges within the batch...")
            iter_graph = iter_graph.remove_edges_by_id(src_key=dataset.name_user,
                                                       dst_key=dataset.name_item,
                                                       node_pair_ids=rating_node_pairs)
        with mx.autograd.record():
            pred_ratings, pred_embeddings, gt_embeddings\
                = net.forward(graph=iter_graph,
                              feature_dict=feature_dict,
                              rating_node_pairs=rating_node_pairs,
                              embed_noise_dict=embed_noise_dict,
                              recon_node_ids_dict=recon_node_ids_dict,
                              graph_sampler_args=graph_sampler_args,
                              symm=_GCN.AGG.NORM_SYMM)
            rating_loss_l = []
            for i in range(_MODEL.NBLOCKS):
                ele_loss = rating_loss_net(mx.nd.reshape(pred_ratings[i], shape=(-1,)),
                                           (nd_gt_ratings - rating_mean) / rating_std ).mean()

                rating_loss_l.append(ele_loss)
            loss = sum(rating_loss_l)

            if _MODEL.USE_DAE:
                recon_loss_l = []
                for i in range(_MODEL.NBLOCKS):
                    block_loss = []
                    for key in gt_embeddings:
                        gt_emb = gt_embeddings[key]
                        pred_emb = pred_embeddings[i][key]
                        ele_loss = mx.nd.mean(mx.nd.sum(mx.nd.square(gt_emb - pred_emb), axis=-1))
                        block_loss.append(ele_loss)
                    recon_loss_l.append(sum(block_loss))
                loss = loss + _MODEL.RECON_LAMBDA * sum(recon_loss_l)
            loss.backward()
        gnorm = params_clip_global_norm(net.collect_params(), _TRAIN.GRAD_CLIP, args.ctx)
        avg_gnorm += gnorm
        trainer.step(1.0)

        if iter_idx == 1:
            logging.info("Total #Param of net: %d" % (gluon_total_param_num(net)))
            logging.info(gluon_net_info(net, save_path=os.path.join(args.save_dir, 'net%d.txt' % args.save_id)))
        # Calculate the avg losses
        for i in range(_MODEL.NBLOCKS):
            rmse = mx.nd.square(pred_ratings[i].reshape((-1,)) * rating_std + rating_mean
                                    - nd_gt_ratings).sum()
            avg_rmse_l[i][0] += rmse.asscalar()
            avg_rmse_l[i][1] += pred_ratings[i].shape[0]

            avg_rating_loss_l[i][0] += rating_loss_l[i].asscalar()
            avg_rating_loss_l[i][1] += 1
            if _MODEL.USE_DAE:
                avg_recon_loss_l[i][0] += recon_loss_l[i].asscalar()
                avg_recon_loss_l[i][1] += 1
        if iter_idx % _TRAIN.LOG_INTERVAL == 0:
            train_loss_info = dict({'iter': iter_idx})
            train_loss_info['loss'] = loss.asscalar()
            for i in range(_MODEL.NBLOCKS):
                train_loss_info['rmse{}'.format(i)] = np.sqrt(avg_rmse_l[i][0] / avg_rmse_l[i][1])
                train_loss_info['rating_loss{}'.format(i)] = avg_rating_loss_l[i][0] / avg_rating_loss_l[i][1]

                train_loss_info['recon_loss{}'.format(i)] = avg_recon_loss_l[i][0] / avg_recon_loss_l[i][1] \
                    if _MODEL.USE_DAE else 0
            train_loss_logger.log(**train_loss_info)

            logging_str = "Iter={}, gnorm={:.3f}, loss={:.3f}".format(iter_idx,
                                                                      avg_gnorm / _TRAIN.LOG_INTERVAL, loss.asscalar())
            logging_str += log_str(avg_rating_loss_l, "RT")
            if _MODEL.USE_DAE:
                logging_str += log_str(avg_recon_loss_l, "RC")
            logging_str += ', '  + ', '.join(["RMSE{}={:.3f}".format(i, np.sqrt(avg_rmse_l[i][0] / avg_rmse_l[i][1]))
                                              for i in range(_MODEL.NBLOCKS)])
            avg_gnorm = 0
            avg_rmse_l = [[0, 0] for _ in range(_MODEL.NBLOCKS)]
            avg_recon_loss_l = [[0, 0] for _ in range(_MODEL.NBLOCKS)]

        if iter_idx % _TRAIN.VALID_INTERVAL == 0:
            valid_rmse_l = evaluate(net=net,
                                    feature_dict=feature_dict,
                                    ctx=args.ctx,
                                    data_iter=data_iter,
                                    segment='valid')
            valid_loss_logger.log(**dict([('iter', iter_idx)] + [('rmse{}'.format(i), ele_rmse)
                                                                 for i, ele_rmse in enumerate(valid_rmse_l)]))
            logging_str += ',\t' + ', '.join(["Val RMSE{}={:.3f}".format(i, ele_rmse)
                                             for i, ele_rmse in enumerate(valid_rmse_l)])

            if valid_rmse_l[-1] < best_valid_rmse:
                best_valid_rmse = valid_rmse_l[-1]
                no_better_valid = 0
                best_iter = iter_idx
                # net.save_parameters(filename=os.path.join(args.save_dir, 'best_valid_net{}.params'.format(args.save_id)))
                test_rmse_l = evaluate(net=net, feature_dict=feature_dict, ctx=args.ctx,
                                       data_iter=data_iter, segment='test')
                best_test_rmse_l = test_rmse_l
                test_loss_logger.log(**dict([('iter', iter_idx)] +
                                            [('rmse{}'.format(i), ele_rmse)
                                             for i, ele_rmse in enumerate(test_rmse_l)]))
                logging_str += ', ' + ', '.join(["Test RMSE{}={:.4f}".format(i, ele_rmse)
                                                 for i, ele_rmse in enumerate(test_rmse_l)])
            else:
                no_better_valid += 1
                if no_better_valid > _TRAIN.EARLY_STOPPING_PATIENCE\
                    and trainer.learning_rate <= _TRAIN.MIN_LR:
                    logging.info("Early stopping threshold reached. Stop training.")
                    break
                if no_better_valid > _TRAIN.DECAY_PATIENCE:
                    new_lr = max(trainer.learning_rate * _TRAIN.LR_DECAY_FACTOR, _TRAIN.MIN_LR)
                    if new_lr < trainer.learning_rate:
                        logging.info("\tChange the LR to %g" % new_lr)
                        trainer.set_learning_rate(new_lr)
                        no_better_valid = 0
        if iter_idx  % _TRAIN.LOG_INTERVAL == 0:
            logging.info(logging_str)
    logging.info('Best Iter Idx={}, Best Valid RMSE={:.3f}, '.format(best_iter, best_valid_rmse) +
                 ', '.join(["Best Test RMSE{}={:.4f}".format(i, ele_rmse)
                            for i, ele_rmse in enumerate(best_test_rmse_l)]))
    train_loss_logger.close()
    valid_loss_logger.close()
    test_loss_logger.close()


if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    np.random.seed(cfg.SEED)
    mx.random.seed(cfg.SEED, args.ctx)
    set_seed(cfg.SEED)
    train(seed=cfg.SEED)
