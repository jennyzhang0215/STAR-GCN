import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet.test_utils import assert_almost_equal

def rand_indptr(seg_num, nnz):
    choose_inds = sorted(np.random.choice(list(range(1, nnz)), seg_num - 1, replace=False))
    choose_inds = [0] + choose_inds + [nnz]
    return np.array(choose_inds, dtype=np.int32)

def npy_seg_sum(data, indptr):
    ret = np.zeros(shape=(data.shape[0], indptr.shape[0] - 1), dtype=np.float32)
    for i in range(indptr.shape[0] - 1):
        ret[:, i] = data[:, indptr[i]:indptr[i+1]].sum(axis=1)
    return ret

def npy_seg_broadcast_add(lhs, rhs, indptr):
    ret = np.zeros_like(lhs)
    ret[:] = lhs
    for i in range(indptr.shape[0] - 1):
        if indptr[i + 1] > indptr[i]:
            ret[:, indptr[i]:indptr[i+1]] += rhs[:, i:(i+1)]
    return ret

def npy_seg_broadcast_mul(lhs, rhs, indptr):
    ret = np.zeros_like(lhs)
    ret[:] = lhs
    for i in range(indptr.shape[0] - 1):
        if indptr[i + 1] > indptr[i]:
            ret[:, indptr[i]:indptr[i+1]] *= rhs[:, i:(i+1)]
    return ret

def npy_seg_broadcast_to(rhs, indptr, nnz):
    ret = np.zeros(shape=(rhs.shape[0], nnz), dtype=np.float32)
    for i in range(indptr.shape[0] - 1):
        if indptr[i + 1] > indptr[i]:
            ret[:, indptr[i]:indptr[i+1]] = rhs[:, i:(i+1)]
    return ret



def npy_softmax_contig(data):
    ret = np.exp(data - data.max(axis=-1, keepdims=True))
    ret /= ret.sum(axis=-1, keepdims=True)
    return ret

def npy_seg_softmax(data, indptr):
    ret = - np.ones(shape=data.shape, dtype=np.float32)
    for i in range(indptr.shape[0] - 1):
        if indptr[i + 1] > indptr[i]:
            ret[:, indptr[i]:indptr[i+1]] = npy_softmax_contig(data[:, indptr[i]:indptr[i + 1]])
    return ret

def npy_seg_take_k_corr(embed1, embed2, neighbor_ids, neighbor_indptr):
    K, node_num, feat_dim = embed1.shape
    nnz = neighbor_ids.shape[0]
    dst = np.zeros(shape=(K, nnz), dtype=np.float32)
    for k in range(K):
        for i in range(node_num):
            for j in range(neighbor_indptr[i], neighbor_indptr[i + 1]):
                dst[k, j] = (embed1[k, i, :] * embed2[k, neighbor_ids[j], :]).sum()
    return dst

def npy_seg_weighted_pool(data, weights, indices, indptr):
    batch_size, total_ind_num, feat_dim = data.shape
    seg_num = indptr.shape[0] - 1
    dst = np.zeros(shape=(batch_size, seg_num, feat_dim), dtype=np.float32)
    for b in range(batch_size):
        for i in range(seg_num):
            dst[b, i, :] = (weights[b, indptr[i]:indptr[i+1]].reshape((-1,1)) * data[b, indices[indptr[i]:indptr[i+1]], :]).sum(axis=0)
    return dst

def npy_seg_pool(data, indices, indptr, pool_type):
    batch_size = data.shape[0]
    seg_num = indptr.shape[0] - 1
    feat_dim = data.shape[2]
    ret = np.zeros(shape=(batch_size, seg_num, feat_dim), dtype=np.float32)
    for i in range(seg_num):
        if pool_type == "sum":
            ret[:, i, :] = data[:, indices[indptr[i]:indptr[i+1]], :].sum(axis=1)
        elif pool_type == "avg":
            ret[:, i, :] = data[:, indices[indptr[i]:indptr[i+1]], :].mean(axis=1)
        elif pool_type == "max":
            ret[:, i, :] = data[:, indices[indptr[i]:indptr[i+1]], :].max(axis=1)
    return ret

def grad_seg_max_pool(ograd, data, indices, indptr):
    ret = np.zeros(shape=data.shape, dtype=np.float32)
    batch_size = data.shape[0]
    seg_num = ograd.shape[1]
    feat_dim = data.shape[2]
    total_ind_num = data.shape[1]
    for b in range(batch_size):
        for c in range(feat_dim):
            for i in range(seg_num):
                seg_inds = indices[indptr[i]:indptr[i+1]]
                argmax_ind = seg_inds[data[b, seg_inds, c].argmax()]
                ret[b, argmax_ind, c] += ograd[b, i, c]
    return ret

def compute_fd(exe, nd_val, npy_val, eps):
    fd_val = np.zeros(nd_val.shape, dtype=np.float32)
    for i in range(npy_val.size):
        orig_val = npy_val.ravel()[i]
        npy_val.ravel()[i] = orig_val + eps
        nd_val[:] = npy_val
        v1 = mx.nd.sum(exe.forward(is_train=True)[0]).asnumpy()[0]
        npy_val.ravel()[i] = orig_val - eps
        nd_val[:] = npy_val
        v2 = mx.nd.sum(exe.forward(is_train=True)[0]).asnumpy()[0]
        npy_val.ravel()[i] = orig_val
        fd_val.ravel()[i] = (v1 - v2) / (2 * eps)
    nd_val[:] = npy_val
    return fd_val

def test_seg_sum_nd():
    for ctx in [mx.cpu(), mx.gpu()]:
        for batch_size, seg_num, nnz in [(1, 5, 10), (10, 50, 100), (4, 1000, 10000)]:
            data_npy = np.random.normal(0, 1, (batch_size, nnz))
            indptr_npy = rand_indptr(seg_num, nnz)
            gt_npy = npy_seg_sum(data_npy, indptr_npy)
            # Test mx.nd
            data_nd = nd.array(data_npy, dtype=np.float32, ctx=ctx)
            indptr_nd = nd.array(indptr_npy, dtype=np.int32, ctx=ctx)
            ret_nd = nd.contrib.seg_sum(data=data_nd, indptr=indptr_nd)
            assert_almost_equal(ret_nd.asnumpy(), gt_npy, rtol=1E-4, atol=1E-4)

def test_seg_sum_backward():
    for ctx in [mx.cpu(), mx.gpu()]:
        for grad_req in ["add", "write"]:
            for batch_size, seg_num, nnz in [(1, 5, 10), (10, 50, 100)]:
                # define symbols
                data = mx.sym.var('data', shape=(batch_size, nnz), dtype=np.float32)
                indptr = mx.sym.var('indptr', shape=(seg_num + 1, ), dtype=np.int32)
                out_grad = mx.sym.var('out_grad', shape=(batch_size, seg_num), dtype=np.float32)
                loss = mx.sym.make_loss(mx.sym.contrib.seg_sum(data=data, indptr=indptr) * out_grad)
                # initialize variables
                data_npy = np.random.normal(0, 1, (batch_size, nnz))
                indptr_npy = rand_indptr(seg_num, nnz)
                out_grad_npy = np.random.normal(0, 1, (batch_size, seg_num))
                data_grad_npy = np.random.normal(0, 1, (batch_size, nnz))
                data_nd = mx.nd.array(data_npy, dtype=np.float32, ctx=ctx)
                indptr_nd = mx.nd.array(indptr_npy, dtype=np.int32, ctx=ctx)
                out_grad_nd = mx.nd.array(out_grad_npy, dtype=np.float32, ctx=ctx)
                data_grad_nd = mx.nd.array(data_grad_npy, dtype=np.float32, ctx=ctx)
                # bind executor
                exe = loss.bind(ctx, grad_req=grad_req,
                                args={'data': data_nd, 'indptr': indptr_nd, 'out_grad': out_grad_nd},
                                args_grad={'data': data_grad_nd})
                exe.forward(is_train=True)
                exe.backward()
                mx_calc_grad = exe.grad_dict['data'].asnumpy() if grad_req != "add"\
                                                               else exe.grad_dict['data'].asnumpy() -\
                                                                    data_grad_npy.astype(np.float32)
                # run finite difference test
                fd_grad = compute_fd(exe, data_nd, data_npy, 1E-2)
                assert_almost_equal(fd_grad, mx_calc_grad, rtol=1E-3, atol=1E-3)



def test_seg_broadcast_binary():
    for ctx in [mx.cpu(), mx.gpu()]:
        for np_func, nd_func, name in [(npy_seg_broadcast_add, nd.contrib.seg_broadcast_add, 'add'),
                                       (npy_seg_broadcast_mul, nd.contrib.seg_broadcast_mul, 'mul')]:
            for batch_size, seg_num, nnz in [(1, 5, 10), (10, 50, 100), (4, 1000, 10000)]:
                lhs_npy = np.random.normal(0, 1, (batch_size, nnz))
                rhs_npy = np.random.normal(0, 1, (batch_size, seg_num))
                indptr_npy = rand_indptr(seg_num, nnz)

                # Test broadcast_add
                print('broadcast_' + name)
                gt_npy = np_func(lhs_npy, rhs_npy, indptr_npy)
                # Test mx.nd
                lhs_nd = nd.array(lhs_npy, dtype=np.float32, ctx=ctx)
                rhs_nd = nd.array(rhs_npy, dtype=np.float32, ctx=ctx)
                indptr_nd = nd.array(indptr_npy, dtype=np.int32, ctx=ctx)
                ret_nd = nd_func(lhs=lhs_nd, rhs=rhs_nd, indptr=indptr_nd)
                assert_almost_equal(ret_nd.asnumpy(), gt_npy, rtol=1E-4, atol=1E-4)

def test_seg_broadcast_binary_backward():
    for ctx in [mx.cpu(), mx.gpu()]:
        for grad_req in ["add", "write"]:
            for batch_size, seg_num, nnz in [(1, 5, 10), (10, 50, 100)]:
                for sym_func, name in [
                    (mx.sym.contrib.seg_broadcast_add, 'add'),
                    (mx.sym.contrib.seg_broadcast_mul, 'mul')]:
                    print('Test broadcast_' + name)
                    # define symbols
                    lhs = mx.sym.var('lhs', shape=(batch_size, nnz), dtype=np.float32)
                    rhs = mx.sym.var('rhs', shape=(batch_size, seg_num), dtype=np.float32)
                    indptr = mx.sym.var('indptr', shape=(seg_num + 1, ), dtype=np.int32)
                    out_grad = mx.sym.var('out_grad', shape=(batch_size, nnz), dtype=np.float32)
                    loss = mx.sym.make_loss(sym_func(lhs=lhs, rhs=rhs, indptr=indptr) * out_grad)
                    # initialize variables
                    lhs_npy = np.random.normal(0, 1, (batch_size, nnz))
                    rhs_npy = np.random.normal(0, 1, (batch_size, seg_num))
                    indptr_npy = rand_indptr(seg_num, nnz)
                    out_grad_npy = np.random.normal(0, 1, (batch_size, nnz))
                    lhs_grad_npy = np.random.normal(0, 1, (batch_size, nnz))
                    rhs_grad_npy = np.random.normal(0, 1, (batch_size, seg_num))
                    lhs_nd = mx.nd.array(lhs_npy, dtype=np.float32, ctx=ctx)
                    rhs_nd = mx.nd.array(rhs_npy, dtype=np.float32, ctx=ctx)
                    indptr_nd = mx.nd.array(indptr_npy, dtype=np.int32, ctx=ctx)
                    out_grad_nd = mx.nd.array(out_grad_npy, dtype=np.float32, ctx=ctx)
                    lhs_grad_nd = mx.nd.array(lhs_grad_npy, dtype=np.float32, ctx=ctx)
                    rhs_grad_nd = mx.nd.array(rhs_grad_npy, dtype=np.float32, ctx=ctx)
                    # bind executor
                    exe = loss.bind(ctx, grad_req=grad_req,
                                    args={'lhs': lhs_nd, 'rhs': rhs_nd, 'indptr': indptr_nd, 'out_grad': out_grad_nd},
                                    args_grad={'lhs': lhs_grad_nd, 'rhs': rhs_grad_nd})
                    exe.forward(is_train=True)
                    exe.backward()
                    mx_lhs_grad = exe.grad_dict['lhs'].asnumpy() if grad_req != "add" \
                        else exe.grad_dict['lhs'].asnumpy() - lhs_grad_npy.astype(np.float32)
                    mx_rhs_grad = exe.grad_dict['rhs'].asnumpy() if grad_req != "add" \
                        else exe.grad_dict['rhs'].asnumpy() - rhs_grad_npy.astype(np.float32)
                    # run finite difference test
                    fd_lhs_grad = compute_fd(exe, lhs_nd, lhs_npy, 1E-2)
                    fd_rhs_grad = compute_fd(exe, rhs_nd, rhs_npy, 1E-2)
                    assert_almost_equal(fd_lhs_grad, mx_lhs_grad, rtol=1E-3, atol=1E-3)
                    assert_almost_equal(fd_rhs_grad, mx_rhs_grad, rtol=1E-3, atol=1E-3)

def test_seg_broadcast_to():
    for ctx in [mx.cpu(), mx.gpu()]:
        for batch_size, seg_num, nnz in [(1, 5, 10), (10, 50, 100), (4, 1000, 10000)]:
            data_npy = np.random.normal(0, 1, (batch_size, seg_num))
            indptr_npy = rand_indptr(seg_num, nnz)
            gt_npy = npy_seg_broadcast_to(data_npy, indptr_npy, nnz)
            # Test mx.nd
            data_nd = nd.array(data_npy, dtype=np.float32, ctx=ctx)
            indptr_nd = nd.array(indptr_npy, dtype=np.int32, ctx=ctx)
            ret_nd = nd.contrib.seg_broadcast_to(data=data_nd, indptr=indptr_nd, nnz=nnz)
            assert_almost_equal(ret_nd.asnumpy(), gt_npy, rtol=1E-4, atol=1E-4)

def test_seg_broadcast_to_backward():
    for ctx in [mx.cpu(), mx.gpu()]:
        for grad_req in ["add", "write"]:
            for batch_size, seg_num, nnz in [(1, 5, 10), (10, 50, 100)]:
                # define symbols
                data = mx.sym.var('data', shape=(batch_size, seg_num), dtype=np.float32)
                indptr = mx.sym.var('indptr', shape=(seg_num + 1, ), dtype=np.int32)
                out_grad = mx.sym.var('out_grad', shape=(batch_size, nnz), dtype=np.float32)
                loss = mx.sym.make_loss(mx.sym.contrib.seg_broadcast_to(data=data, indptr=indptr, nnz=nnz) * out_grad)
                # initialize variables
                data_npy = np.random.normal(0, 1, (batch_size, seg_num))
                indptr_npy = rand_indptr(seg_num, nnz)
                out_grad_npy = np.random.normal(0, 1, (batch_size, nnz))
                data_grad_npy = np.random.normal(0, 1, (batch_size, seg_num))
                data_nd = mx.nd.array(data_npy, dtype=np.float32, ctx=ctx)
                indptr_nd = mx.nd.array(indptr_npy, dtype=np.int32, ctx=ctx)
                out_grad_nd = mx.nd.array(out_grad_npy, dtype=np.float32, ctx=ctx)
                data_grad_nd = mx.nd.array(data_grad_npy, dtype=np.float32, ctx=ctx)
                # bind executor
                exe = loss.bind(ctx, grad_req=grad_req,
                                args={'data': data_nd, 'indptr': indptr_nd, 'out_grad': out_grad_nd},
                                args_grad={'data': data_grad_nd})
                exe.forward(is_train=True)
                exe.backward()
                mx_data_grad = exe.grad_dict['data'].asnumpy() if grad_req != "add" \
                    else exe.grad_dict['data'].asnumpy() - data_grad_npy.astype(np.float32)
                # run finite difference test
                fd_data_grad = compute_fd(exe, data_nd, data_npy, 1E-2)
                assert_almost_equal(fd_data_grad, mx_data_grad, rtol=1E-3, atol=1E-3)



def test_seg_softmax():
    for ctx in [mx.cpu(), mx.gpu()]:
        for batch_size, seg_num, nnz in [(1, 5, 10), (10, 50, 100), (4, 1000, 10000)]:
            data_npy = np.random.normal(0, 1, (batch_size, nnz))
            indptr_npy = rand_indptr(seg_num, nnz)
            gt_npy = npy_seg_softmax(data_npy, indptr_npy)
            # Test mx.nd
            data_nd = nd.array(data_npy, dtype=np.float32, ctx=ctx)
            indptr_nd = nd.array(indptr_npy, dtype=np.int32, ctx=ctx)
            ret_nd = nd.contrib.seg_softmax(data=data_nd, indptr=indptr_nd)
            assert_almost_equal(ret_nd.asnumpy(), gt_npy, rtol=1E-4, atol=1E-4)

def test_seg_softmax_backward():
    for ctx in [mx.cpu(), mx.gpu()]:
        for grad_req in ["add", "write"]:
            for batch_size, seg_num, nnz in [(1, 5, 10), (10, 50, 100)]:
                # define symbols
                data = mx.sym.var('data', shape=(batch_size, nnz), dtype=np.float32)
                indptr = mx.sym.var('indptr', shape=(seg_num + 1, ), dtype=np.int32)
                out_grad = mx.sym.var('out_grad', shape=(batch_size, nnz), dtype=np.float32)
                loss = mx.sym.make_loss(mx.sym.contrib.seg_softmax(data=data, indptr=indptr) * out_grad)
                # initialize variables
                data_npy = np.random.normal(0, 1, (batch_size, nnz))
                indptr_npy = rand_indptr(seg_num, nnz)
                out_grad_npy = np.random.normal(0, 1, (batch_size, nnz))
                data_grad_npy = np.random.normal(0, 1, (batch_size, nnz))
                data_nd = mx.nd.array(data_npy, dtype=np.float32, ctx=ctx)
                indptr_nd = mx.nd.array(indptr_npy, dtype=np.int32, ctx=ctx)
                out_grad_nd = mx.nd.array(out_grad_npy, dtype=np.float32, ctx=ctx)
                data_grad_nd = mx.nd.array(data_grad_npy, dtype=np.float32, ctx=ctx)
                # bind executor
                exe = loss.bind(ctx, grad_req=grad_req,
                                args={'data': data_nd, 'indptr': indptr_nd, 'out_grad': out_grad_nd},
                                args_grad={'data': data_grad_nd})
                exe.forward(is_train=True)
                exe.backward()
                mx_calc_grad = exe.grad_dict['data'].asnumpy() if grad_req != "add"\
                                                               else exe.grad_dict['data'].asnumpy() -\
                                                                    data_grad_npy.astype(np.float32)
                # run finite difference test
                fd_grad = compute_fd(exe, data_nd, data_npy, 1E-2)
                assert_almost_equal(fd_grad, mx_calc_grad, rtol=1E-3, atol=1E-3)



def test_seg_take_k_corr():
    for ctx in [mx.cpu(), mx.gpu()]:
        for K, node_num, neighbor_node_num, nnz, feat_dim in [(1, 5, 10, 30, 128),
                                                              (10, 50, 20, 500, 4),
                                                              (4, 1000, 10000, 50000, 4)]:
            embed1_npy = np.random.normal(0, 1, (K, node_num, feat_dim))
            embed2_npy = np.random.normal(0, 1, (K, neighbor_node_num, feat_dim))
            neighbor_ids_npy = np.random.randint(0, neighbor_node_num, size=(nnz,))
            neighbor_indptr_npy = rand_indptr(seg_num=node_num, nnz=nnz)
            gt_npy = npy_seg_take_k_corr(embed1=embed1_npy, embed2=embed2_npy,
                                         neighbor_ids=neighbor_ids_npy, neighbor_indptr=neighbor_indptr_npy)
            # Test mx.nd
            embed1_nd = nd.array(embed1_npy, dtype=np.float32, ctx=ctx)
            embed2_nd = nd.array(embed2_npy, dtype=np.float32, ctx=ctx)
            neighbor_ids_nd = nd.array(neighbor_ids_npy, dtype=np.int32, ctx=ctx)
            neighbor_indptr_nd = nd.array(neighbor_indptr_npy, dtype=np.int32, ctx=ctx)
            ret_nd = nd.contrib.seg_take_k_corr(embed1=embed1_nd, embed2=embed2_nd,
                                                neighbor_ids=neighbor_ids_nd, neighbor_indptr=neighbor_indptr_nd)
            assert_almost_equal(ret_nd.asnumpy(), gt_npy, rtol=1E-4, atol=1E-4)

def test_seg_take_k_corr_backward():
    for ctx in [mx.cpu(), mx.gpu()]:
        for grad_req in ["add", "write"]:
            for K, node_num, neighbor_node_num, nnz, feat_dim in [(1, 5, 10, 30, 8),
                                                                  (10, 50, 20, 100, 4)]:
                # define symbols
                embed1 = mx.sym.var('embed1', shape=(K, node_num, feat_dim), dtype=np.float32)
                embed2 = mx.sym.var('embed2', shape=(K, neighbor_node_num, feat_dim), dtype=np.float32)
                neighbor_ids = mx.sym.var('neighbor_ids', shape=(nnz,), dtype=np.int32)
                neighbor_indptr = mx.sym.var('neighbor_indptr', shape=(node_num + 1,), dtype=np.int32)
                out_grad = mx.sym.var('out_grad', shape=(K, nnz), dtype=np.float32)
                loss = mx.sym.make_loss(mx.sym.contrib.seg_take_k_corr(embed1=embed1, embed2=embed2,
                                                                       neighbor_ids=neighbor_ids,
                                                                       neighbor_indptr=neighbor_indptr) * out_grad)
                # initialize variables
                embed1_npy = np.random.normal(0, 1, (K, node_num, feat_dim))
                embed2_npy = np.random.normal(0, 1, (K, neighbor_node_num, feat_dim))
                neighbor_ids_npy = np.random.randint(0, neighbor_node_num, size=(nnz,))
                neighbor_indptr_npy = rand_indptr(seg_num=node_num, nnz=nnz)
                out_grad_npy = np.random.normal(0, 1, (K, nnz))
                embed1_grad_npy = np.random.normal(0, 1, (K, node_num, feat_dim))
                embed2_grad_npy = np.random.normal(0, 1, (K, neighbor_node_num, feat_dim))
                embed1_nd = mx.nd.array(embed1_npy, dtype=np.float32, ctx=ctx)
                embed2_nd = mx.nd.array(embed2_npy, dtype=np.float32, ctx=ctx)
                neighbor_ids_nd = mx.nd.array(neighbor_ids_npy, dtype=np.int32, ctx=ctx)
                neighbor_indptr_nd = mx.nd.array(neighbor_indptr_npy, dtype=np.int32, ctx=ctx)
                out_grad_nd = mx.nd.array(out_grad_npy, dtype=np.float32, ctx=ctx)
                embed1_grad_nd = mx.nd.array(embed1_grad_npy, dtype=np.float32, ctx=ctx)
                embed2_grad_nd = mx.nd.array(embed2_grad_npy, dtype=np.float32, ctx=ctx)
                # bind executor
                exe = loss.bind(ctx, grad_req=grad_req,
                                args={'embed1': embed1_nd, 'embed2': embed2_nd, 'neighbor_ids': neighbor_ids_nd,
                                      'neighbor_indptr': neighbor_indptr_nd, 'out_grad': out_grad_nd},
                                args_grad={'embed1': embed1_grad_nd, 'embed2': embed2_grad_nd})
                exe.forward(is_train=True)
                exe.backward()
                mx_embed1_grad = exe.grad_dict['embed1'].asnumpy() if grad_req != "add" \
                    else exe.grad_dict['embed1'].asnumpy() - embed1_grad_npy.astype(np.float32)
                mx_embed2_grad = exe.grad_dict['embed2'].asnumpy() if grad_req != "add" \
                    else exe.grad_dict['embed2'].asnumpy() - embed2_grad_npy.astype(np.float32)
                # run finite difference test
                fd_embed1_grad = compute_fd(exe, embed1_nd, embed1_npy, 2E-2)
                fd_embed2_grad = compute_fd(exe, embed2_nd, embed2_npy, 2E-2)
                assert_almost_equal(fd_embed1_grad, mx_embed1_grad, rtol=1E-3, atol=1E-3)
                assert_almost_equal(fd_embed2_grad, mx_embed2_grad, rtol=1E-3, atol=1E-3)



def test_seg_weighted_pool():
    for ctx in [mx.cpu(), mx.gpu()]:
        for batch_size, seg_num, total_ind_num, nnz, feat_dim in [(1, 5, 10, 30, 128),
                                                                  (10, 50, 20, 500, 4),
                                                                  (4, 1000, 10000, 50000, 4)]:
            data_npy = np.random.normal(0, 1, (batch_size, total_ind_num, feat_dim))
            weights_npy = np.random.normal(0, 1, (batch_size, nnz))
            indices_npy = np.random.randint(0, total_ind_num, size=(nnz,), dtype=np.int32)
            indptr_npy = rand_indptr(seg_num, nnz)
            gt_npy = npy_seg_weighted_pool(data=data_npy, weights=weights_npy,
                                           indices=indices_npy, indptr=indptr_npy)
            # Test mx.nd
            data_nd = nd.array(data_npy, dtype=np.float32, ctx=ctx)
            weights_nd = nd.array(weights_npy, dtype=np.float32, ctx=ctx)
            indices_nd = nd.array(indices_npy, dtype=np.int32, ctx=ctx)
            indptr_nd = nd.array(indptr_npy, dtype=np.int32, ctx=ctx)
            ret_nd = nd.contrib.seg_weighted_pool(data=data_nd, weights=weights_nd,
                                                  indices=indices_nd, indptr=indptr_nd)
            assert_almost_equal(ret_nd.asnumpy(), gt_npy, rtol=1E-4, atol=1E-4)

def test_seg_weighted_pool_backward():
    for ctx in [mx.cpu(), mx.gpu()]:
        for grad_req in ["add", "write"]:
            for batch_size, seg_num, total_ind_num, nnz, feat_dim in [(1, 5, 10, 30, 8),
                                                                      (10, 50, 20, 500, 4)]:
                # define symbols
                data = mx.sym.var('data', shape=(batch_size, total_ind_num, feat_dim), dtype=np.float32)
                weights = mx.sym.var('weights', shape=(batch_size, nnz), dtype=np.float32)
                indices = mx.sym.var('indices', shape=(nnz,), dtype=np.int32)
                indptr = mx.sym.var('indptr', shape=(seg_num + 1), dtype=np.int32)
                out_grad = mx.sym.var('out_grad', shape=(batch_size, seg_num, feat_dim), dtype=np.float32)
                loss = mx.sym.make_loss(mx.sym.contrib.seg_weighted_pool(data=data, weights=weights,
                                                                         indices=indices, indptr=indptr) * out_grad)
                # initialize variables
                data_npy = np.random.normal(0, 1, (batch_size, total_ind_num, feat_dim))
                weights_npy = np.random.normal(0, 1, (batch_size, nnz))
                indices_npy = np.random.randint(0, total_ind_num, size=(nnz,), dtype=np.int32)
                indptr_npy = rand_indptr(seg_num, nnz)
                out_grad_npy = np.random.normal(0, 1, (batch_size, seg_num, feat_dim))
                data_grad_npy = np.random.normal(0, 1, (batch_size, total_ind_num, feat_dim))
                weights_grad_npy = np.random.normal(0, 1, (batch_size, nnz))
                data_nd = mx.nd.array(data_npy, dtype=np.float32, ctx=ctx)
                weights_nd = mx.nd.array(weights_npy, dtype=np.float32, ctx=ctx)
                indices_nd = mx.nd.array(indices_npy, dtype=np.int32, ctx=ctx)
                indptr_nd = mx.nd.array(indptr_npy, dtype=np.int32, ctx=ctx)
                out_grad_nd = mx.nd.array(out_grad_npy, dtype=np.float32, ctx=ctx)
                data_grad_nd = mx.nd.array(data_grad_npy, dtype=np.float32, ctx=ctx)
                weights_grad_nd = mx.nd.array(weights_grad_npy, dtype=np.float32, ctx=ctx)
                # bind executor
                exe = loss.bind(ctx, grad_req=grad_req,
                                args={'data': data_nd, 'weights': weights_nd, 'indices': indices_nd,
                                      'indptr': indptr_nd, 'out_grad': out_grad_nd},
                                args_grad={'data': data_grad_nd, 'weights': weights_grad_nd})
                exe.forward(is_train=True)
                exe.backward()
                mx_data_grad = exe.grad_dict['data'].asnumpy() if grad_req != "add" \
                    else exe.grad_dict['data'].asnumpy() - data_grad_npy.astype(np.float32)
                mx_weights_grad = exe.grad_dict['weights'].asnumpy() if grad_req != "add" \
                    else exe.grad_dict['weights'].asnumpy() - weights_grad_npy.astype(np.float32)
                # run finite difference test
                fd_data_grad = compute_fd(exe, data_nd, data_npy, 2E-2)
                fd_weights_grad = compute_fd(exe, weights_nd, weights_npy, 2E-2)
                assert_almost_equal(fd_data_grad, mx_data_grad, rtol=1E-3, atol=1E-3)
                assert_almost_equal(fd_weights_grad, mx_weights_grad, rtol=1E-3, atol=1E-3)



def test_seg_pool():
    for ctx in [mx.cpu(), mx.gpu()]:
        for pool_type in ["sum", "avg", "max"]:
            for batch_size, seg_num, total_ind_num, nnz, feat_dim in [(1, 5, 10, 30, 128),
                                                                      (10, 50, 20, 500, 4),
                                                                      (4, 1000, 10000, 50000, 4)]:
                data_npy = np.random.normal(0, 1, (batch_size, total_ind_num, feat_dim))
                indices_npy = np.random.randint(0, total_ind_num, size=(nnz,), dtype=np.int32)
                indptr_npy = rand_indptr(seg_num, nnz)
                gt_npy = npy_seg_pool(data=data_npy, indices=indices_npy, indptr=indptr_npy, pool_type=pool_type)
                # Test mx.nd
                data_nd = nd.array(data_npy, dtype=np.float32, ctx=ctx)
                indices_nd = nd.array(indices_npy, dtype=np.int32, ctx=ctx)
                indptr_nd = nd.array(indptr_npy, dtype=np.int32, ctx=ctx)
                ret_nd = nd.contrib.seg_pool(data=data_nd, indices=indices_nd, indptr=indptr_nd, pool_type=pool_type)
                assert_almost_equal(ret_nd.asnumpy(), gt_npy, rtol=1E-4, atol=1E-4)

def test_seg_pool_backward():
    for ctx in [mx.cpu(), mx.gpu()]:
        np.random.seed(1000)
        for pool_type in ["sum", "avg", "max"]:
            for grad_req in ["add", "write"]:
                for batch_size, seg_num, total_ind_num, nnz, feat_dim in [(1, 5, 10, 30, 8),
                                                                          (10, 50, 20, 500, 4)]:
                    # define symbols
                    data = mx.sym.var('data', shape=(batch_size, total_ind_num, feat_dim), dtype=np.float32)
                    indices = mx.sym.var('indices', shape=(nnz,), dtype=np.int32)
                    indptr = mx.sym.var('indptr', shape=(seg_num + 1), dtype=np.int32)
                    out_grad = mx.sym.var('out_grad', shape=(batch_size, seg_num, feat_dim), dtype=np.float32)
                    loss = mx.sym.make_loss(mx.sym.contrib.seg_pool(data=data,
                                                                    indices=indices, indptr=indptr,
                                                                    pool_type=pool_type) * out_grad)
                    # initialize variables
                    if pool_type == "max":
                        data_npy = np.random.normal(0, 10, (batch_size, total_ind_num, feat_dim))
                    else:
                        data_npy = np.random.normal(0, 1, (batch_size, total_ind_num, feat_dim))
                    indices_npy = np.random.randint(0, total_ind_num, size=(nnz,), dtype=np.int32)
                    indptr_npy = rand_indptr(seg_num, nnz)
                    out_grad_npy = np.random.normal(0, 1, (batch_size, seg_num, feat_dim))
                    data_grad_npy = np.random.normal(0, 1, (batch_size, total_ind_num, feat_dim))
                    data_nd = mx.nd.array(data_npy, dtype=np.float32, ctx=ctx)
                    indices_nd = mx.nd.array(indices_npy, dtype=np.int32, ctx=ctx)
                    indptr_nd = mx.nd.array(indptr_npy, dtype=np.int32, ctx=ctx)
                    out_grad_nd = mx.nd.array(out_grad_npy, dtype=np.float32, ctx=ctx)
                    data_grad_nd = mx.nd.array(data_grad_npy, dtype=np.float32, ctx=ctx)
                    # bind executor
                    exe = loss.bind(ctx, grad_req=grad_req,
                                    args={'data': data_nd, 'indices': indices_nd, 'indptr': indptr_nd,
                                          'out_grad': out_grad_nd},
                                    args_grad={'data': data_grad_nd})
                    exe.forward(is_train=True)
                    exe.backward()
                    mx_data_grad = exe.grad_dict['data'].asnumpy() if grad_req != "add" \
                        else exe.grad_dict['data'].asnumpy() - data_grad_npy.astype(np.float32)
                    print('batch_size=', batch_size, 'pool_type=', pool_type, 'ctx=', ctx, 'grad_req=', grad_req)
                    # run finite difference test
                    if pool_type != "max":
                        fd_data_grad = compute_fd(exe, data_nd, data_npy, 2E-2)
                    else:
                        fd_data_grad = grad_seg_max_pool(ograd=out_grad_npy, data=data_npy, indices=indices_npy,
                                                         indptr=indptr_npy)
                    assert_almost_equal(fd_data_grad, mx_data_grad, rtol=2E-3, atol=2E-3)

if __name__ == '__main__':
    import nose
    nose.runmodule()
