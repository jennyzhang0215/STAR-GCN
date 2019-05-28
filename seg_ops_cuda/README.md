Implementation of the Segmented Operators
-----------------------------------------
To use the following operators, copy the `seg_op.*` files in [mxnet_op](mxnet_op) to `mxnet/src/operator/contrib`.

- [seg_sum](#seg_sum)
- [seg_broadcast_add](#seg_broadcast_add)
- [seg_broadcast_mul](#seg_broadcast_mul)
- [seg_broadcast_to](#seg_broadcast_to)
- [seg_softmax](#seg_softmax)
- [seg_take_k_corr](#seg_take_k_corr)
- [seg_weighted_pool](#seg_weighted_pool)
- [seg_pool](#seg_pool)

 After copying and recompiling MXNet, test the operators by running

```python
python mxnet_op/test_seg_ops.py
```

## seg_sum

Reduce the last dimension of the input based on the given segment indicators.

Inputs:
- data: Shape (batch_num, nnz)
- indptr: Shape (seg_num + 1,)

Outputs:
- ret: Shape (batch_num, seg_num)
```c++
for k = 0 to batch_num - 1
    for i = 0 to seg_num - 1
        ret[k, i] = reduce(data[k, indptr[i]], ..., data[k, indptr[i + 1] - 1])
```
Examples::

      out = seg_sum(data=data, indptr=indptr)

## seg_broadcast_add

Broadcast rhs according to the segment indicators and add to lhs to get the result.

Inputs:
- lhs: Shape (batch_num, nnz)
- rhs: Shape (batch_num, seg_num)
- indptr: Shape (seg_num + 1,)

Outputs:
- ret: Shape (batch_num, nnz)

Examples::

    ret = seg_broadcast_add(lhs=lhs, rhs=rhs, indptr=indptr)

## seg_broadcast_mul

Broadcast rhs according to the segment indicators and mul to lhs to get the result.

Inputs:
- lhs: Shape (batch_num, nnz)
- rhs: Shape (batch_num, seg_num)
- indptr: Shape (seg_num + 1,)

Outputs:
- ret: Shape (batch_num, nnz)

Examples::

    ret = seg_broadcast_mul(lhs=lhs, rhs=rhs, indptr=indptr)

## seg_broadcast_to

Broadcast rhs according to the segment indicators and add to lhs to get the result.

Inputs:
- data: Shape (batch_num, seg_num)
- indptr: Shape (seg_num + 1,)
- int nnz

Outputs:
- ret: Shape (batch_num, nnz)

Examples::

    ret = seg_broadcast_to(data=data, indptr=indptr, nnz=nnz)

## seg_softmax

Calculate the softmax of the the input based on the given segment indicators.

Inputs:
- data: Shape (batch_num, nnz)
- indptr: Shape (seg_num + 1,)

Outputs:
- ret: Shape (batch_num, nnz)

```c++
for k = 0 to batch_num - 1
    for i = 0 to seg_num - 1
        ret[k, indptr[i]:indptr[i+1]] = softmax(data[k, indptr[i]:indptr[i+1]])
```

Examples::

    out = seg_softmax(data=data, indptr=indptr)

## seg_take_k_corr

For all the nodes, computes the inner product between the node and it's neighborhoods and add to dst.
We assume the node_ids are 0, 1, 2, ..., node_num - 1

Inputs:
- embed1: Shape (K, node_num, feat_dim)
- embed2: Shape (K, neighbor_node_num, feat_dim)
- neighbor_ids: Shape (nnz, )
- neighbor_indptr: Shape(node_num + 1, )

Outputs:
- dst: Shape (K, nnz)

```c++
for k = 0 to K-1
    for i = 0  to node_num - 1
        for j = ind_ptr[i] to ind_ptr[i+1] - 1
            neighbor_id = neighbor_ids[j]
            dst[k, j] += InnerProduct(embed1[k, i], embed2[k, neighbor_id])
```

Examples::

    out = seg_take_k_corr(embed1=embed1, embed2=embed2, neighbor_ids=neighbor_ids, neighbor_indptr=neighbor_indptr)

## seg_weighted_pool

Compute weighted average of values in the segments

Inputs:
- data: Shape (batch_size, total_ind_num, feat_dim)
- weights: Shape (batch_size, nnz)
- indices: Shape (nnz, )
- indptr: Shape (seg_num + 1,)

Outputs:
- dst: Shape (batch_size, seg_num, feat_dim)

```c++
for k = 0 to K-1
    for i = 0  to node_num - 1
        for j = ind_ptr[i] to ind_ptr[i+1] - 1
            dst[k, i, :] += weights[k, j] * data[k, neighbor_ids[j], :]
```

Examples::

    out = seg_weighted_pool(data=data, weights=weights, indices=indices, indptr=indptr)

## seg_pool

Pooling of the values in the segments

Inputs:
- data : Shape (batch_size, total_ind_num, feat_dim)
- indices : Shape (nnz,)
- indptr : Shape (seg_num + 1,)
- pool_type : 'avg' or 'sum' or 'max'

Outputs:
- dst : Shape (batch_size, seg_num, feat_dim)

Examples::

    out = seg_pool(data=data,
                   indices=indices,
                   indptr=indptr,
                   pool_type='avg')