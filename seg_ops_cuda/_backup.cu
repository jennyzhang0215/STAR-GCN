

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/*Reduce the data within the segments using atomic functions

dst: Shape (batch_num, seg_num, feat_dim)
data: Shape (batch_num, node_num, feat_dim)
indices: Shape (nnz, )
indptr: Shape (seg_num + 1, )
seg_ids: Shape(nnz, ), Reverse mapping from indices to seg_ids

reduce_type can be either mean or sum, only works for these two operators because we use the atomic function

for k = 0 to batch_num - 1
    for i = 0 to seg_num - 1
        for j = indptr[i]  to indptr[i+1] - 1
            dst[k, i, :] += data[k, indices[j], :]
Algorithm:
1. Fetch a bunch of seg_ids
2. Perform reduction using the shared memory. We use atomic add to accelerate the speed.

*/
template<int reduce_type, int UNROLL_NODE = 1, int TY_SZ = 16, int UNROLL_Y = 4, int UNROLL_X = 4, int WARP_SZ = 32>
__global__ void
__launch_bounds__(TY_SZ * WARP_SZ)
BatchSegReduceNonContigAtomicKernel(float* dst, const float* data, const int* indices, const int* indptr, const int* seg_ids, int batch_num, int seg_num, int feat_dim, int node_num, int nnz) {
    int batch_id = blockIdx.z;
    int c_begin = blockIdx.y * UNROLL_X * WARP_SZ;
    __shared__ float dst_shared[UNROLL_NODE * TY_SZ][UNROLL_X * WARP_SZ];
    __shared__ int seg_ids_shared[UNROLL_Y * TY_SZ];
    __shared__ int indices_shared[UNROLL_Y * TY_SZ];
    // Handle UNROLL_NODE * TY_SZ segments for one thread block
    for (int seg_begin = UNROLL_NODE * TY_SZ * blockIdx.x; seg_begin < seg_num; seg_begin += UNROLL_NODE * TY_SZ * gridDim.x) {
        int seg_end = min(seg_begin + UNROLL_NODE * TY_SZ, seg_num);
        int ind_begin = indptr[seg_begin];
        int ind_end = indptr[seg_end];

        // 1. Initialize the dst_shared
        #pragma unroll
        for(int j=0; j < UNROLL_NODE; j++) {
            #pragma unroll
            for(int i = 0; i < UNROLL_X; i++) {
                if(reduce_type == SegReduceType::kSum || reduce_type == SegReduceType::kMean) {
                    dst_shared[j * TY_SZ + threadIdx.y][i * WARP_SZ + threadIdx.x] = 0;
                } else if(reduce_type == SegReduceType::kMax) {
                    dst_shared[j * TY_SZ + threadIdx.y][i * WARP_SZ + threadIdx.x] = FLT_MIN;
                } else if(reduce_type == SegReduceType::kMin) {
                    dst_shared[j * TY_SZ + threadIdx.y][i * WARP_SZ + threadIdx.x] = FLT_MAX;
                }
            }
        }
        // 2. Reduce the values
        for (int inner_ind_begin = ind_begin; inner_ind_begin < ind_end; inner_ind_begin += UNROLL_Y * TY_SZ) {
            #pragma unroll
            for(int i = 0; i < CEIL_DIV(UNROLL_Y * TY_SZ, WARP_SZ); i++) {
                int inner_ind_delta = i * WARP_SZ + threadIdx.x;
                if(inner_ind_begin + inner_ind_delta < ind_end && inner_ind_delta < UNROLL_Y * TY_SZ && threadIdx.y == 0) {
                    seg_ids_shared[inner_ind_delta] = seg_ids[inner_ind_begin + inner_ind_delta] - seg_begin;
                    indices_shared[inner_ind_delta] = indices[inner_ind_begin + inner_ind_delta];
                }
            }
            __syncthreads();
            #pragma unroll
            for(int j = 0; j < UNROLL_Y; j++) {
                int inner_ind_delta = threadIdx.y * UNROLL_Y + j; // By sxjscience, do so to reduce the writing conflicts in atomic operations.
                #pragma unroll
                for(int i = 0; i < UNROLL_X; i++) {
                    int c_delta = i * WARP_SZ + threadIdx.x;
                    if(inner_ind_begin + inner_ind_delta < ind_end && c_begin + c_delta < feat_dim) {
                        if(reduce_type == SegReduceType::kSum || reduce_type == SegReduceType::kMean) {
                            atomicAdd(&dst_shared[seg_ids_shared[inner_ind_delta]][c_delta],
                                      data[IND3(batch_id, indices_shared[inner_ind_delta], c_begin + c_delta, node_num, feat_dim)]);
                        } else if(reduce_type == SegReduceType::kMax) {
                            atomicMax(&dst_shared[seg_ids_shared[inner_ind_delta]][c_delta],
                                      data[IND3(batch_id, indices_shared[inner_ind_delta], c_begin + c_delta, node_num, feat_dim)]);
                        } else if(reduce_type == SegReduceType::kMin) {
                            atomicMin(&dst_shared[seg_ids_shared[inner_ind_delta]][c_delta],
                                      data[IND3(batch_id, indices_shared[inner_ind_delta], c_begin + c_delta, node_num, feat_dim)]);
                        }
                    }
                }
            }
            __syncthreads();
        }
        // Write back the dst_shared to the global memory
        #pragma unroll
        for(int j=0; j < UNROLL_NODE; j++) {
            int seg_delta = j * TY_SZ + threadIdx.y;
            #pragma unroll
            for(int i = 0; i < UNROLL_X; i++) {
                int c_delta = i * WARP_SZ + threadIdx.x;
                if(c_begin + c_delta < feat_dim && seg_begin + seg_delta < seg_num) {
                    int seg_size = indptr[seg_begin + seg_delta + 1] - indptr[seg_begin + seg_delta];
                    if(reduce_type == SegReduceType::kMean && seg_size > 0) {
                        dst[IND3(batch_id, seg_begin + seg_delta, c_begin + c_delta, seg_num, feat_dim)] = dst_shared[seg_delta][c_delta] / seg_size;
                    } else {
                        dst[IND3(batch_id, seg_begin + seg_delta, c_begin + c_delta, seg_num, feat_dim)] = dst_shared[seg_delta][c_delta];
                    }
                }
            }
        }
    }
}














/*Compute the backward pass of SegTakeKCorr w.r.t embed1 when inner product is used.

dst: Shape (K, node_num, feat_dim)
g_out: Shape (K, nnz)
embed2: Shape (K, neighbor_node_num, feat_dim)
neighbor_ids: Shape (nnz, )
neighbor_ind_ptr: Shape(node_num + 1, )
rev_node_ids : Shape(nnz, ), The reverse mapping from 0->nnz-1 to node_ids


for k = 0 to K-1
    for i = 0  to node_num - 1
        dst[k, i, :] = 0
        for j = ind_ptr[i] to ind_ptr[i+1] - 1
            dst[k, i, :] += g_out[k, j] * embed2[k, neighbor_ids[j], :]
*/
template<int TY_SZ = 16, int UNROLL_Y = 8, int UNROLL_X = 4, int WARP_SZ = 32>
__global__ void
__launch_bounds__(TY_SZ * WARP_SZ)
SegTakeKCorrBackwardEmbed1AtomicKernel(float* __restrict__ dst,
                                       const float* g_out,
                                       const float* embed2,
                                       const int* neighbor_ids,
                                       const int* neighbor_ind_ptr,
                                       const int* rev_node_ids,
                                       int K, int node_num, int neighbor_node_num,
                                       int nnz, int feat_dim) {
    int k = blockIdx.z;
    int c_begin = blockIdx.y * WARP_SZ * UNROLL_X; // We deal with a bunch of channels and scan through the neighboring nodes to write to the destination
    __shared__ float dst_shared[TY_SZ][WARP_SZ * UNROLL_X];
    __shared__ float g_out_shared[UNROLL_Y * TY_SZ];
    __shared__ int rev_node_ids_shared[UNROLL_Y * TY_SZ];
    __shared__ int neighbor_ids_shared[UNROLL_Y * TY_SZ];
    for(int b_nid = TY_SZ * blockIdx.x; b_nid < node_num; b_nid += TY_SZ * gridDim.x) {
        int e_nid = min(b_nid + TY_SZ, node_num);
        int b_neighbor_ind = neighbor_ind_ptr[b_nid];
        int e_neighbor_ind = neighbor_ind_ptr[e_nid];
        // 1. Initialize the dst_shared to be all zero
        #pragma unroll
        for (int i = 0; i < UNROLL_X; i++) {
            dst_shared[threadIdx.y][i * WARP_SZ + threadIdx.x] = 0;
        }
        // 2. Compute the gradient
        for(int b_ind_inner = b_neighbor_ind; b_ind_inner < e_neighbor_ind; b_ind_inner += UNROLL_Y * TY_SZ) {
            // 2.1 Load to shared memory
            if(threadIdx.y == 0) {
                #pragma unroll
                for(int i = 0; i < CEIL_DIV(UNROLL_Y * TY_SZ, WARP_SZ); i++) {
                    int ind_delta = i * WARP_SZ + threadIdx.x;
                    if(b_ind_inner + ind_delta < e_neighbor_ind && ind_delta < UNROLL_Y * TY_SZ) {
                        g_out_shared[ind_delta] = g_out[k * nnz + b_ind_inner + ind_delta];
                        rev_node_ids_shared[ind_delta] = rev_node_ids[b_ind_inner + ind_delta] - b_nid;
                        neighbor_ids_shared[ind_delta] = neighbor_ids[b_ind_inner + ind_delta];
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for(int j = 0; j < UNROLL_Y; j++) {
                int ind_delta = threadIdx.y * UNROLL_Y + j; // By sxjscience: I do this instead of "j * TY_SZ + threadIdx.y" to reduce the possibility that the atomic_adds will write to the same index.
                #pragma unroll
                for (int i = 0; i < UNROLL_X; i++) {
                    int c_delta = i * WARP_SZ + threadIdx.x;
                    if (c_begin + c_delta < feat_dim && ind_delta + b_ind_inner < e_neighbor_ind) {
                        atomicAdd(&dst_shared[rev_node_ids_shared[ind_delta]][c_delta], g_out_shared[ind_delta] * embed2[IND3(k, neighbor_ids_shared[ind_delta], c_begin + c_delta, neighbor_node_num, feat_dim)]);
                    }
                }
            }
            __syncthreads();
        }
        // 3. Write the shared variable back to the global memory
        #pragma unroll
        for (int i = 0; i < UNROLL_X; i++) {
            int c_delta = i * WARP_SZ + threadIdx.x;
            int nid = b_nid + threadIdx.y;
            if (c_begin + c_delta < feat_dim && nid < node_num) {
                dst[IND3(k, nid, c_begin + c_delta, node_num, feat_dim)] += dst_shared[threadIdx.y][c_delta];
            }
        }
    }
}