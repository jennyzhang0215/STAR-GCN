#include <iostream>
#include <cstring>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>
#include <ctime>
#include <utility>
#include <stdio.h>
#include <cuda.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "helper_math.h"
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>

static std::default_random_engine generator(1000);
const int kMaxGridNum = 65535;
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define ASSERT(x) if(!(x)) {std::cout << #x << " does not hold! File =" << __FILE__ << " Line=" << __LINE__ << std::endl;exit(0);}

#define IND2(x, y, sy) ((x) * (sy) + (y))
#define IND3(x, y, z, sy, sz) (IND2(x, y, sy) * (sz) + (z))
#define CUDA_POST_KERNEL_CHECK(x) \
  do { \
    cudaError err = cudaPeekAtLastError(); \
    if(err != cudaSuccess) {std::cout << "Name: " << #x << " Line: " << __LINE__ << " ErrStr:" << cudaGetErrorString(err) << std::endl;exit(0);} \
  } while (0)


enum SegReduceType {kSum, kMean, kMax, kMin};
enum SegTakeKCorrType { kInnerProduct, kEuclidean };


std::pair<dim3, dim3> KernelLauchParamB1G2(int total_count) {
    const int thread_num = 256;
    dim3 dimBlock(thread_num);
    int grid_size = CEIL_DIV(total_count, thread_num);
    int grid_dim_x = grid_size > kMaxGridNum ? 1024 : grid_size;
    int grid_dim_y = grid_size > kMaxGridNum ? CEIL_DIV(grid_size, 1024) : 1;
    dim3 dimGrid(grid_dim_x, grid_dim_y);
    return std::make_pair(dimBlock, dimGrid);
}

std::pair<dim3, dim3> KernelLauchParamB1G1(int total_count) {
    const int thread_num = 256;
    dim3 dimBlock(thread_num);
    int grid_size = CEIL_DIV(total_count, thread_num);
    int grid_dim_x = grid_size > kMaxGridNum ? kMaxGridNum : grid_size;
    dim3 dimGrid(grid_dim_x);
    return std::make_pair(dimBlock, dimGrid);
}

class GPUTimer {
public:
    GPUTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        cudaEventRecord(start_);
    }

    float stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        return milliseconds;
    }
private:
    cudaEvent_t start_, stop_;
};

template<int UNROLL_X, typename DType>
__device__ void SumSharedMem(volatile DType* data) {
    if(UNROLL_X == 1) {
        if(threadIdx.x < 16) {
            data[threadIdx.x] += data[threadIdx.x + 16];
            data[threadIdx.x] += data[threadIdx.x + 8];
            data[threadIdx.x] += data[threadIdx.x + 4];
            data[threadIdx.x] += data[threadIdx.x + 2];
            data[threadIdx.x] += data[threadIdx.x + 1];
        }
    } else {
        //TODO Enable arbitrary UNROLL_X
        if(UNROLL_X >= 8) {
            data[threadIdx.x] += data[threadIdx.x + 128];
            data[threadIdx.x + 32] += data[threadIdx.x + 128 + 32];
            data[threadIdx.x + 64] += data[threadIdx.x + 128 + 64];
            data[threadIdx.x + 96] += data[threadIdx.x + 128 + 96];
        }
        if(UNROLL_X >= 4) {
            data[threadIdx.x] += data[threadIdx.x + 64];
            data[threadIdx.x + 32] += data[threadIdx.x + 96];
        }
        data[threadIdx.x] += data[threadIdx.x + 32];
        data[threadIdx.x] += data[threadIdx.x + 16];
        data[threadIdx.x] += data[threadIdx.x + 8];
        data[threadIdx.x] += data[threadIdx.x + 4];
        data[threadIdx.x] += data[threadIdx.x + 2];
        data[threadIdx.x] += data[threadIdx.x + 1];
    }
}


/*Fills the selected position in the destination with the index
Important! Here we assume that the sel is sorted!

for i = 0 to seg_num - 1
    if(ind_ptr[idx] == ind_ptr[idx + 1]) dst[sel[i]] = i
*/
__global__ void FillSegStartIndex(int* __restrict__ dst, const int* __restrict__ ind_ptr, int seg_num) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < seg_num; idx += blockDim.x * gridDim.x) {
        if (ind_ptr[idx] != ind_ptr[idx + 1]) {
            dst[ind_ptr[idx]] = idx;
        }
    }
}

struct GetSegId {
    static size_t get_temp_bytes(int nnz) {
        size_t temp_storage_bytes = 0;
        cub::Max max_op;
        cub::DeviceScan::InclusiveScan<int*, int*>(NULL, temp_storage_bytes, NULL, NULL, max_op, nnz);
        return temp_storage_bytes;
    }

    static void compute(int* seg_ids, const int* ind_ptr, int seg_num, int nnz, char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
        cudaMemsetAsync(seg_ids, 0, sizeof(int) * nnz, stream);
        std::pair<dim3, dim3> block_grid_dim3 = KernelLauchParamB1G1(seg_num);
        FillSegStartIndex <<<block_grid_dim3.second, block_grid_dim3.first, 0, stream>>> (seg_ids, ind_ptr, seg_num);
        cub::Max max_op;
        cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes, seg_ids, seg_ids, max_op, nnz, stream);
        return;
    }
};


struct MinOpCUB {
    cub::Min cub_op;
    template<typename DType>
    __forceinline__ DType init_value() const {
        return std::numeric_limits<DType>::max();
    }
};

struct MaxOpCUB {
    cub::Max cub_op;
    template<typename DType>
    __forceinline__ DType init_value() const {
        return std::numeric_limits<DType>::lowest();
    }
};

struct SumOpCUB {
    cub::Sum cub_op;
    template<typename DType>
    __forceinline__ DType init_value() const {
        return static_cast<DType>(0);
    }
};

/*Expand the indptr to make it repeat k times and increase the values of the indices during the expansion

for i = 0 to repeat_num - 1
for j = 0 to seg_num - 1
dst[i * seg_num + j] = i * nnz + ind_ptr[j];
dst[repeat_num * seg_num] = repeat_num * nnz
*/
__global__ void ExpandIndptr(int* __restrict__ dst, const int* __restrict__ ind_ptr, int seg_num, int nnz, int repeat_num) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx <= seg_num * repeat_num; idx += blockDim.x * gridDim.x) {
        if (idx < seg_num * repeat_num) {
            int seg_id = idx % seg_num;
            int repeat_id = idx / seg_num;
            dst[idx] = ind_ptr[seg_id] + repeat_id * nnz;
        } else {
            dst[idx] = repeat_num * nnz;
        }
    }
}

template<typename ReduceOp, typename DType>
struct BatchSegReduceContig {
    static size_t get_temp_bytes(int batch_num, int nnz, int seg_num) {
        size_t temp_storage_bytes = 0;
        ReduceOp op;
        cub::DeviceSegmentedReduce::Reduce<DType*, DType*>(NULL, temp_storage_bytes, NULL, NULL,
                                                           batch_num * seg_num, NULL, NULL,
                                                           op.cub_op, op.template init_value<DType>());
        temp_storage_bytes += sizeof(int) * (seg_num * batch_num + 1); // Size for the new expaned ind_ptr
        return temp_storage_bytes;
    }

    static void compute(DType* dst, const DType* src, const int* ind_ptr, int batch_num, int nnz, int seg_num, char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
        int expanded_ind_ptr_size = seg_num * batch_num + 1;
        ASSERT(temp_storage_bytes >= sizeof(int) * expanded_ind_ptr_size);
        ReduceOp op;
        int* expanded_ind_ptr = reinterpret_cast<int*>(temp_storage);
        std::pair<dim3, dim3> block_grid_dim3 = KernelLauchParamB1G1(expanded_ind_ptr_size);
        ExpandIndptr <<<block_grid_dim3.second, block_grid_dim3.first, 0, stream>>> (expanded_ind_ptr, ind_ptr, seg_num, nnz, batch_num);
        //TODO(sxjscience) Use MSHADOW_CUDA_POST_KERNEL_CHECK
        CUDA_POST_KERNEL_CHECK(ExpandIndptr);
        temp_storage_bytes -= sizeof(int) * expanded_ind_ptr_size;
        char* cub_temp_storage = temp_storage + sizeof(int) * expanded_ind_ptr_size;
        cub::DeviceSegmentedReduce::Reduce(cub_temp_storage,
                                           temp_storage_bytes,
                                           src, dst, batch_num * seg_num,
                                           expanded_ind_ptr, expanded_ind_ptr + 1, op.cub_op, op.template init_value<DType>(), stream);
    }
};

struct plus {
    template<typename DType>
    __host__ __device__ __forceinline__ static DType Map(DType a, DType b) {
        return a + b;
    }
};

struct minus {
    template<typename DType>
    __host__ __device__ __forceinline__ static DType Map(DType a, DType b) {
        return a - b;
    }
};

struct div {
    template<typename DType>
    __host__ __device__ __forceinline__ static DType Map(DType a, DType b) {
        return a / b;
    }
};

struct mul {
    template<typename DType>
    __host__ __device__ __forceinline__ static DType Map(DType a, DType b) {
        return a * b;
    }
};

struct right {
    template<typename DType>
    __host__ __device__ __forceinline__ static DType Map(DType a, DType b) {
        return b;
    }
};

struct diff_square {
    template<typename DType>
    __host__ __device__ __forceinline__ static DType Map(DType a, DType b) {
        return (a - b) * (a - b);
    }
};

/* Compute broadcast rhs and apply the binary OP between lhs and rhs. Add the result to dst

dst : Shape (batch_num, nnz)
lhs : Shape (batch_num, nnz)
rhs : Shape (batch_num, seg_num)
seg_ids: Shape(nnz,)

for i = 0 to batch_num - 1
    for j = 0 to nnz - 1
        dst[i, j] += OP::Map(lhs[i, j], rhs[i, seg_ids[j]])
*/
template<typename OP, bool add_to, typename DType>
__global__ void SegBroadcastBinaryContigKernel(DType* __restrict__ dst,
                                               const DType* lhs,
                                               const DType* rhs,
                                               const int* seg_ids,
                                               int batch_num, int nnz, int seg_num) {
    for(int idx= threadIdx.x + blockDim.x * blockIdx.x; idx < batch_num * nnz; idx += blockDim.x * gridDim.x) {
        int batch_id = idx / nnz;
        int ele_id = idx % nnz;
        DType res = OP::Map(lhs[idx], rhs[batch_id * seg_num + seg_ids[ele_id]]);
        if(add_to) {
            dst[idx] += res;
        } else {
            dst[idx] = res;
        }
    }
}

struct BatchSegBroadcastBinaryContig {
    static size_t get_temp_bytes(int nnz) {
        size_t temp_storage_bytes = GetSegId::get_temp_bytes(nnz);
        temp_storage_bytes += sizeof(int) * nnz; // Size of temp seg_ids
        return temp_storage_bytes;
    }

    template<typename OP, bool add_to, typename DType>
    static void compute(DType* dst, const DType* lhs, const DType* rhs, const int* ind_ptr, int batch_num, int nnz, int seg_num,
                        char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
        int* seg_ids = reinterpret_cast<int*>(temp_storage);
        GetSegId::compute(seg_ids, ind_ptr, seg_num, nnz,
                          temp_storage + sizeof(int) * nnz, temp_storage_bytes - sizeof(int) * nnz, stream);
        std::pair<dim3, dim3> block_grid_dim3 = KernelLauchParamB1G1(batch_num * nnz);
        SegBroadcastBinaryContigKernel<OP, add_to> <<<block_grid_dim3.second, block_grid_dim3.first, 0, stream>>>
            (dst, lhs, rhs, seg_ids, batch_num, nnz, seg_num);
        CUDA_POST_KERNEL_CHECK(SegBroadcastBinaryContigKernel);
    }
};


struct BatchSegSoftmaxContig {
    static size_t get_temp_bytes(int batch_num, int nnz, int seg_num) {
        return 0;
    }
};

struct BatchSegSoftmaxBackwardContig {
    static size_t get_temp_bytes(int batch_num, int nnz, int seg_num) {
        return 0;
    }
};


/*For all the nodes, computes the inner product between the node and it's neighborhoods and add to dst.

dst: Shape (K, nnz)
embed1: Shape (K, node_num, feat_dim)
embed2: Shape (K, neighbor_node_num, feat_dim)
neighbor_ids: Shape (nnz, )
neighbor_ind_ptr: Shape(node_num + 1, )
rev_node_ids : Shape(nnz, ), The reverse mapping from 0->nnz-1 to node_ids


use mul to compute the inner-product and use squared_diff to compute the squared distance.

for k = 0 to K-1
    for i = 0  to node_num - 1
        for j = ind_ptr[i] to ind_ptr[i+1] - 1
            neighbor_id = neighbor_ids[j]
            dst[k, j] += InnerProduct(embed1[k, i], embed2[k, neighbor_id]) or ||embed1[k, i] - embed2[k, neighbor_id]||^2_2

*/
template<typename OP, int UNROLL_NODE = 1, int TY_SZ = 16, int UNROLL_Y = 4, int UNROLL_X = 4, int WARP_SZ = 32>
__global__ void
__launch_bounds__(TY_SZ * WARP_SZ)
SegTakeKCorrKernel(float* __restrict__ dst,
                   const float* embed1,
                   const float* embed2,
                   const int* neighbor_ids,
                   const int* neighbor_ind_ptr,
                   const int* rev_node_ids,
                   int K, int node_num, int neighbor_node_num,
                   int nnz, int feat_dim) {
    int k = blockIdx.y;
    __shared__ float embed1_shared[UNROLL_NODE * TY_SZ][WARP_SZ * UNROLL_X];
    __shared__ float embed2_shared[TY_SZ][WARP_SZ * UNROLL_X];
    __shared__ int rev_node_ids_shared[UNROLL_Y * TY_SZ];
    __shared__ int neighbor_ids_shared[UNROLL_Y * TY_SZ];
    __shared__ float dst_shared[UNROLL_Y * TY_SZ]; // Shared variable to store the result that should be saved to dst
    for(int c_begin = 0; c_begin < feat_dim; c_begin += WARP_SZ * UNROLL_X) { // We deal with a bunch of channels and scan through the neighboring nodes to write to the destination
        for(int b_nid = UNROLL_NODE * TY_SZ * blockIdx.x; b_nid < node_num; b_nid += UNROLL_NODE * TY_SZ * gridDim.x) {
            int e_nid = min(b_nid + UNROLL_NODE * TY_SZ, node_num);
            int b_neighbor_ind = neighbor_ind_ptr[b_nid];
            int e_neighbor_ind = neighbor_ind_ptr[e_nid];
            // 1. Load embed1 to shared memory
            #pragma unroll
            for(int j = 0; j < UNROLL_NODE; j++) {
                int nid_delta = j * TY_SZ + threadIdx.y;
                #pragma unroll
                for(int i = 0; i < UNROLL_X; i++) {
                    int c_delta = i * WARP_SZ + threadIdx.x;
                    if(c_begin + c_delta < feat_dim && b_nid + nid_delta < e_nid) {
                        embed1_shared[nid_delta][c_delta] = embed1[IND3(k, b_nid + nid_delta, c_begin + c_delta, node_num, feat_dim)];
                    } else {
                        embed1_shared[nid_delta][c_delta] = 0.0f;
                    }
                }
            }
            // 2. Compute the inner product between embed1 and embed2
            for(int b_ind_inner = b_neighbor_ind; b_ind_inner < e_neighbor_ind; b_ind_inner += UNROLL_Y * TY_SZ) {
                int e_ind_inner = min(b_ind_inner + UNROLL_Y * TY_SZ, e_neighbor_ind);
                // 2.1 Initilaize the shared dst variables to zero.
                #pragma unroll
                for(int i = 0; i < CEIL_DIV(UNROLL_Y * TY_SZ, WARP_SZ); i++) {
                    if(threadIdx.y == 0) dst_shared[i * WARP_SZ + threadIdx.x] = 0.0f;
                }
                // 2.2 Load the rev_node_ids and neighbor_node_ids to shared memory
                if(threadIdx.y == 0) {
                    #pragma unroll
                    for(int i = 0; i < CEIL_DIV(UNROLL_Y * TY_SZ, WARP_SZ); i++) {
                        if (b_ind_inner + i * WARP_SZ + threadIdx.x < e_ind_inner && i * WARP_SZ + threadIdx.x < UNROLL_Y * TY_SZ) {
                            rev_node_ids_shared[i * WARP_SZ + threadIdx.x] = rev_node_ids[b_ind_inner + i * WARP_SZ + threadIdx.x] - b_nid;
                            neighbor_ids_shared[i * WARP_SZ + threadIdx.x] = neighbor_ids[b_ind_inner + i * WARP_SZ + threadIdx.x];
                        }
                    }
                }
                __syncthreads();
                // 2.3 Load embed2 to shared memory and do the computation
                #pragma unroll
                for(int j = 0; j < UNROLL_Y; j++) {
                    int ind_inner_delta = j * TY_SZ + threadIdx.y;
                    // 2.3.1 Perform the loading
                    #pragma unroll
                    for(int i = 0; i < UNROLL_X; i++) {
                        int c_delta = i * WARP_SZ + threadIdx.x;
                        if(c_delta + c_begin < feat_dim && b_ind_inner + ind_inner_delta < e_ind_inner) {
                            // Load and perform the binary operator
                            // TODO(sxjscience) potential overflow problem, consider use size_t instead
                            embed2_shared[threadIdx.y][c_delta] = OP::Map(embed2[IND3(k, neighbor_ids_shared[ind_inner_delta], c_delta + c_begin, neighbor_node_num, feat_dim)],
                                                                            embed1_shared[rev_node_ids_shared[ind_inner_delta]][c_delta]);
                        } else {
                            embed2_shared[threadIdx.y][c_delta] = 0.0f;
                        }
                    }
                    // 2.3.2 Perform the reduction
                    SumSharedMem<UNROLL_X>(embed2_shared[threadIdx.y]);
                    // 2.3.3 Accumulate the result to the local dst variable
                    if(threadIdx.x == 0) dst_shared[j * TY_SZ + threadIdx.y] += embed2_shared[threadIdx.y][0];
                    __syncthreads();
                }
                // 2.4 Write the shared variable back to the global memory
                if(threadIdx.y == 0) {
                    #pragma unroll
                    for (int i = 0; i < CEIL_DIV(UNROLL_Y * TY_SZ, WARP_SZ); i++) {
                        if (b_ind_inner + i * WARP_SZ + threadIdx.x < e_ind_inner && i * WARP_SZ + threadIdx.x < UNROLL_Y * TY_SZ) {
                            dst[k * nnz + b_ind_inner + i * WARP_SZ + threadIdx.x] += dst_shared[i * WARP_SZ + threadIdx.x];
                        }
                    }
                }
                __syncthreads();
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


for k = 0 to K-1
    for i = 0  to node_num - 1
        dst[k, i, :] = 0
        for j = ind_ptr[i] to ind_ptr[i+1] - 1
            dst[k, i, :] += g_out[k, j] * embed2[k, neighbor_ids[j], :]
*/
template<int UNROLL_X = 4, int TX_SZ = 32>
__global__ void
__launch_bounds__(TX_SZ)
SegTakeKCorrBackwardEmbed1Kernel(float* __restrict__ dst,
                                 const float* g_out,
                                 const float* embed2,
                                 const int* neighbor_ids,
                                 const int* neighbor_ind_ptr,
                                 int K, int node_num, int neighbor_node_num,
                                 int nnz, int feat_dim) {
    int k = blockIdx.z;
    int c_begin = blockIdx.y * TX_SZ * UNROLL_X; // We deal with a bunch of channels and scan through the neighboring nodes to write to the destination
    float dst_local[UNROLL_X];
    for (int nid = blockIdx.x; nid < node_num; nid += gridDim.x) {
        int b_neighbor_ind = neighbor_ind_ptr[nid];
        int e_neighbor_ind = neighbor_ind_ptr[nid + 1];
        #pragma unroll
        for(int i = 0; i < UNROLL_X; i++) {
            int c = c_begin + i * TX_SZ + threadIdx.x;
            if(c < feat_dim) {
                dst_local[i] = dst[IND3(k, nid, c, node_num, feat_dim)];
            }
        }
        for(int j = b_neighbor_ind; j < e_neighbor_ind; j++) {
            #pragma unroll
            for(int i = 0; i < UNROLL_X; i++) {
                int c = c_begin + i * TX_SZ + threadIdx.x;
                if (c < feat_dim) {
                    dst_local[i] += g_out[k * nnz + j] * embed2[IND3(k, neighbor_ids[j], c, neighbor_node_num, feat_dim)];
                }
            }
        }
        #pragma unroll
        for(int i = 0; i < UNROLL_X; i++) {
            int c = c_begin + i * TX_SZ + threadIdx.x;
            if(c < feat_dim) {
                dst[IND3(k, nid, c, node_num, feat_dim)] = dst_local[i];
            }
        }
    }
}

__global__ void IdxArrayKernel(int* dst, int size) {
  for(int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
    dst[idx] = idx;
  }
}

/*Compute the backward pass of SegTakeKCorr w.r.t embed2 when inner product is used.

dst: Shape (K, neighbor_node_num, feat_dim)
g_out: Shape (K, nnz)
embed1: Shape (K, node_num, feat_dim)
sorted_neighbor_ids: Shape (nnz, )
original_inds: Shape(nnz, )
rev_node_ids : Shape(nnz, ), The reverse mapping from 0->nnz-1 to node_ids

for k = 0 to K - 1
    for i = 0  to nnz - 1
      dst_ind = sorted_neighbor_ids[i]
      src_ind = sorted_inds[i]
      dst[k, dst_ind, :] += g_out[k, src_ind] * embed1[k, rev_node_ids[src_ind], :]

*/
template<int UNROLL_X = 4, int TX_SZ = 32, int TY_SZ = 1>
__global__ void
__launch_bounds__(TX_SZ * TY_SZ)
SegTakeKCorrBackwardEmbed2Kernel(float* dst,
                                 const float* g_out,
                                 const float* embed1,
                                 const int* sorted_neighbor_ids,
                                 const int* sorted_inds,
                                 const int* rev_node_ids,
                                 int K, int node_num, int neighbor_node_num,
                                 int nnz, int feat_dim) {
  int k = blockIdx.z;
  int c_begin = blockIdx.y * TX_SZ * UNROLL_X; // We deal with a bunch of channels and scan through the neighboring nodes to write to the destination
  float dst_local[UNROLL_X];
  int idx = blockIdx.x * TY_SZ + threadIdx.y;
  if (idx < nnz && (idx == 0 || sorted_neighbor_ids[idx] != sorted_neighbor_ids[idx - 1])) {
    const int dst_ind = sorted_neighbor_ids[idx];
    #pragma unroll
    for (int i = 0; i < UNROLL_X; i++) {
      int c = c_begin + i * TX_SZ + threadIdx.x;
      if (c < feat_dim) {
        dst_local[i] = dst[IND3(k, dst_ind, c, neighbor_node_num, feat_dim)];
      }
    }
    do {
      const int src_ind = sorted_inds[idx];
      #pragma unroll
      for(int i = 0; i < UNROLL_X; i++) {
          int c = c_begin + i * TX_SZ + threadIdx.x;
          if (c < feat_dim) {
              dst_local[i] += g_out[k * nnz + src_ind] * embed1[IND3(k, rev_node_ids[src_ind], c, node_num, feat_dim)];
          }
      }
      idx++;
    } while (idx < nnz && (sorted_neighbor_ids[idx] == sorted_neighbor_ids[idx - 1]));
    #pragma unroll
    for (int i = 0; i < UNROLL_X; i++) {
      int c = c_begin + i * TX_SZ + threadIdx.x;
      if (c < feat_dim) {
        dst[IND3(k, dst_ind, c, neighbor_node_num, feat_dim)] = dst_local[i];
      }
    }
  }
}

/*Compute the backward pass of SegTakeKCorr w.r.t embed2 when inner product is used.

dst: Shape (K, neighbor_node_num, feat_dim)
g_out: Shape (K, nnz)
embed1: Shape (K, node_num, feat_dim)
neighbor_ids: Shape (nnz, )
neighbor_ind_ptr: Shape(node_num + 1, )
rev_node_ids : Shape(nnz, ), The reverse mapping from 0->nnz-1 to node_ids


for k = 0 to K-1
    for i = 0  to node_num - 1
        for j = ind_ptr[i] to ind_ptr[i+1] - 1
            dst[k, neighbor_ids[j], :] += g_out[k, j] * embed1[k, rev_node_ids[j], :]

TODO(sxjscience) Optimize the speed of this function
First reorganize the data in neighbor_ids, g_out, embed1, ...

for k = 0 to K-1
    for i = 0 to neighbor_node_num - 1
        for j = rev_ind_ptr[i] to rev_ind_ptr[i + 1] - 1
            dst[k, i, :] += reorder_g_out[k, j] * embed1[k, node_ids[j], :]

*/
template<int TY_SZ = 16, int UNROLL_Y = 4, int UNROLL_X = 4, int WARP_SZ = 32>
__global__ void
__launch_bounds__(TY_SZ * WARP_SZ)
SegTakeKCorrBackwardEmbed2KernelAtomic(float* __restrict__ dst,
                                 const float* g_out,
                                 const float* embed1,
                                 const int* neighbor_ids,
                                 const int* neighbor_ind_ptr,
                                 const int* rev_node_ids,
                                 int K, int node_num, int neighbor_node_num,
                                 int nnz, int feat_dim) {
    int k = blockIdx.z;
    int c_begin = blockIdx.y * WARP_SZ * UNROLL_X; // We deal with a bunch of channels and scan through the neighboring nodes to write to the destination
    __shared__ float embed1_shared[TY_SZ][WARP_SZ * UNROLL_X];
    __shared__ float g_out_shared[TY_SZ * UNROLL_Y];
    __shared__ int neighbor_ids_shared[TY_SZ * UNROLL_Y];
    __shared__ int rev_node_ids_shared[TY_SZ * UNROLL_Y];
    for(int b_nid = TY_SZ * blockIdx.x; b_nid < node_num; b_nid += TY_SZ * gridDim.x) {
        int e_nid = min(b_nid + TY_SZ, node_num);
        int b_neighbor_ind = neighbor_ind_ptr[b_nid];
        int e_neighbor_ind = neighbor_ind_ptr[e_nid];
        // 1. Load embed1 to shared memory
        #pragma unroll
        for(int i = 0; i < UNROLL_X; i++) {
            int c_delta = i * WARP_SZ + threadIdx.x;
            if(c_begin + c_delta < feat_dim && b_nid + threadIdx.y < e_nid) {
                embed1_shared[threadIdx.y][c_delta] = embed1[IND3(k, b_nid + threadIdx.y, c_begin + c_delta, node_num, feat_dim)];
            } else {
                embed1_shared[threadIdx.y][c_delta] = 0.0f;
            }
        }
        // 2. Write back the gradient, use atomic write
        for(int b_ind_inner = b_neighbor_ind; b_ind_inner < e_neighbor_ind; b_ind_inner += TY_SZ * UNROLL_Y) {
            int e_ind_inner = min(b_ind_inner + UNROLL_Y * TY_SZ, e_neighbor_ind);
            // 2.1 Load g_out
            if(threadIdx.y == 0) {
                #pragma unroll
                for (int i = 0; i < CEIL_DIV(UNROLL_Y * TY_SZ, WARP_SZ); i++) {
                    int ind_delta = i * WARP_SZ + threadIdx.x;
                    if (b_ind_inner + ind_delta < e_ind_inner  && ind_delta < UNROLL_Y * TY_SZ) {
                        g_out_shared[ind_delta] = g_out[k * nnz + b_ind_inner + ind_delta];
                        neighbor_ids_shared[ind_delta] = neighbor_ids[b_ind_inner + ind_delta];
                        rev_node_ids_shared[ind_delta] = rev_node_ids[b_ind_inner + ind_delta] - b_nid;
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for(int j = 0; j < UNROLL_Y; j++) {
                int ind_delta = j * TY_SZ + threadIdx.y;
                float g_out_ele = g_out_shared[ind_delta];
                int neighbor_id = neighbor_ids_shared[ind_delta];
                int rev_node_id = rev_node_ids_shared[ind_delta];
                #pragma unroll
                for(int i = 0; i < UNROLL_X; i++) {
                    int c_delta = i * WARP_SZ + threadIdx.x;
                    if(c_delta + c_begin < feat_dim && b_ind_inner + ind_delta < e_ind_inner) {
                        atomicAdd(dst + IND3(k, neighbor_id, c_delta + c_begin, neighbor_node_num, feat_dim),
                                    g_out_ele * embed1_shared[rev_node_id][c_delta]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

struct SegTakeKCorr {
    static size_t get_temp_bytes(int nnz) {
        size_t temp_storage_bytes = GetSegId::get_temp_bytes(nnz);
        temp_storage_bytes += sizeof(int) * nnz; // Size of temp seg_ids
        return temp_storage_bytes;
    }

    static size_t get_sort_temp_bytes(int nnz, int node_num) {
      size_t temp_storage_bytes = 0;
      cub::DeviceRadixSort::SortPairs<int, int>(NULL, temp_storage_bytes, NULL, NULL, NULL, NULL, nnz);
      return temp_storage_bytes;
    }

    static size_t get_temp_bytes_backward_embed2(int nnz, int seg_num) {
      size_t temp_storage_bytes = GetSegId::get_temp_bytes(nnz);
      temp_storage_bytes += get_sort_temp_bytes(nnz, seg_num);
      temp_storage_bytes += sizeof(int) * nnz; // Size of temp seg_ids
      temp_storage_bytes += sizeof(int) * nnz; // Size of temp sorted_neighbor_ids
      temp_storage_bytes += sizeof(int) * nnz; // Size of temp value_in
      temp_storage_bytes += sizeof(int) * nnz; // Size of temp value_out
      return temp_storage_bytes;
    }

    template<bool add_to>
    static void compute(float* dst, const float* embed1, const float* embed2,
                        const int* neighbor_ids, const int* neighbor_ind_ptr,
                        int K, int node_num, int neighbor_node_num, int nnz, int feat_dim, int type,
                        char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
        long long K_ll = static_cast<long long>(K);
        long long node_num_ll = static_cast<long long>(node_num);
        long long neighbor_node_num_ll = static_cast<long long>(neighbor_node_num);
        long long feat_dim_ll = static_cast<long long>(feat_dim);
        long long int_max_ll = static_cast<long long>(std::numeric_limits<int>::max());
        ASSERT(K_ll * node_num_ll * feat_dim_ll < int_max_ll);
        ASSERT(K_ll * neighbor_node_num_ll * feat_dim_ll < int_max_ll);
        if(!add_to) {
            cudaMemsetAsync(dst, 0, sizeof(float) * K * nnz, stream);
        }
        int* rev_node_ids = reinterpret_cast<int*>(temp_storage);
        GetSegId::compute(rev_node_ids, neighbor_ind_ptr, node_num, nnz,
                          temp_storage + sizeof(int) * nnz, temp_storage_bytes - sizeof(int) * nnz, stream);
        static const int UNROLL_NODE = 1;
        static const int TY_SZ = 16;
        static const int UNROLL_Y = 4;
        static const int UNROLL_X = 4;
        static const int WARP_SZ = 32;
        dim3 dimBlock(WARP_SZ, TY_SZ);
        dim3 dimGrid(CEIL_DIV(node_num, UNROLL_NODE * TY_SZ), K);
        if (type == SegTakeKCorrType::kInnerProduct) {
            SegTakeKCorrKernel<mul, UNROLL_NODE, TY_SZ, UNROLL_Y, UNROLL_X, WARP_SZ> <<<dimGrid, dimBlock, 0, stream >>>
                (dst, embed1, embed2, neighbor_ids, neighbor_ind_ptr, rev_node_ids, K, node_num, neighbor_node_num, nnz, feat_dim);
        } else if (type == SegTakeKCorrType::kEuclidean) {
            SegTakeKCorrKernel<diff_square, UNROLL_NODE, TY_SZ, UNROLL_Y, UNROLL_X, WARP_SZ> <<<dimGrid, dimBlock, 0, stream >>>
                (dst, embed1, embed2, neighbor_ids, neighbor_ind_ptr, rev_node_ids, K, node_num, neighbor_node_num, nnz, feat_dim);
        } else {
            ASSERT(false);
        }
        CUDA_POST_KERNEL_CHECK(SegTakeKCorrKernel);
    }

    template<bool add_to>
    static void compute_grad_embed1(float* dst, const float* g_out, const float* embed2,
                                    const int* neighbor_ids, const int* neighbor_ind_ptr,
                                    int K, int node_num, int neighbor_node_num, int nnz, int feat_dim, int type, cudaStream_t stream) {
        ASSERT(type == SegTakeKCorrType::kInnerProduct);
        long long K_ll = static_cast<long long>(K);
        long long node_num_ll = static_cast<long long>(node_num);
        long long neighbor_node_num_ll = static_cast<long long>(neighbor_node_num);
        long long feat_dim_ll = static_cast<long long>(feat_dim);
        long long int_max_ll = static_cast<long long>(std::numeric_limits<int>::max());
        ASSERT(K_ll * node_num_ll * feat_dim_ll < int_max_ll);
        ASSERT(K_ll * neighbor_node_num_ll * feat_dim_ll < int_max_ll);
        if(!add_to) {
            cudaMemsetAsync(dst, 0, sizeof(float) * K * node_num * feat_dim, stream);
        }
        static const int UNROLL_X = 4;
        static const int TX_SZ = 32;
        dim3 dimBlock(TX_SZ);
        dim3 dimGrid(node_num, CEIL_DIV(feat_dim, TX_SZ * UNROLL_X), K);
        SegTakeKCorrBackwardEmbed1Kernel<UNROLL_X, TX_SZ> <<<dimGrid, dimBlock, 0, stream >>>
                (dst, g_out, embed2, neighbor_ids, neighbor_ind_ptr, K, node_num,
                 neighbor_node_num, nnz, feat_dim);
        CUDA_POST_KERNEL_CHECK(SegTakeKCorrBackwardEmbed1Kernel);
    }

    template<bool add_to>
    static void compute_grad_embed2(float* dst, const float* g_out, const float* embed1,
                                    const int* neighbor_ids, const int* neighbor_ind_ptr,
                                    int K, int node_num, int neighbor_node_num, int nnz, int feat_dim, int type,
                                    char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
        ASSERT(type == SegTakeKCorrType::kInnerProduct);
        long long K_ll = static_cast<long long>(K);
        long long node_num_ll = static_cast<long long>(node_num);
        long long neighbor_node_num_ll = static_cast<long long>(neighbor_node_num);
        long long feat_dim_ll = static_cast<long long>(feat_dim);
        long long int_max_ll = static_cast<long long>(std::numeric_limits<int>::max());
        ASSERT(K_ll * node_num_ll * feat_dim_ll < int_max_ll);
        ASSERT(K_ll * neighbor_node_num_ll * feat_dim_ll < int_max_ll);
        if (!add_to) {
            cudaMemsetAsync(dst, 0, sizeof(float) * K * neighbor_node_num * feat_dim, stream);
        }
        int* rev_node_ids = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        int* sorted_neighbor_ids = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        int* temp_ind_in = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        int* sorted_ind = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        // 1. Sort the neighbor_ids
        std::pair<dim3, dim3> block_grid_dim3 = KernelLauchParamB1G1(nnz);
        IdxArrayKernel <<<block_grid_dim3.second, block_grid_dim3.first, 0, stream>>> (temp_ind_in, nnz);
        size_t temp_sort_bytes = get_sort_temp_bytes(nnz, node_num);
        cub::DeviceRadixSort::SortPairs(temp_storage, temp_sort_bytes,
                                        neighbor_ids, sorted_neighbor_ids,
                                        temp_ind_in, sorted_ind, nnz, 0, sizeof(int) * 8, stream);
        temp_storage += get_sort_temp_bytes(nnz, node_num);
        // 2. Compute the rev mapping
        GetSegId::compute(rev_node_ids, neighbor_ind_ptr, node_num, nnz, temp_storage, GetSegId::get_temp_bytes(nnz), stream);
        // 3. Run the kernel
        static const int UNROLL_X = 4;
        static const int TX_SZ = 32;
        static const int TY_SZ = 1;
        dim3 dimBlock(TX_SZ, TY_SZ);
        dim3 dimGrid(CEIL_DIV(nnz, TY_SZ), CEIL_DIV(feat_dim, TX_SZ * UNROLL_X), K);
        SegTakeKCorrBackwardEmbed2Kernel<UNROLL_X, TX_SZ, TY_SZ> <<<dimGrid, dimBlock, 0, stream >>>
                (dst, g_out, embed1, sorted_neighbor_ids, sorted_ind, rev_node_ids, K, node_num,
                 neighbor_node_num, nnz, feat_dim);
        CUDA_POST_KERNEL_CHECK(SegTakeKCorrBackwardEmbed2Kernel);
    }
};


template<typename OP>
void SegTakeKCorrCPU(float* dst, const float* embed1, const float* embed2,
                     const int* neighbor_ids, const int* neighbor_ind_ptr,
                     int K, int node_num, int neighbor_node_num, int nnz, int feat_dim) {
    for(int k = 0; k < K; k++) {
        #pragma omp parallel for
        for(int i = 0; i < node_num; i++) {
            for (int j = neighbor_ind_ptr[i]; j < neighbor_ind_ptr[i + 1]; j++) {
                // Calculate the distance between embed1[k, i, :] and embed2[k, neighbor_ids[j], :]
                dst[k * nnz + j] = 0;
                for(int c = 0; c < feat_dim; c++) {
                    dst[k * nnz + j] += OP::Map(embed1[IND3(k, i, c, node_num, feat_dim)],
                                                embed2[IND3(k, neighbor_ids[j], c, neighbor_node_num, feat_dim)]);
                }
            }
        }
    }
    return;
}

void SegTakeKCorrCPUBackwardEmbed1(float* dst, const float* g_out, const float* embed2,
                                   const int* neighbor_ids, const int* neighbor_ind_ptr,
                                   int K, int node_num, int neighbor_node_num, int nnz, int feat_dim) {
    for(int k = 0; k < K; k++) {
        #pragma omp parallel for
        for(int i = 0; i < node_num; i++) {
            for(int c = 0; c < feat_dim; c++) {
                dst[IND3(k, i, c, node_num, feat_dim)] = 0;
            }
            for(int j = neighbor_ind_ptr[i]; j < neighbor_ind_ptr[i + 1]; j++) {
                for(int c = 0; c < feat_dim; c++) {
                    dst[IND3(k, i, c, node_num, feat_dim)] += g_out[k * nnz + j] * embed2[IND3(k, neighbor_ids[j], c, neighbor_node_num, feat_dim)];
                }
            }
        }
    }
}

void SegTakeKCorrCPUBackwardEmbed2(float* dst, const float* g_out, const float* embed1,
                                   const int* neighbor_ids, const int* neighbor_ind_ptr,
                                   int K, int node_num, int neighbor_node_num, int nnz, int feat_dim) {
    std::vector<int> seg_ids(nnz);
    for(int i = 0; i < node_num; i++) {
        for(int j = neighbor_ind_ptr[i]; j < neighbor_ind_ptr[i + 1]; j++) {
            seg_ids[j] = i;
        }
    }
    std::memset(dst, 0, sizeof(float) * K * neighbor_node_num * feat_dim);
    #pragma omp parallel for
    for(int k = 0; k < K; k++) {
        for(int i = 0; i < nnz; i++) {
            for(int c = 0; c < feat_dim; c++) {
                dst[IND3(k, neighbor_ids[i], c, neighbor_node_num, feat_dim)] += g_out[k * nnz + i] * embed1[IND3(k, seg_ids[i], c, node_num, feat_dim)];
            }
        }
    }
}

/*Divide the elements in a segmentation by the length of the segmentation.
If the length of the segmentation is zero, no division will take place.

data: Shape(batch_num, seg_num, feat_dim)
indptr: Shape(seg_num + 1,)
*/
template<int UNROLL_X = 4, int WARP_SZ = 32>
__global__ void
__launch_bounds__(WARP_SZ)
BatchDivSegLength(float* data, const int* indptr, int batch_num, int seg_num, int feat_dim) {
    int batch_id = blockIdx.z;
    int c_begin = blockIdx.y * UNROLL_X * WARP_SZ;
    for (int seg_id = blockIdx.x; seg_id < seg_num; seg_id += gridDim.x) {
        int ind_end = indptr[seg_id + 1];
        int ind_begin = indptr[seg_id];
        #pragma unroll
        for(int i = 0; i < UNROLL_X; i++) {
            int c = c_begin + i * WARP_SZ + threadIdx.x;
            if (c < feat_dim && ind_end > ind_begin) {
                data[IND3(batch_id, seg_id, c, seg_num, feat_dim)] /= (ind_end - ind_begin);
            }
        }
    }
}


/*Take the sum/mean/max/min of the data within the segments

dst_value: Shape (batch_num, seg_num, feat_dim)
dst_index: Shape (batch_num, seg_num, feat_dim)
data: Shape (batch_num, total_ind_num, feat_dim)
indices: Shape (nnz, )
indptr: Shape (seg_num + 1, )

for k = 0 to batch_num - 1
    for i = 0 to seg_num - 1
        if max or min:
            initialize dst_index to -1
        for j = indptr[i]  to indptr[i+1] - 1
            if sum:
                dst_value[k, i, :] += data[k, indices[j], :]
            else if max:
                if(dst_value[k, i, :] > data[k, indices[j], :]
                    dst_value[k, i, :] = data[k, indices[j], :]
                    dst_index[k, i, :] = indices[j]
            else if min:
                if(dst_value[k, i, :] < data[k, indices[j], :]
                    dst_value[k, i, :] = data[k, indices[j], :]
                    dst_index[k, i, :] = indices[j]
*/
template<int reduce_type, int UNROLL_X = 4, int WARP_SZ = 32>
__global__ void
__launch_bounds__(WARP_SZ)
SegPoolKernel(float* dst_value, int* dst_index,
                              const float* data, const int* indices, const int* indptr,
                              int batch_num, int seg_num, int feat_dim, int total_ind_num, int nnz) {
    int batch_id = blockIdx.z;
    int c_begin = blockIdx.y * UNROLL_X * WARP_SZ;
    float dst_value_local[UNROLL_X];
    int dst_index_local[UNROLL_X];
    for(int seg_id = blockIdx.x; seg_id < seg_num; seg_id += gridDim.x) {
        int ind_begin = indptr[seg_id];
        int ind_end = indptr[seg_id + 1];
        #pragma unroll
        for(int i = 0; i < UNROLL_X; i++) {
            if(reduce_type == SegReduceType::kSum || reduce_type == SegReduceType::kMean) {
                dst_value_local[i] = 0;
            } else if(reduce_type == SegReduceType::kMax) {
                dst_value_local[i] = -FLT_MAX;
                dst_index_local[i] = -1;
            } else if(reduce_type == SegReduceType::kMin) {
                dst_value_local[i] = FLT_MAX;
                dst_index_local[i] = -1;
            }
        }
        for(int j = ind_begin; j < ind_end; j++) {
            int data_ind = indices[j];
            // Perform the reduction
            #pragma unroll
            for(int i = 0; i < UNROLL_X; i++) {
                int c = c_begin + i * WARP_SZ + threadIdx.x;
                if(c < feat_dim) {
                    float data_val = data[IND3(batch_id, data_ind, c, total_ind_num, feat_dim)];
                    if (reduce_type == SegReduceType::kSum || reduce_type == SegReduceType::kMean) {
                        dst_value_local[i] += data_val;
                    } else if (reduce_type == SegReduceType::kMax) {
                        if(data_val > dst_value_local[i]) {
                            dst_value_local[i] = data_val;
                            dst_index_local[i] = j;
                        }
                    } else if (reduce_type == SegReduceType::kMin) {
                        if (data_val < dst_value_local[i]) {
                            dst_value_local[i] = data_val;
                            dst_index_local[i] = j;
                        }
                    }
                }
            }
        }
        if(reduce_type == SegReduceType::kMean) {
            #pragma unroll
            for(int i = 0; i < UNROLL_X; i++) {
                int c = c_begin + i * WARP_SZ + threadIdx.x;
                if (c < feat_dim && ind_end - ind_begin > 0) {
                    dst_value_local[i] /= (ind_end - ind_begin);
                }
            }
        }
        #pragma unroll
        for(int i = 0; i < UNROLL_X; i++) {
            int c = c_begin + i * WARP_SZ + threadIdx.x;
            if(c < feat_dim) {
                int dst_ind = IND3(batch_id, seg_id, c, seg_num, feat_dim);
                dst_value[dst_ind] = dst_value_local[i];
                if(reduce_type == SegReduceType::kMax || reduce_type == SegReduceType::kMin) {
                    dst_index[dst_ind] = dst_index_local[i];
                }
            }
        }
    }
}

/*Backward pass of the SegPool operator when sum is used

dst: Shape (batch_num, total_ind_num, feat_dim)
g_out: Shape(batch_num, seg_num, feat_dim)
out_index: Shape (batch_num, seg_num, feat_dim)
sorted_indices : Shape (nnz,)
sorted_orig_inds: Shape (nnz, )
seg_ids: Shape(nnz,)
indptr: Shape (seg_num + 1, )

for k = 0 to batch_num - 1
    for i = 0 to seg_num - 1
        for j = indptr[i]  to indptr[i+1] - 1
            if sum:
                dst[k, indices[j], :] += g_out[k, i, :]
            elif mean:
                dst[k, indices[j], :] += g_out[k, i, :] / (indptr[i+1] - indptr[i])
            else:
                dst[k, indices[j], :] += g_out[k, i, :] * (out_index[k, i, :] == indices[j])
Sorted Case ==>
for k = 0 to batch_num - 1
    for i = 0 to nnz - 1
        dst_ind = sorted_indices[i] --> indices[j]
        orig_ind = sorted_orig_inds[i] --> j
        seg_id = seg_ids[orig_ind] --> i
        if sum:
            dst[k, dst_ind, :] += g_out[k, seg_id, :]
        elif mean:
            dst[k, dst_ind, :] += g_out[k, seg_id, :] / (indptr[seg_id + 1] - indptr[seg_id])
        else:
            orig_ind = sorted_orig_inds[i]  --> j
            dst[k, dst_ind, :] += g_out[k, seg_id, :] * (out_index[k, seg_id, :] == orig_ind)

*/
template<int reduce_type, int UNROLL_X = 4, int TX_SZ = 32>
__global__ void
__launch_bounds__(TX_SZ)
SegPoolBackwardKernel(float* dst, const float* g_out, const int* out_index, const int* sorted_indices, const int* sorted_orig_inds, const int* seg_ids,
                      const int* indptr, int batch_num, int seg_num, int feat_dim, int total_ind_num, int nnz) {
  int k = blockIdx.z;
  int c_begin = blockIdx.y * TX_SZ * UNROLL_X; // We deal with a bunch of channels and scan through the neighboring nodes to write to the destination
  float dst_local[UNROLL_X];
  int idx = blockIdx.x;
  if (idx < nnz && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])) {
    const int dst_ind = sorted_indices[idx];
    #pragma unroll
    for (int i = 0; i < UNROLL_X; i++) {
      int c = c_begin + i * TX_SZ + threadIdx.x;
      if (c < feat_dim) {
        dst_local[i] = dst[IND3(k, dst_ind, c, total_ind_num, feat_dim)];
      }
    }
    do {
      const int orig_ind = sorted_orig_inds[idx];
      const int seg_id = seg_ids[orig_ind];
      #pragma unroll
      for(int i = 0; i < UNROLL_X; i++) {
          int c = c_begin + i * TX_SZ + threadIdx.x;
          if (c < feat_dim) {
            if(reduce_type == SegReduceType::kSum) {
              dst_local[i] += g_out[IND3(k, seg_id, c, seg_num, feat_dim)];
            } else if(reduce_type == SegReduceType::kMean) {
              dst_local[i] += g_out[IND3(k, seg_id, c, seg_num, feat_dim)] / (indptr[seg_id + 1] - indptr[seg_id]);
            } else {
              dst_local[i] += g_out[IND3(k, seg_id, c, seg_num, feat_dim)] * (out_index[IND3(k, seg_id, c, seg_num, feat_dim)] == orig_ind);
            }
          }
      }
      idx++;
    } while (idx < nnz && (sorted_indices[idx] == sorted_indices[idx - 1]));
    #pragma unroll
    for (int i = 0; i < UNROLL_X; i++) {
      int c = c_begin + i * TX_SZ + threadIdx.x;
      if (c < feat_dim) {
        dst[IND3(k, dst_ind, c, total_ind_num, feat_dim)] = dst_local[i];
      }
    }
  }
}

template<int reduce_type>
void SegPoolCPU(float* dst_value, int* dst_index, const float* data, const int* indices, const int* indptr,
                                int batch_num, int seg_num, int feat_dim, int total_ind_num, int nnz) {
    for(int k = 0; k < batch_num; k++) {
        #pragma omp parallel for
        for (int i = 0; i < seg_num; i++) {
            for(int c = 0; c < feat_dim; c++) {
                int l_ind = IND3(k, i, c, seg_num, feat_dim);
                if(reduce_type == SegReduceType::kSum || reduce_type == SegReduceType::kMean) {
                    dst_value[l_ind] = 0;
                } else if(reduce_type == SegReduceType::kMax) {
                    dst_value[l_ind] = -FLT_MAX;
                    dst_index[l_ind] = -1;
                } else if(reduce_type == SegReduceType::kMin) {
                    dst_value[l_ind] = FLT_MAX;
                    dst_index[l_ind] = -1;
                }
            }
            for (int j = indptr[i]; j < indptr[i + 1]; j++) {
                for (int c = 0; c < feat_dim; c++) {
                    int l_ind = IND3(k, i, c, seg_num, feat_dim);
                    int r_ind = IND3(k, indices[j], c, total_ind_num, feat_dim);
                    if (reduce_type == SegReduceType::kSum || reduce_type == SegReduceType::kMean) {
                        dst_value[l_ind] += data[r_ind];
                    }
                    else if (reduce_type == SegReduceType::kMax) {
                        if(data[r_ind] > dst_value[l_ind]) {
                            dst_value[l_ind] = data[r_ind];
                            dst_index[l_ind] = j;
                        }
                    }
                    else if (reduce_type == SegReduceType::kMin) {
                        if (data[r_ind] < dst_value[l_ind]) {
                            dst_value[l_ind] = data[r_ind];
                            dst_index[l_ind] = j;
                        }
                    }
                }
            }
            if (reduce_type == SegReduceType::kMean && (indptr[i + 1] - indptr[i]) > 0) {
                for (int c = 0; c < feat_dim; c++) {
                    int l_ind = IND3(k, i, c, seg_num, feat_dim);
                    dst_value[l_ind] /= (indptr[i + 1] - indptr[i]);
                }
            }
        }
    }
    return;
}

/* Calculate the backward pass of the SegPool operation
for k = 0 to batch_num - 1
    for i = 0 to seg_num - 1
        for j = indptr[i]  to indptr[i+1] - 1
            dst[k, indices[j], :] += g_out[k, i, :]
*/
template<int reduce_type>
void SegPoolBackwardCPU(float* dst, const float* g_out, const int* out_index, const int* indices, const int* indptr,
                        int batch_num, int seg_num, int feat_dim, int total_ind_num, int nnz) {
    std::memset(dst, 0, sizeof(float) * batch_num * total_ind_num * feat_dim);
    #pragma omp parallel for
    for(int k = 0; k < batch_num; k++) {
        for(int i = 0; i < seg_num; i++) {
            for(int j = indptr[i]; j < indptr[i + 1]; j++) {
                for(int c = 0; c < feat_dim; c++) {
                    float g_out_local = g_out[IND3(k, i, c, seg_num, feat_dim)];
                    if(reduce_type == SegReduceType::kMean) {
                        g_out_local /= (indptr[i + 1] - indptr[i]);
                    }
                    if(reduce_type == SegReduceType::kSum) {
                      dst[IND3(k, indices[j], c, total_ind_num, feat_dim)] += g_out[IND3(k, i, c, seg_num, feat_dim)];
                    } else if(reduce_type == SegReduceType::kMean) {
                      dst[IND3(k, indices[j], c, total_ind_num, feat_dim)] += g_out[IND3(k, i, c, seg_num, feat_dim)] / (indptr[i + 1] - indptr[i]);
                    } else {
                      dst[IND3(k, indices[j], c, total_ind_num, feat_dim)] += g_out[IND3(k, i, c, seg_num, feat_dim)] *
                                                                              (out_index[IND3(k, i, c, seg_num, feat_dim)] == j);
                    }
                }
            }
        }
    }
}

struct SegPool {
    static size_t get_sort_temp_bytes(int nnz) {
      size_t temp_storage_bytes = 0;
      cub::DeviceRadixSort::SortPairs<int, int>(NULL, temp_storage_bytes, NULL, NULL, NULL, NULL, nnz);
      return temp_storage_bytes;
    }

    static size_t get_temp_bytes_backward(int nnz) {
      size_t temp_storage_bytes = get_sort_temp_bytes(nnz); // Tempspace for sorting
      temp_storage_bytes += GetSegId::get_temp_bytes(nnz);
      temp_storage_bytes += sizeof(int) * nnz; // seg_ids
      temp_storage_bytes += sizeof(int) * nnz; // sorted_indices
      temp_storage_bytes += sizeof(int) * nnz; // temp_ind_in
      temp_storage_bytes += sizeof(int) * nnz; // sorted_orig_inds
      return temp_storage_bytes;
    }

    template<int reduce_type>
    static void compute(float* dst_value, int* dst_index, const float* data, const int* indices, const int* indptr,
                        int batch_num, int seg_num, int feat_dim, int total_ind_num, int nnz, cudaStream_t stream) {
        static const int UNROLL_X = 4;
        static const int WARP_SZ = 32;
        dim3 dimBlock(WARP_SZ);
        dim3 dimGrid(seg_num, CEIL_DIV(feat_dim, WARP_SZ * UNROLL_X), batch_num);
        SegPoolKernel<reduce_type, UNROLL_X, WARP_SZ> <<<dimGrid, dimBlock, 0, stream >>> (dst_value, dst_index, data, indices, indptr, batch_num, seg_num, feat_dim, total_ind_num, nnz);
        CUDA_POST_KERNEL_CHECK(SegPoolKernel);
    }

    template<int reduce_type, bool add_to>
    static void compute_grad_data(float* dst, const float* g_out, const int* out_index, const int* indices, const int* indptr,
                                  int batch_num, int seg_num, int feat_dim, int total_ind_num, int nnz,
                                  char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
        if(!add_to) {
            cudaMemsetAsync(dst, 0, sizeof(float) * batch_num * total_ind_num * feat_dim, stream);
        }
        int* seg_ids = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        int* sorted_indices = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        int* temp_ind_in = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        int* sorted_orig_inds = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        // 1. Sort the indices
        std::pair<dim3, dim3> block_grid_dim3 = KernelLauchParamB1G1(nnz);
        IdxArrayKernel << <block_grid_dim3.second, block_grid_dim3.first, 0, stream >> > (temp_ind_in, nnz);
        size_t temp_sort_bytes = get_sort_temp_bytes(nnz);
        cub::DeviceRadixSort::SortPairs(temp_storage, temp_sort_bytes,
          indices, sorted_indices,
          temp_ind_in, sorted_orig_inds, nnz, 0, sizeof(int) * 8, stream);
        temp_storage += get_sort_temp_bytes(nnz);
        // 2. Compute the rev mapping
        GetSegId::compute(seg_ids, indptr, seg_num, nnz, temp_storage, GetSegId::get_temp_bytes(nnz), stream);
        // 3. Run the kernel
        static const int UNROLL_X = 4;
        static const int TX_SZ = 32;
        dim3 dimBlock(TX_SZ);
        dim3 dimGrid(nnz, CEIL_DIV(feat_dim, TX_SZ * UNROLL_X), batch_num);
        SegPoolBackwardKernel<reduce_type, UNROLL_X, TX_SZ> << <dimGrid, dimBlock, 0, stream >> > (dst, g_out, out_index, sorted_indices,
          sorted_orig_inds, seg_ids, indptr, batch_num, seg_num, feat_dim, total_ind_num, nnz);
        CUDA_POST_KERNEL_CHECK(SegPoolBackwardKernel);
    }
};

void GenRandIndptr(int nnz, std::vector<int> &data) {
    int seg_num = data.size() - 1;
    std::vector<int> temp(nnz);
    for (int i = 0; i < nnz; i++) {
        temp[i] = i;
    }
    std::random_shuffle(temp.begin(), temp.end());
    std::sort(temp.begin(), temp.begin() + seg_num - 1);
    data[0] = 0;
    data[seg_num] = nnz;
    for(int i=0; i < seg_num - 1; i++) {
        data[i + 1] = temp[i];
    }
}

void GenRandVec(std::vector<float> &data) {
    std::uniform_real_distribution<float> dist(-1, 1);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dist(generator);
    }
}

void GenRandNodeID(int node_num, std::vector<int> &data) {
    std::uniform_int_distribution<> dist(0, node_num - 1);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dist(generator);
    }
}

void test_GetSegId(int nnz, int seg_num) {
    GPUTimer gpu_timer;
    std::vector<int> ind_ptr(seg_num + 1);
    std::vector<int> seg_ids(nnz);
    GenRandIndptr(nnz, ind_ptr);
    int* ind_ptr_dev = NULL;
    int* seg_ids_dev = NULL;
    cudaMalloc(&ind_ptr_dev, sizeof(int) * ind_ptr.size());
    cudaMalloc(&seg_ids_dev, sizeof(int) * seg_ids.size());
    cudaMemcpy(ind_ptr_dev, ind_ptr.data(), sizeof(int) * ind_ptr.size(), cudaMemcpyHostToDevice);
    cudaStream_t stream = NULL;
    size_t temp_storage_bytes = GetSegId::get_temp_bytes(nnz);
    char* temp_storage;
    cudaMalloc(&temp_storage, temp_storage_bytes);
    gpu_timer.start();
    GetSegId::compute(seg_ids_dev, ind_ptr_dev, seg_num, nnz, temp_storage, temp_storage_bytes, stream);
    float spent = gpu_timer.stop();
    cudaMemcpy(seg_ids.data(), seg_ids_dev, sizeof(int) * nnz, cudaMemcpyDeviceToHost);

    // Check correctness
    bool failed = false;
    std::cout << "Test GetSegId, nnz=" << nnz << ", seg_num=" << seg_num << ", Time spent=" << spent << "ms.   Check...";
    for (int i = 0; i < seg_num; i++) {
        for (int j = ind_ptr[i]; j < ind_ptr[i + 1]; j++) {
            if (seg_ids[j] != i) {
                std::cout << "Fail! " << j << " " << "seg_ids:" << seg_ids[j] << " correct:" << i << std::endl;
                failed = true;
            }
        }
    }
    if(!failed) std::cout << "Passed!" << std::endl;
    cudaFree(ind_ptr_dev);
    cudaFree(seg_ids_dev);
    cudaFree(temp_storage);

}

void test_BatchSegReduce(int batch_num, int nnz, int seg_num, std::string reduce_type) {
    GPUTimer gpu_timer;
    std::vector<int> ind_ptr(seg_num + 1);
    std::vector<float> src(batch_num * nnz);
    GenRandIndptr(nnz, ind_ptr);
    GenRandVec(src);
    std::vector<float> dst(batch_num * seg_num);
    int* ind_ptr_dev = nullptr;
    float* src_dev = nullptr;
    float* dst_dev = nullptr;
    char* temp_storage = nullptr;
    cudaMalloc(&ind_ptr_dev, sizeof(int) * ind_ptr.size());
    cudaMalloc(&src_dev, sizeof(float) * src.size());
    cudaMalloc(&dst_dev, sizeof(float) * dst.size());
    cudaMemcpy(ind_ptr_dev, ind_ptr.data(), sizeof(int) * ind_ptr.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(src_dev, src.data(), sizeof(float) * src.size(), cudaMemcpyHostToDevice);
    cudaStream_t stream = NULL;
    float time_spent = -1;
    if(reduce_type == "sum") {
        size_t temp_storage_bytes = BatchSegReduceContig<SumOpCUB, float>::get_temp_bytes(batch_num, nnz, seg_num);
        cudaMalloc(&temp_storage, temp_storage_bytes);
        gpu_timer.start();
        BatchSegReduceContig<SumOpCUB, float>::compute(dst_dev, src_dev, ind_ptr_dev, batch_num, nnz, seg_num, temp_storage, temp_storage_bytes, stream);
        time_spent = gpu_timer.stop();
    } else if (reduce_type == "min") {
        size_t temp_storage_bytes = BatchSegReduceContig<MinOpCUB, float>::get_temp_bytes(batch_num, nnz, seg_num);
        cudaMalloc(&temp_storage, temp_storage_bytes);
        gpu_timer.start();
        BatchSegReduceContig<MinOpCUB, float>::compute(dst_dev, src_dev, ind_ptr_dev, batch_num, nnz, seg_num, temp_storage, temp_storage_bytes, stream);
        time_spent = gpu_timer.stop();
    } else if (reduce_type == "max") {
        size_t temp_storage_bytes = BatchSegReduceContig<MaxOpCUB, float>::get_temp_bytes(batch_num, nnz, seg_num);
        cudaMalloc(&temp_storage, temp_storage_bytes);
        gpu_timer.start();
        BatchSegReduceContig<MaxOpCUB, float>::compute(dst_dev, src_dev, ind_ptr_dev, batch_num, nnz, seg_num, temp_storage, temp_storage_bytes, stream);
        time_spent = gpu_timer.stop();
    }
    cudaMemcpy(dst.data(), dst_dev, sizeof(float) * dst.size(), cudaMemcpyDeviceToHost);
    //std::cout << "ind_ptr=";
    //for(int i=0; i < ind_ptr.size(); i ++) {
    //    if(i == ind_ptr.size() - 1) {
    //        std::cout << ind_ptr[i] << std::endl;
    //    } else {
    //        std::cout << ind_ptr[i] << ", ";
    //    }
    //}
    // Check correctness
    std::cout << "Test BatchSegReduce, reduce_type=" << reduce_type << ", batch_num=" << batch_num << ", nnz=" << nnz << ", seg_num=" << seg_num << ", Time spent=" << time_spent << "ms    Check...";
    for(int k=0; k < batch_num; k++) {
        for (int i = 0; i < seg_num; i++) {
            float res = 0.0;
            if (reduce_type == "max") {
                res = std::numeric_limits<float>::lowest();
            } else if (reduce_type == "min") {
                res = std::numeric_limits<float>::max();
            }
            for (int j = ind_ptr[i]; j < ind_ptr[i + 1]; j++) {
                if(reduce_type == "sum") {
                    res += src[k * nnz + j];
                } else if (reduce_type == "max") {
                    res = std::max(res, src[k * nnz + j]);
                } else if (reduce_type == "min") {
                    res = std::min(res, src[k * nnz + j]);
                }
            }
            float diff = std::abs(res - dst[k * seg_num + i]);
            float max_val = std::max(std::abs(res), std::abs(dst[k * seg_num + i]));
            if(!(diff < 1E-3 || diff / max_val < 1E-5)) {
                std::cout << "Fail! " << res << " " << dst[k * seg_num + i] << " " << std::abs(res - dst[k * seg_num + i]) << std::endl;
            }
        }
    }
    std::cout << "Passed!" << std::endl;
    cudaFree(ind_ptr_dev);
    cudaFree(src_dev);
    cudaFree(dst_dev);
    cudaFree(temp_storage);
}

void test_BatchSegBroadcastBinary(int batch_num, int nnz, int seg_num, std::string binary_type) {
    GPUTimer gpu_timer;
    std::vector<int> ind_ptr(seg_num + 1);
    std::vector<float> rhs(batch_num * seg_num);
    std::vector<float> lhs(batch_num * nnz);
    std::vector<float> dst(batch_num * nnz);
    GenRandIndptr(nnz, ind_ptr);
    GenRandVec(rhs);
    if (binary_type == "plus") {
        GenRandVec(lhs);
    }
    int* ind_ptr_dev = nullptr;
    float* lhs_dev = nullptr;
    float* rhs_dev = nullptr;
    float* dst_dev = nullptr;
    char* temp_storage = nullptr;
    cudaMalloc(&ind_ptr_dev, sizeof(int) * ind_ptr.size());
    cudaMalloc(&rhs_dev, sizeof(float) * rhs.size());
    cudaMalloc(&lhs_dev, sizeof(float) * lhs.size());
    cudaMalloc(&dst_dev, sizeof(float) * dst.size());
    cudaMemcpy(ind_ptr_dev, ind_ptr.data(), sizeof(int) * ind_ptr.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(rhs_dev, rhs.data(), sizeof(float) * rhs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(lhs_dev, lhs.data(), sizeof(float) * lhs.size(), cudaMemcpyHostToDevice);
    cudaStream_t stream = NULL;
    float time_spent = -1;
    if (binary_type == "plus") {
        size_t temp_storage_bytes = BatchSegBroadcastBinaryContig::get_temp_bytes(nnz);
        cudaMalloc(&temp_storage, temp_storage_bytes);
        gpu_timer.start();
        BatchSegBroadcastBinaryContig::compute<plus, false>(
            dst_dev, lhs_dev, rhs_dev, ind_ptr_dev, batch_num, nnz, seg_num, temp_storage, temp_storage_bytes, stream);
        time_spent = gpu_timer.stop();
    } else if (binary_type == "broadcast_to") {
        size_t temp_storage_bytes = BatchSegBroadcastBinaryContig::get_temp_bytes(nnz);
        cudaMalloc(&temp_storage, temp_storage_bytes);
        gpu_timer.start();
        BatchSegBroadcastBinaryContig::compute<right, false>(
            dst_dev, dst_dev, rhs_dev, ind_ptr_dev, batch_num, nnz, seg_num, temp_storage, temp_storage_bytes, stream);
        time_spent = gpu_timer.stop();
    }
    cudaMemcpy(dst.data(), dst_dev, sizeof(float) * dst.size(), cudaMemcpyDeviceToHost);
    std::cout << "Test BatchSegBroadcastBinary, binary_type=" << binary_type << ", batch_num=" << batch_num << ", nnz=" << nnz << ", seg_num=" << seg_num << ", Time spent=" << time_spent << "ms    Check...";
    for(int k=0; k < batch_num; k++) {
        for(int i=0; i < seg_num; i++) {
            for(int j=ind_ptr[i]; j < ind_ptr[i+1]; j++) {
                float truth = 0.0f;
                if (binary_type == "plus") {
                    truth = lhs[k * nnz + j] + rhs[k * seg_num + i];
                } else if (binary_type == "broadcast_to") {
                    truth = rhs[k * seg_num + i];
                }
                float diff = std::abs(truth - dst[k * nnz + j]);
                if (diff > 1E-5) {
                    std::cout << "Fail! " << truth << " " << dst[k * nnz + j] << " " << diff << std::endl;
                }
            }
        }
    }
    std::cout << "Passed!" << std::endl;

    cudaFree(ind_ptr_dev);
    cudaFree(lhs_dev);
    cudaFree(rhs_dev);
    cudaFree(dst_dev);
    cudaFree(temp_storage);
}

template<int corr_type>
void test_SegTakeKCorr(int K, int node_num, int neighbor_node_num, int nnz, int feat_dim) {
    GPUTimer gpu_timer;
    std::vector<int> neighbor_ind_ptr(node_num + 1);
    std::vector<int> neighbor_ids(nnz);
    std::vector<float> embed1(K * node_num * feat_dim);
    std::vector<float> embed2(K * neighbor_node_num * feat_dim);
    std::vector<float> g_out(K * nnz);
    std::vector<float> dst_cpu_compute(K * nnz);
    std::vector<float> dst(K * nnz);
    std::vector<float> g_embed1_cpu_compute(K * node_num * feat_dim);
    std::vector<float> g_embed1(K * node_num * feat_dim);
    std::vector<float> g_embed2_cpu_compute(K * neighbor_node_num * feat_dim);
    std::vector<float> g_embed2(K * neighbor_node_num * feat_dim);
    GenRandVec(embed1);
    GenRandVec(embed2);
    GenRandVec(g_out);
    GenRandIndptr(nnz, neighbor_ind_ptr);
    GenRandNodeID(neighbor_node_num, neighbor_ids);

    // Compute the result by CPU
    double start = omp_get_wtime();
    if(corr_type == SegTakeKCorrType::kInnerProduct) {
        SegTakeKCorrCPU<mul>(dst_cpu_compute.data(), embed1.data(), embed2.data(), neighbor_ids.data(), neighbor_ind_ptr.data(),
                             K, node_num, neighbor_node_num, nnz, feat_dim);
    } else if (corr_type == SegTakeKCorrType::kEuclidean) {
        SegTakeKCorrCPU<diff_square>(dst_cpu_compute.data(), embed1.data(), embed2.data(), neighbor_ids.data(), neighbor_ind_ptr.data(),
                             K, node_num, neighbor_node_num, nnz, feat_dim);
    }
    double cpu_duration = omp_get_wtime() - start;
    
    // Compute the result by GPU
    int* neighbor_ind_ptr_dev = nullptr;
    int* neighbor_ids_dev = nullptr;
    float* embed1_dev = nullptr;
    float* embed2_dev = nullptr;
    float* g_out_dev = nullptr;
    float* dst_dev = nullptr;
    float* g_embed1_dev = nullptr;
    float* g_embed2_dev = nullptr;
    cudaMalloc(&neighbor_ind_ptr_dev, sizeof(int) * neighbor_ind_ptr.size());
    cudaMalloc(&neighbor_ids_dev, sizeof(int) * neighbor_ids.size());
    cudaMalloc(&embed1_dev, sizeof(float) * embed1.size());
    cudaMalloc(&embed2_dev, sizeof(float) * embed2.size());
    cudaMalloc(&g_out_dev, sizeof(float) * g_out.size());
    cudaMalloc(&dst_dev, sizeof(float) * dst.size());
    cudaMalloc(&g_embed1_dev, sizeof(float) * g_embed1.size());
    cudaMalloc(&g_embed2_dev, sizeof(float) * g_embed2.size());
    cudaMemcpy(neighbor_ind_ptr_dev, neighbor_ind_ptr.data(), sizeof(int) * neighbor_ind_ptr.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(neighbor_ids_dev, neighbor_ids.data(), sizeof(int) * neighbor_ids.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(embed1_dev, embed1.data(), sizeof(float) * embed1.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(embed2_dev, embed2.data(), sizeof(float) * embed2.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(g_out_dev, g_out.data(), sizeof(float) * g_out.size(), cudaMemcpyHostToDevice);

    float gpu_time_spent = -1;
    cudaStream_t stream = NULL;
    size_t temp_storage_bytes = SegTakeKCorr::get_temp_bytes(nnz);
    char* temp_storage = nullptr;
    cudaMalloc(&temp_storage, temp_storage_bytes);
    gpu_timer.start();
    SegTakeKCorr::compute<false>(
        dst_dev, embed1_dev, embed2_dev, neighbor_ids_dev, neighbor_ind_ptr_dev,
        K, node_num, neighbor_node_num, nnz, feat_dim, corr_type,
        temp_storage, temp_storage_bytes, stream);
    gpu_time_spent = gpu_timer.stop();

    // Check GPU correctness
    cudaMemcpy(dst.data(), dst_dev, sizeof(float) * dst.size(), cudaMemcpyDeviceToHost);
    std::cout << "Test SegTakeKCorr, type=" << corr_type << ", K=" << K << ", node_num="
              << node_num << ", neighbor_node_num=" << neighbor_node_num << ", nnz="
              << nnz << ", feat_dim=" << feat_dim << std::endl;
    std::cout << "    CPU Time Spent:" << cpu_duration * 1000 << "ms, GPU Time Spent:" << gpu_time_spent << "ms,   Check...";
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < nnz; i++) {
            float diff = std::abs(dst_cpu_compute[k * nnz + i] - dst[k * nnz + i]);
            // float max_val = std::max(std::abs(res), std::abs(dst[k * seg_num + i]));
            if (!(diff < 1E-4)) {
                std::cout << "Fail! " << "k=" << k << " i=" << i << " " << dst_cpu_compute[k * nnz + i] << " " << dst[k * nnz + i] << " " << diff << std::endl;
            }
        }
    }
    std::cout << "Passed!" << std::endl;
    if(corr_type == SegTakeKCorrType::kInnerProduct) {
        // Test Backward of Embed1
        start = omp_get_wtime();
        SegTakeKCorrCPUBackwardEmbed1(g_embed1_cpu_compute.data(), g_out.data(), embed2.data(),
                                      neighbor_ids.data(), neighbor_ind_ptr.data(),
                                      K, node_num, neighbor_node_num, nnz, feat_dim);
        double backward_embed1_cpu_duration = omp_get_wtime() - start;
        gpu_timer.start();
        SegTakeKCorr::compute_grad_embed1<false>(
            g_embed1_dev, g_out_dev, embed2_dev, neighbor_ids_dev, neighbor_ind_ptr_dev,
            K, node_num, neighbor_node_num, nnz, feat_dim, corr_type, stream);
        double backward_embed1_gpu_time_spent = gpu_timer.stop();
        cudaMemcpy(g_embed1.data(), g_embed1_dev, sizeof(float) * g_embed1.size(), cudaMemcpyDeviceToHost);
        std::cout << "Test SegTakeKCorr Backward of embed1, K=" << K << ", node_num="
              << node_num << ", neighbor_node_num=" << neighbor_node_num << ", nnz="
              << nnz << ", feat_dim=" << feat_dim << std::endl;
        std::cout << "    CPU Time Spent:" << backward_embed1_cpu_duration * 1000 << "ms, GPU Time Spent:" << backward_embed1_gpu_time_spent << "ms,   Check...";
        for (int k = 0; k < K; k++) {
            for(int i = 0; i < node_num; i++) {
                for(int c = 0; c < feat_dim; c++) {
                    int ind = IND3(k, i, c, node_num, feat_dim);
                    float diff = std::abs(g_embed1_cpu_compute[ind] - g_embed1[ind]);
                    // float max_val = std::max(std::abs(res), std::abs(dst[k * seg_num + i]));
                    if (!(diff < 1E-4)) {
                        std::cout << "Fail! " << "k=" << k << " i=" << i << " c=" << c <<" " << g_embed1_cpu_compute[ind] << " " << g_embed1[ind] << " " << diff << std::endl;
                    }
                }
            }
        }
        std::cout << "Passed!" << std::endl;

        ///*
        // Test Backward of Embed2
        start = omp_get_wtime();
        SegTakeKCorrCPUBackwardEmbed2(g_embed2_cpu_compute.data(), g_out.data(), embed1.data(),
                                      neighbor_ids.data(), neighbor_ind_ptr.data(),
                                      K, node_num, neighbor_node_num, nnz, feat_dim);
        double backward_embed2_cpu_duration = omp_get_wtime() - start;

        size_t temp_storage_bytes_backward_embed2 = SegTakeKCorr::get_temp_bytes_backward_embed2(nnz, node_num);
        char* temp_storage_backward_embed2 = nullptr;
        cudaMalloc(&temp_storage_backward_embed2, temp_storage_bytes_backward_embed2);
        gpu_timer.start();
        SegTakeKCorr::compute_grad_embed2<false>(
            g_embed2_dev, g_out_dev, embed1_dev, neighbor_ids_dev, neighbor_ind_ptr_dev,
            K, node_num, neighbor_node_num, nnz, feat_dim, corr_type,
            temp_storage_backward_embed2, temp_storage_bytes_backward_embed2, stream);
        double backward_embed2_gpu_time_spent = gpu_timer.stop();
        cudaFree(temp_storage_backward_embed2);
        cudaMemcpy(g_embed2.data(), g_embed2_dev, sizeof(float) * g_embed2.size(), cudaMemcpyDeviceToHost);
        std::cout << "Test SegTakeKCorr Backward of embed2, K=" << K << ", node_num="
              << node_num << ", neighbor_node_num=" << neighbor_node_num << ", nnz="
              << nnz << ", feat_dim=" << feat_dim << std::endl;
        std::cout << "    CPU Time Spent:" << backward_embed2_cpu_duration * 1000 << "ms, GPU Time Spent:" << backward_embed2_gpu_time_spent << "ms,   Check...";
        for (int k = 0; k < K; k++) {
            for(int i = 0; i < neighbor_node_num; i++) {
                for(int c = 0; c < feat_dim; c++) {
                    int ind = IND3(k, i, c, node_num, feat_dim);
                    float diff = std::abs(g_embed2_cpu_compute[ind] - g_embed2[ind]);
                    // float max_val = std::max(std::abs(res), std::abs(dst[k * seg_num + i]));
                    if (!(diff < 1E-4)) {
                        std::cout << "Fail! " << "k=" << k << " i=" << i << " c=" << c <<" " << g_embed2_cpu_compute[ind] << " " << g_embed2[ind] << " " << diff << std::endl;
                    }
                }
            }
        }
        std::cout << "Passed!" << std::endl;
        //*/
    }

    cudaFree(neighbor_ind_ptr_dev);
    cudaFree(neighbor_ids_dev);
    cudaFree(embed1_dev);
    cudaFree(embed2_dev);
    cudaFree(g_out_dev);
    cudaFree(temp_storage);
    cudaFree(dst_dev);
    cudaFree(g_embed1_dev);
    cudaFree(g_embed2_dev);
}

template<int reduce_type>
void test_SegPool(int batch_num, int seg_num, int total_ind_num, int nnz, int feat_dim) {
    GPUTimer gpu_timer;
    std::vector<float> dst_value(batch_num * seg_num * feat_dim);
    std::vector<int> dst_index(batch_num * seg_num * feat_dim);
    std::vector<float> dst_value_cpu(batch_num * seg_num * feat_dim);
    std::vector<int> dst_index_cpu(batch_num * seg_num * feat_dim);
    std::vector<float> g_data(batch_num * total_ind_num * feat_dim);
    std::vector<float> g_data_cpu(batch_num * total_ind_num * feat_dim);
    std::vector<float> data(batch_num * total_ind_num * feat_dim);
    std::vector<float> g_out(batch_num * seg_num * feat_dim);
    std::vector<int> indices(nnz);
    std::vector<int> indptr(seg_num + 1);
    GenRandVec(data);
    GenRandVec(g_out);
    GenRandIndptr(nnz, indptr);
    GenRandNodeID(total_ind_num, indices);
    double start = omp_get_wtime();
    SegPoolCPU<reduce_type>(dst_value_cpu.data(), dst_index_cpu.data(), data.data(), indices.data(), indptr.data(), batch_num, seg_num, feat_dim, total_ind_num, nnz);
    double cpu_duration = omp_get_wtime() - start;
    float* dst_value_dev = nullptr;
    int* dst_index_dev = nullptr;
    float* g_data_dev = nullptr;
    float* data_dev = nullptr;
    float* g_out_dev = nullptr;
    int* indices_dev = nullptr;
    int* indptr_dev = nullptr;
    cudaMalloc(&dst_value_dev, sizeof(float) * dst_value.size());
    cudaMalloc(&dst_index_dev, sizeof(int) * dst_index.size());
    cudaMalloc(&g_data_dev, sizeof(float) * g_data.size());
    cudaMalloc(&data_dev, sizeof(float) * data.size());
    cudaMalloc(&g_out_dev, sizeof(float) * g_out.size());
    cudaMalloc(&indices_dev, sizeof(int) * indices.size());
    cudaMalloc(&indptr_dev, sizeof(int) * indptr.size());
    cudaMemcpy(data_dev, data.data(), sizeof(float) * data.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(g_out_dev, g_out.data(), sizeof(float) * g_out.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(indices_dev, indices.data(), sizeof(int) * indices.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(indptr_dev, indptr.data(), sizeof(int) * indptr.size(), cudaMemcpyHostToDevice);
    cudaStream_t stream = NULL;
    gpu_timer.start();
    SegPool::compute<reduce_type>(dst_value_dev, dst_index_dev, data_dev, indices_dev, indptr_dev, batch_num, seg_num, feat_dim, total_ind_num, nnz, stream);
    double gpu_time_spent = gpu_timer.stop();
    cudaMemcpy(dst_value.data(), dst_value_dev, sizeof(float) * dst_value.size(), cudaMemcpyDeviceToHost);
    if(reduce_type == SegReduceType::kMin || reduce_type == SegReduceType::kMax) {
        cudaMemcpy(dst_index.data(), dst_index_dev, sizeof(int) * dst_index.size(), cudaMemcpyDeviceToHost);
    }
    std::string reduce_type_str;
    if(reduce_type == SegReduceType::kSum) {
        reduce_type_str = "sum";
    } else if (reduce_type == SegReduceType::kMean) {
        reduce_type_str = "mean";
    } else if (reduce_type == SegReduceType::kMax) {
        reduce_type_str = "max";
    } else if (reduce_type == SegReduceType::kMin) {
        reduce_type_str = "min";
    }
    std::cout << "Test SegPool, reduce_type=" << reduce_type_str << ", batch_num=" << batch_num << ", seg_num="
        << seg_num << ", feat_dim=" << feat_dim << ", nnz="
        << nnz << std::endl;
    std::cout << "    CPU Time Spent:" << cpu_duration * 1000 << "ms, GPU Time Spent:" << gpu_time_spent << "ms,   Check...";
    for (int k = 0; k < batch_num; k++) {
        for(int i = 0; i < seg_num; i++) {
            for(int c = 0; c < feat_dim; c++) {
                int ind = IND3(k, i, c, seg_num, feat_dim);
                float diff = std::abs(dst_value_cpu[ind] - dst_value[ind]);
                // float max_val = std::max(std::abs(res), std::abs(dst[k * seg_num + i]));
                if (!(diff < 1E-4)) {
                    std::cout << "Value Fail! " << "k=" << k << " i=" << i << " c=" << c <<" " << dst_value_cpu[ind] << " " << dst_value[ind] << " " << diff << std::endl;
                }
                if(reduce_type == SegReduceType::kMin || reduce_type == SegReduceType::kMax) {
                    if(dst_index_cpu[ind] != dst_index[ind]) {
                        std::cout << "Index Fail! " << "k=" << k << " i=" << i << " c=" << c << " " << dst_index_cpu[ind] << " " << dst_index[ind] << " " << diff << std::endl;
                    }
                }
            }
        }
    }
    std::cout << "Passed!" << std::endl;

    //Test Backward
    start = omp_get_wtime();
    SegPoolBackwardCPU<reduce_type>(g_data_cpu.data(), g_out.data(), dst_index_cpu.data(), indices.data(), indptr.data(), batch_num, seg_num, feat_dim, total_ind_num, nnz);
    cpu_duration = omp_get_wtime() - start;

    size_t temp_storage_backward_bytes = SegPool::get_temp_bytes_backward(nnz);
    char* temp_storage_backward = nullptr;
    cudaMalloc(&temp_storage_backward, temp_storage_backward_bytes);
    gpu_timer.start();
    SegPool::compute_grad_data<reduce_type, false>(g_data_dev, g_out_dev, dst_index_dev, indices_dev, indptr_dev,
                                                   batch_num, seg_num, feat_dim, total_ind_num, nnz, temp_storage_backward, temp_storage_backward_bytes, stream);
    gpu_time_spent = gpu_timer.stop();
    cudaFree(temp_storage_backward);
    cudaMemcpy(g_data.data(), g_data_dev, sizeof(float) * g_data.size(), cudaMemcpyDeviceToHost);
    std::cout << "Test SegPool Backward, reduce_type=" << reduce_type_str << ", batch_num=" << batch_num << ", seg_num="
        << seg_num << ", feat_dim=" << feat_dim << ", nnz="
        << nnz << std::endl;
    std::cout << "    CPU Time Spent:" << cpu_duration * 1000 << "ms, GPU Time Spent:" << gpu_time_spent << "ms,   Check...";
    for (int k = 0; k < batch_num; k++) {
        for(int i = 0; i < total_ind_num; i++) {
            for(int c = 0; c < feat_dim; c++) {
                int ind = IND3(k, i, c, total_ind_num, feat_dim);
                float diff = std::abs(g_data_cpu[ind] - g_data[ind]);
                // float max_val = std::max(std::abs(res), std::abs(dst[k * seg_num + i]));
                if (!(diff < 1E-4)) {
                    std::cout << "Value Fail! " << "k=" << k << " i=" << i << " c=" << c <<" " << g_data_cpu[ind] << " " << g_data[ind] << " " << diff << std::endl;
                }
            }
        }
    }
    std::cout << "Passed!" << std::endl;
    cudaFree(dst_value_dev);
    cudaFree(dst_index_dev);
    cudaFree(g_data_dev);
    cudaFree(data_dev);
    cudaFree(g_out_dev);
    cudaFree(indices_dev);
    cudaFree(indptr_dev);
}

int main() {
    test_GetSegId(10000000, 2000);
    test_BatchSegReduce(5, 100, 5, "sum");
    test_BatchSegReduce(5, 100, 5, "max");
    test_BatchSegReduce(5, 100, 5, "min");
    test_BatchSegReduce(20, 10000000, 2000, "sum");
    test_BatchSegReduce(20, 10000000, 2000, "max");
    test_BatchSegReduce(20, 10000000, 2000, "min");
    test_BatchSegBroadcastBinary(5, 100, 5, "plus");
    test_BatchSegBroadcastBinary(5, 100, 5, "broadcast_to");
    test_BatchSegBroadcastBinary(20, 10000000, 2000, "plus");
    test_BatchSegBroadcastBinary(20, 10000000, 2000, "broadcast_to");
    
    test_SegTakeKCorr<SegTakeKCorrType::kInnerProduct>(2, 100, 100, 500, 64);
    test_SegTakeKCorr<SegTakeKCorrType::kInnerProduct>(2, 100, 100, 2000, 128);
    test_SegTakeKCorr<SegTakeKCorrType::kInnerProduct>(10, 1000, 2000, 100000, 64);
    test_SegTakeKCorr<SegTakeKCorrType::kInnerProduct>(20, 2000, 4000, 200000, 128);
    test_SegTakeKCorr<SegTakeKCorrType::kInnerProduct>(20, 2000, 4000, 200000, 256);
    test_SegTakeKCorr<SegTakeKCorrType::kInnerProduct>(30, 4000, 8000, 800000, 512);

    test_SegPool<SegReduceType::kSum>(2, 100, 100, 500, 64);
    test_SegPool<SegReduceType::kSum>(2, 1000, 2000, 5000, 128);
    test_SegPool<SegReduceType::kSum>(2, 1000, 2000, 10000, 128);
    test_SegPool<SegReduceType::kSum>(2, 1000, 2000, 20000, 128);
    test_SegPool<SegReduceType::kSum>(10, 2000, 4000, 100000, 256);

    test_SegPool<SegReduceType::kSum>(4, 5000, 5000, 100000, 512);
    test_SegPool<SegReduceType::kMean>(4, 5000, 5000, 100000, 512);
    test_SegPool<SegReduceType::kMax>(4, 5000, 5000, 100000, 512);
    test_SegPool<SegReduceType::kMin>(4, 5000, 5000, 100000, 512);
}
