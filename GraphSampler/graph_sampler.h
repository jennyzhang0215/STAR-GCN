#ifndef GRAPH_SAMPLER_H_
#define GRAPH_SAMPLER_H_

#if !defined(_WIN32)
#define _USE_SPARSEHASH
#endif

#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <utility>
#include <omp.h>
#if defined (_USE_SPARSEHASH)
#include "sparsehash/dense_hash_map"
#include "sparsehash/dense_hash_set"

using google::dense_hash_map;
using google::dense_hash_set;
#endif

#define ASSERT(x) if(!(x)) {std::cout << "Line:" << __LINE__ << " " #x << " does not hold!" << std::endl;exit(0);}

namespace graph_sampler {
typedef std::unordered_map<int, std::unordered_set<int> > GRAPH_DATA_T;
typedef std::mt19937 RANDOM_ENGINE;
const static int MAX_RANDOM_ENGINE_NUM = 128;
const static int MAX_ALLOWED_NODE = std::numeric_limits<int>::max();
const static long long MAX_ALLOWED_EDGE = std::numeric_limits<long long>::max();

int mxgraph_set_omp_thread_num(int max_num=4);

class SimpleGraph {
public:
  SimpleGraph() {}
	SimpleGraph(bool undirected,
		        int max_node_num = std::numeric_limits<int>::max(),
		        long long max_edge_num = std::numeric_limits<long long>::max()) {
        set_undirected(undirected);
        set_max(max_node_num, max_edge_num);
	}

    void set_undirected(bool undirected) {
        undirected_ = undirected;
    }

    void set_max(int max_node_num,
                 long long max_edge_num) {
        max_node_num_ = max_node_num;
        max_edge_num_ = max_edge_num;
        if (max_node_num_ < 0) max_node_num_ = MAX_ALLOWED_NODE;
        if (max_edge_num_ < 0) max_edge_num_ = MAX_ALLOWED_EDGE;
    }

    bool undirected() const { return undirected_; }
    int node_num() const { return node_num_; }
    int edge_num() const { return edge_num_; }
    const GRAPH_DATA_T* data() const { return &data_; }

	bool is_full() {
		return edge_num_ >= max_edge_num_ || node_num_ >= max_node_num_;
	}

    bool has_node(int node) {
        return data_.find(node) != data_.end();
    }

    bool insert_new_node(int node) {
		if (is_full()) {
			return false;
		}
		GRAPH_DATA_T::iterator node_it = data_.find(node);
		if (node_it == data_.end()) {
			data_[node] = std::unordered_set<int>();
			node_num_++;
		}
		return true;
    }

    bool insert_new_edge(std::pair<int, int> edge) {
        if(is_full()) {
            return false;
        }
		// 1. Insert the start point and end point
		GRAPH_DATA_T::iterator start_node_it = data_.find(edge.first);
		GRAPH_DATA_T::iterator end_node_it = data_.find(edge.second);
		bool has_insert_start = false;
		if (start_node_it == data_.end()) {
			has_insert_start = true;
			data_[edge.first] = std::unordered_set<int>();
            node_num_++;
			start_node_it = data_.find(edge.first);
		}
		if(end_node_it == data_.end()) {
			// Deal with the special case that the graph will be full after inserting the first node
			if (has_insert_start && is_full()) {
				data_.erase(start_node_it);
                node_num_--;
				return false;
			}
			data_[edge.second] = std::unordered_set<int>();
            node_num_++;
			end_node_it = data_.find(edge.second);
		}
        if (edge.second == edge.first) return true; // Return if the edge is a self-loop
		if(start_node_it->second.find(edge.second) == start_node_it->second.end()) {
			start_node_it->second.insert(edge.second);
			edge_num_++;
		}
		if (undirected_) {
			if (end_node_it->second.find(edge.first) == end_node_it->second.end()) {
				end_node_it->second.insert(edge.first);
			}
		}
        return true;
    }

    bool insert_nodes(const std::vector<int> &ids) {
		std::vector<int> inserted_ids;
		for (int id: ids) {
			if(is_full()) {
				for(int insert_id: inserted_ids) {
					data_.erase(insert_id);
				}
				return false;
			}
            if(data_.find(id) != data_.end()) {
                continue;
            } else {
				inserted_ids.push_back(id);
                data_[id] = std::unordered_set<int>();
                node_num_++;
            }
        }
        return true;
    }

    void convert_to_csr(std::vector<int> *end_points,
		                std::vector<int> *ind_ptr,
		                std::vector<int> *node_ids,
		                const int* src_node_ids,
		                int src_node_size) {
        int shift = 0;
        std::unordered_map<int, int> node_id_map;
        int counter = 0;
        for(const auto& ele: data_) {
            node_id_map[ele.first] = counter;
            counter++;
        }
        for (const auto &ele: data_) {
            node_ids->push_back(src_node_ids[ele.first]);
            ind_ptr->push_back(shift);
            for (int node: ele.second) {
                end_points->push_back(node_id_map[node]);
				shift++;
            }
        }
        ind_ptr->push_back(shift);
    }
private:
    int max_node_num_ = MAX_ALLOWED_NODE;
    long long max_edge_num_ = MAX_ALLOWED_EDGE;
    int node_num_ = 0;
    long long edge_num_ = 0;
    GRAPH_DATA_T data_;
    bool undirected_ = true;
};

class GraphSampler {
public:
GraphSampler(int seed_id=-1) {
	set_seed(seed_id);
}

void set_seed(int seed_id) {
	std::vector<std::uint32_t> seeds(MAX_RANDOM_ENGINE_NUM);
  int u_seed_id = seed_id;
	if(seed_id < 0) {
		//Randomly set seed of the engine
		std::random_device rd;
		std::uniform_int_distribution<int> dist(0, 100000);
		u_seed_id = dist(rd);	
	}
  RANDOM_ENGINE base_engine;
  base_engine.seed(u_seed_id);
  std::unordered_map<int, int> pool;
  for(int i = 0; i < MAX_RANDOM_ENGINE_NUM; i++) {
    std::uniform_int_distribution<int> dist(i, 100000000);
    int val = dist(base_engine);
    if(pool.find(val) != pool.end()) {
      eng_[i].seed(pool[val]);
    } else {
      eng_[i].seed(val);
    }
    if(pool.find(i) != pool.end()) {
      pool[val] = pool[i];
    } else {
      pool[val] = i;
    }
  }
}

/*
Sampling the graph by randomwalk.
At every step, we will return to the original node with return_p. Otherwise, we will jump randomly to a conneted node.
See [KDD06] Sampling from Large Graphs
------------------------------------------------------------------
Params:
src_end_points: end points in the source graph
src_ind_ptr: ind ptr in the source graph
src_node_ids: node ids of the source graph
src_undirected: whether the source graph is undirected
src_node_num: number of nodes in the source graph
initial_node: initial node of the random walk, if set to negative, the initial node will be chosen randomly from the original graph
walk_length: length of the random walk
return_prob: the returning probability
max_node_num: the maximum node num allowed in the sampled subgraph
max_edge_num: the maximum edge num allowed in the sampled subgraph
------------------------------------------------------------------
Return:
subgraph: the sampled graph
*/
SimpleGraph* random_walk(const int* src_end_points,
	                       const int* src_ind_ptr,
	                       const int* src_node_ids,
                         bool src_undirected,
	                       int src_node_num,
	                       int initial_node,
	                       int walk_length=10,
	                       double return_prob=0.15,
                         int max_node_num=std::numeric_limits<int>::max(),
	                       long long max_edge_num = std::numeric_limits<long long>::max(),
	                       int eng_id=0);
/*
Draw edges from the graph by negative sampling.

*/
void uniform_neg_sampling(const int* src_end_points,
                          const int* src_ind_ptr,
                          const int* target_indices,
                          int nnz,
                          int node_num,
                          int dst_node_num,
                          float neg_sample_scale,
                          bool replace,
                          int** dst_end_points,
                          int** dst_ind_ptr,
                          int** dst_edge_label,
                          int** dst_edge_count,
                          int* dst_nnz);

/*
Begin random walk from a given index
*/
void get_random_walk_nodes(const int* src_end_points,
                           const int* src_ind_ptr,
                           int nnz,
                           int node_num,
                           int initial_node,
                           int max_node_num,
                           int walk_length,
                           std::vector<int>* dst_indices);
/*
Random select the neighbors and return a new csr_mat
*/
void random_sample_fix_neighbor(const int* src_ind_ptr,
                                const int* sel_indices,
                                int sel_node_num,
                                int neighbor_num,
                                std::vector<int>* sampled_indices,
                                std::vector<int>* dst_ind_ptr);

/*
Randomly select the neighborhoods and merge
*/
void random_sel_neighbor_and_merge(const int* src_end_points,
                                   const int* src_ind_ptr,
                                   const int* src_node_ids,
                                   const int* sel_indices,
                                   int nnz,
                                   int sel_node_num,
                                   int neighbor_num,
                                   float neighbor_frac,
                                   bool sample_all,
                                   bool replace,
                                   std::vector<int>* dst_end_points,
                                   std::vector<int>* dst_ind_ptr,
                                   std::vector<int>* merged_node_ids,
                                   std::vector<int>* indices_in_merged);

private:
	RANDOM_ENGINE eng_[MAX_RANDOM_ENGINE_NUM];
};

void slice_csr_mat(const int* src_end_points,
                   const float* src_values,
                   const int* src_ind_ptr,
                   const int* src_row_ids,
                   const int* src_col_ids,
                   int src_row_num,
                   int src_col_num,
                   int src_nnz,
                   const int* sel_row_indices,
                   const int* sel_col_indices,
                   int dst_row_num,
                   int dst_col_num,
                   int** dst_end_points,
                   float** dst_values,
                   int** dst_ind_ptr,
                   int** dst_row_ids,
                   int** dst_col_ids,
                   int* dst_nnz);

void remove_edges(const int* end_points,
                  const float* values,
                  const int* ind_ptr,
                  const int* row_indices,
                  const int* col_indices,
                  int row_num,
                  int nnz,
                  int edge_num,
                  std::vector<int> *dst_end_points,
                  std::vector<float> *dst_values,
                  std::vector<int> *dst_ind_ptr);

void remove_edges_omp(const int* end_points,
                      const float* values,
                      const int* ind_ptr,
                      const int* row_indices,
                      const int* col_indices,
                      int row_num,
                      int nnz,
                      int edge_num,
                      std::vector<int> *dst_end_points,
                      std::vector<float> *dst_values,
                      std::vector<int> *dst_ind_ptr);

template<typename DType>
void seg_mul(const DType* lhs,
             const int* ind_ptr,
             const DType* rhs,
             int seg_num,
             int nnz,
             DType** ret) {
  *ret = new DType[nnz];
  ASSERT(ind_ptr[0] == 0);
  ASSERT(ind_ptr[seg_num] == nnz);
  mxgraph_set_omp_thread_num();
#pragma omp parallel for
  for(int i = 0; i < seg_num; i++) {
    for(int j = ind_ptr[i]; j < ind_ptr[i + 1]; j++) {
      (*ret)[j] = lhs[j] * rhs[i];
    }
  }
}

template<typename DType>
void seg_add(const DType* lhs,
             const int* ind_ptr,
             const DType* rhs,
             int seg_num,
             int nnz,
             DType** ret) {
  *ret = new DType[nnz];
  ASSERT(ind_ptr[0] == 0);
  ASSERT(ind_ptr[seg_num] == nnz);
  mxgraph_set_omp_thread_num();
#pragma omp parallel for
  for(int i = 0; i < seg_num; i++) {
    for(int j = ind_ptr[i]; j < ind_ptr[i + 1]; j++) {
      (*ret)[j] = lhs[j] + rhs[i];
    }
  }
}

template<typename DType>
void seg_sum(const DType* data,
             const int* ind_ptr,
             int seg_num,
             int nnz,
             DType** ret) {
  *ret = new DType[seg_num];
  ASSERT(ind_ptr[0] == 0);
  ASSERT(ind_ptr[seg_num] == nnz);
  mxgraph_set_omp_thread_num();
#pragma omp parallel for
  for(int i = 0; i < seg_num; i++) {
    (*ret)[i] = 0;
    for(int j = ind_ptr[i]; j < ind_ptr[i + 1]; j++) {
      (*ret)[i] += data[j];
    }
  }
}

template<typename DType>
void unique_cnt_omp(const DType* data,
                    int num,
                    std::vector<DType>* unique_data,
                    std::vector<int>* cnt) {
  int thread_num = mxgraph_set_omp_thread_num();
#if defined(_USE_SPARSEHASH)
  std::vector<dense_hash_map<DType, int>> data_map_vec(thread_num);
  dense_hash_map<DType, int> merged_data_map;
  for (int i = 0; i < thread_num; i++) {
    data_map_vec[i].set_empty_key(DType(-10000));
  }
  merged_data_map.set_empty_key(DType(-10000));
#else
  std::vector<std::unordered_map<DType, int>> data_map_vec(thread_num);
  std::unordered_map<DType, int> merged_data_map;
#endif
#pragma omp parallel for
  for (int i = 0; i < num; i++) {
    int tid = omp_get_thread_num();
    if (data_map_vec[tid].find(data[i]) != data_map_vec[tid].end()) {
      data_map_vec[tid][data[i]] += 1;
    } else {
      data_map_vec[tid][data[i]] = 1;
    }
  }
  // Merge the results of different threads
  for (int i = 0; i < thread_num; i++) {
    for (auto it = data_map_vec[i].begin(); it != data_map_vec[i].end(); ++it) {
      auto fit = merged_data_map.find(it->first);
      if (fit != merged_data_map.end()) {
        fit->second += it->second;
      } else {
        merged_data_map[it->first] = it->second;
      }
    }
  }
  // Write the merged results
  for (auto it = merged_data_map.begin(); it != merged_data_map.end(); ++it) {
    unique_data->push_back(it->first);
    cnt->push_back(it->second);
  }
}

template<typename DType>
void unique_cnt(const DType* data,
                int num,
                std::vector<DType>* unique_data,
                std::vector<int>* cnt) {
  if (num > 10000) return unique_cnt_omp(data, num, unique_data, cnt);
#if defined(_USE_SPARSEHASH)
  dense_hash_map<DType, int> data_map;
  data_map.set_empty_key(DType(-1));
#else
  std::unordered_map<DType, int> data_map;
#endif
  for (int i = 0; i < num; i++) {
    if (data_map.find(data[i]) != data_map.end()) {
      data_map[data[i]] += 1;
    } else {
      data_map[data[i]] = 1;
    }
  }
  for (auto it = data_map.begin(); it != data_map.end(); ++it) {
    unique_data->push_back(it->first);
    cnt->push_back(it->second);
  }
}

template<typename DType>
void unique_inverse_omp(const DType* data,
                        int num,
                        std::vector<DType>* unique_data,
                        std::vector<int>* idx) {
  *idx = std::vector<int>(num);
  int thread_num = mxgraph_set_omp_thread_num();
#if defined(_USE_SPARSEHASH)
  dense_hash_map<DType, int> merged_data_map;
  std::vector<dense_hash_set<DType>> hash_set_vec(thread_num);
  for (int i = 0; i < thread_num; i++) {
    hash_set_vec[i].set_empty_key(-10000);
  }
  merged_data_map.set_empty_key(-10000);
#else
  std::vector<std::unordered_map<DType, int>> data_map_vec(thread_num);
  std::unordered_map<DType, int> merged_data_map;
  std::vector<std::unordered_set<DType>> hash_set_vec(thread_num);
#endif
#pragma omp parallel for
  for (int i = 0; i < num; i++) {
    int tid = omp_get_thread_num();
    hash_set_vec[tid].insert(data[i]);
  }
  // Merge the unique values
  int ptr = 0;
  for (int i = 0; i < thread_num; i++) {
    for (auto it = hash_set_vec[i].begin(); it != hash_set_vec[i].end(); ++it) {
      auto fit = merged_data_map.find(*it);
      if (fit == merged_data_map.end()) {
        unique_data->push_back(*it);
        merged_data_map[*it] = ptr;
        ptr++;
      }
    }
  }
  // Fill in the reverse idx
#pragma omp parallel for
  for (int i = 0; i < num; i++) {
    (*idx)[i] = merged_data_map[data[i]];
  }
}


template<typename DType>
void unique_inverse(const DType* data,
                    int num,
                    std::vector<DType>* unique_data,
                    std::vector<int>* idx) {
  if (num > 10000) return unique_inverse_omp(data, num, unique_data, idx);
  *idx = std::vector<int>(num);
#if defined(_USE_SPARSEHASH)
  dense_hash_map<DType, int> data_map;
  data_map.set_empty_key(DType(-10000));
#else
  std::unordered_map<DType, int> data_map;
#endif
  int ptr = 0;
  for (int i = 0; i < num; i++) {
    auto it = data_map.find(data[i]);
    if (it != data_map.end()) {
      (*idx)[i] = it->second;
    } else {
      unique_data->push_back(data[i]);
      data_map[data[i]] = ptr;
      (*idx)[i] = ptr;
      ptr++;
    }
  }
}

void multi_link_split_by_value(const float* edge_values,
                               const int* ind_ptr,
                               const float* possible_edge_values,
                               int node_num,
                               int nnz,
                               int val_num,
                               std::vector<std::vector<int>> *split_indices,
                               std::vector<std::vector<int>> *split_ind_ptrs);

void multi_link_split_by_value_omp(const float* edge_values,
                                   const int* ind_ptr,
                                   const float* possible_edge_values,
                                   int node_num,
                                   int nnz,
                                   int val_num,
                                   std::vector<std::vector<int>> *split_indices,
                                   std::vector<std::vector<int>> *split_ind_ptrs);
template<typename DType>
void take_1d_omp(const DType* data,
                 const int* sel,
                 int data_size,
                 int sel_size,
                 std::vector<DType> *ret) {
  int thread_num = mxgraph_set_omp_thread_num(4);
  *ret = std::vector<DType>(sel_size);
#pragma omp parallel for
  for (int i = 0; i < sel_size; i++) {
   (*ret)[i] = data[sel[i]];
  }
}

void gen_row_indices_by_indptr(const int* ind_ptr,
                               int num,
                               int nnz,
                               std::vector<int>* row_indices);

void get_support(const int* row_degrees,
                 const int* col_degrees,
                 const int* ind_ptr,
                 const int* end_points,
                 int num,
                 int nnz,
                 bool symm,
                 std::vector<float>* support);
} // namespace graph_sampler
#endif
