#include "Python.h"
#include "numpy/arrayobject.h"
#include "graph_sampler.h"
#include <sstream>
#include <iostream>
#include <string>
#include <utility>

#if PY_MAJOR_VERSION >= 3
static PyObject *GraphSamplerError;
static graph_sampler::GraphSampler handle;

#define CHECK_SEQUENCE(obj)                                                                                      \
{                                                                                                                \
  if (!PySequence_Check(obj)) {                                                                                  \
    PyErr_SetString(GraphSamplerError, "Need a sequence!");                                                      \
    return NULL;                                                                                                 \
  }                                                                                                              \
}

#define PY_CHECK_EQUAL(a, b)                                                                                     \
{                                                                                                                \
  if ((a) != (b)) {                                                                                              \
    std::ostringstream err_msg;                                                                                  \
    err_msg << "Line:" << __LINE__ << ", Check\"" << #a << " == " << #b << "\" failed";                             \
    PyErr_SetString(GraphSamplerError, err_msg.str().c_str());                                                   \
    return NULL;                                                                                                 \
  }                                                                                                              \
}                                                                                                                \

#define PY_CHECK_CONTIGUOUS(a)  PY_CHECK_EQUAL(PyArray_ISCONTIGUOUS((a)), true)


void alloc_npy_from_ptr(const int* arr_ptr, const size_t arr_size, PyObject** arr_obj) {
    npy_intp siz[] = { static_cast<npy_intp>(arr_size) };
    *arr_obj = PyArray_EMPTY(1, siz, NPY_INT32, 0);
    memcpy(PyArray_DATA(*arr_obj), static_cast<const void*>(arr_ptr), sizeof(int) * arr_size);
    return;
}

void alloc_npy_from_ptr(const float* arr_ptr, const size_t arr_size, PyObject** arr_obj) {
    npy_intp siz[] = { static_cast<npy_intp>(arr_size) };
    *arr_obj = PyArray_EMPTY(1, siz, NPY_FLOAT32, 0);
    memcpy(PyArray_DATA(*arr_obj), static_cast<const void*>(arr_ptr), sizeof(float) * arr_size);
    return;
}

template<typename DType>
void alloc_npy_from_vector(const std::vector<DType> &arr_vec, PyObject** arr_obj) {
    alloc_npy_from_ptr(arr_vec.data(), arr_vec.size(), arr_obj);
    return;
}

/*
Inputs:
NDArray(int) src_indptr,
NDArray(int) sel_indices,
int neighbor_num

---------------------------------
Outputs:
NDArray(int) sampled_indices,
NDArray(int) dst_ind_ptr
*/
static PyObject* random_sample_fix_neighbor(PyObject* self, PyObject* args) {
    PyArrayObject* src_ind_ptr;
    PyArrayObject* sel_indices;
    int neighbor_num;
    if (!PyArg_ParseTuple(args, "O!O!i",
                          &PyArray_Type, &src_ind_ptr,
                          &PyArray_Type, &sel_indices,
                          &neighbor_num)) return NULL;
    PY_CHECK_CONTIGUOUS(src_ind_ptr);
    PY_CHECK_CONTIGUOUS(sel_indices);
    // Check Type
    PY_CHECK_EQUAL(PyArray_TYPE(src_ind_ptr), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(sel_indices), NPY_INT32);

    int sel_node_num = PyArray_SIZE(sel_indices);
    std::vector<int> sampled_indices_vec;
    std::vector<int> dst_ind_ptr_vec;
    handle.random_sample_fix_neighbor(static_cast<int*>(PyArray_DATA(src_ind_ptr)),
                                      static_cast<int*>(PyArray_DATA(sel_indices)),
                                      sel_node_num,
                                      neighbor_num,
                                      &sampled_indices_vec,
                                      &dst_ind_ptr_vec);
    PyObject* sampled_indices = NULL;
    PyObject* dst_ind_ptr = NULL;
    alloc_npy_from_vector(sampled_indices_vec, &sampled_indices);
    alloc_npy_from_vector(dst_ind_ptr_vec, &dst_ind_ptr);
    return Py_BuildValue("(NN)", sampled_indices, dst_ind_ptr);
}

/*
Inputs:
NDArray(int) seed
---------------------------------
Outputs:
NDArray(int) ret_val
*/
static PyObject* set_seed(PyObject* self, PyObject* args) {
    int seed;
    if (!PyArg_ParseTuple(args, "i", &seed)) return NULL;
    handle.set_seed(seed);
    return Py_BuildValue("i", 1);
}


/*
Inputs:
NDArray(int) src_end_points
NDArray(int) or None src_values
NDArray(int) src_ind_ptr
NDArray(int) src_row_ids
NDArray(int) src_col_ids
NDArray(int) sel_row_indices
NDArray(int) sel_col_indices
---------------------------------
Outputs:
NDArray(int) dst_end_points
NDArray(int) dst_ind_ptr
NDArray(int) dst_row_ids
NDArray(int) dst_col_ids
int dst_row_num
int dst_col_num
int dst_nnz
*/
static PyObject* csr_submat(PyObject* self, PyObject* args) {
    PyObject* src_end_points;
    PyObject* src_values;
    PyObject* src_ind_ptr;
    PyObject* src_row_ids;
    PyObject* src_col_ids;
    PyObject* sel_row_indices;
    PyObject* sel_col_indices;
    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &src_end_points,
                          &src_values,
                          &src_ind_ptr,
                          &src_row_ids,
                          &src_col_ids,
                          &sel_row_indices,
                          &sel_col_indices)) return NULL;
    // Check Type
    PY_CHECK_EQUAL(PyArray_TYPE(src_end_points), NPY_INT32);
    if(src_values != Py_None) {
      PY_CHECK_EQUAL(PyArray_TYPE(src_values), NPY_FLOAT32);
    }
    PY_CHECK_EQUAL(PyArray_TYPE(src_ind_ptr), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(src_row_ids), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(src_col_ids), NPY_INT32);
    if(sel_row_indices != Py_None) {
      PY_CHECK_EQUAL(PyArray_TYPE(sel_row_indices), NPY_INT32);
    }
    if(sel_col_indices != Py_None) {
      PY_CHECK_EQUAL(PyArray_TYPE(sel_col_indices), NPY_INT32);
    }


    long long src_row_num = PyArray_SIZE(src_row_ids);
    long long src_col_num = PyArray_SIZE(src_col_ids);
    long long src_nnz = PyArray_SIZE(src_end_points);
    ASSERT(src_row_num <= std::numeric_limits<int>::max());
    ASSERT(src_col_num <= std::numeric_limits<int>::max());
    ASSERT(src_nnz < std::numeric_limits<int>::max());
    int dst_row_num = (sel_row_indices == Py_None) ? src_row_num : PyArray_SIZE(sel_row_indices);
    int dst_col_num = (sel_col_indices == Py_None) ? src_col_num : PyArray_SIZE(sel_col_indices);
    float* src_values_ptr = (src_values == Py_None) ? nullptr : static_cast<float*>(PyArray_DATA(src_values));
    int* sel_row_indices_ptr = (sel_row_indices == Py_None) ? nullptr : static_cast<int*>(PyArray_DATA(sel_row_indices));
    int* sel_col_indices_ptr = (sel_col_indices == Py_None) ? nullptr : static_cast<int*>(PyArray_DATA(sel_col_indices));
    int* dst_end_points_d = NULL;
    float* dst_values_d = NULL;
    int* dst_ind_ptr_d = NULL;
    int* dst_row_ids_d = NULL;
    int* dst_col_ids_d = NULL;
    int dst_nnz;
    graph_sampler::slice_csr_mat(static_cast<int*>(PyArray_DATA(src_end_points)),
                                 src_values_ptr,
                                 static_cast<int*>(PyArray_DATA(src_ind_ptr)),
                                 static_cast<int*>(PyArray_DATA(src_row_ids)),
                                 static_cast<int*>(PyArray_DATA(src_col_ids)),
                                 src_row_num,
                                 src_col_num,
                                 src_nnz,
                                 sel_row_indices_ptr,
                                 sel_col_indices_ptr,
                                 dst_row_num,
                                 dst_col_num,
                                 &dst_end_points_d,
                                 &dst_values_d,
                                 &dst_ind_ptr_d,
                                 &dst_row_ids_d,
                                 &dst_col_ids_d,
                                 &dst_nnz);
    PyObject* dst_end_points = NULL;
    PyObject* dst_values = NULL;
    PyObject* dst_ind_ptr = NULL;
    PyObject* dst_row_ids = NULL;
    PyObject* dst_col_ids = NULL;
    alloc_npy_from_ptr(dst_end_points_d, dst_nnz, &dst_end_points);
    if(dst_values_d == nullptr) {
        Py_INCREF(Py_None);
        dst_values = Py_None;
    } else {
        alloc_npy_from_ptr(dst_values_d, dst_nnz, &dst_values);
    }
    alloc_npy_from_ptr(dst_ind_ptr_d, dst_row_num + 1, &dst_ind_ptr);
    alloc_npy_from_ptr(dst_row_ids_d, dst_row_num, &dst_row_ids);
    alloc_npy_from_ptr(dst_col_ids_d, dst_col_num, &dst_col_ids);

    //Clear Allocated Variables
    delete[] dst_end_points_d;
    if (dst_values_d != nullptr) delete[] dst_values_d;
    delete[] dst_ind_ptr_d;
    delete[] dst_row_ids_d;
    delete[] dst_col_ids_d;
    return Py_BuildValue("(NNNNN)", dst_end_points, dst_values, dst_ind_ptr, dst_row_ids, dst_col_ids);
}

/*
Inputs:
NDArray(DType) lhs
NDArray(DType) ind_ptr
NDArray(DType) rhs
---------------------------------
Outputs:
NDArray(DType) ret
*/
static PyObject* seg_mul(PyObject* self, PyObject* args) {
    PyArrayObject* lhs;
    PyArrayObject* ind_ptr;
    PyArrayObject* rhs;
    PyObject* ret = NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          &PyArray_Type, &lhs,
                          &PyArray_Type, &ind_ptr,
                          &PyArray_Type, &rhs)) return NULL;
    PY_CHECK_CONTIGUOUS(lhs);
    PY_CHECK_CONTIGUOUS(ind_ptr);
    PY_CHECK_CONTIGUOUS(rhs);
    PY_CHECK_EQUAL(PyArray_TYPE(ind_ptr), NPY_INT32);
    int seg_num = PyArray_SIZE(ind_ptr) - 1;
    int nnz = PyArray_SIZE(lhs);
    if(PyArray_TYPE(lhs) == NPY_INT32) {
        PY_CHECK_EQUAL(PyArray_TYPE(rhs), NPY_INT32);
        int* ret_d;
        graph_sampler::seg_mul(static_cast<int*>(PyArray_DATA(lhs)),
                               static_cast<int*>(PyArray_DATA(ind_ptr)),
                               static_cast<int*>(PyArray_DATA(rhs)),
                               seg_num,
                               nnz,
                               &ret_d);
        alloc_npy_from_ptr(ret_d, nnz, &ret);
        delete[] ret_d;
    } else if(PyArray_TYPE(lhs) == NPY_FLOAT32) {
        PY_CHECK_EQUAL(PyArray_TYPE(rhs), NPY_FLOAT32);
        float* ret_d;
        graph_sampler::seg_mul(static_cast<float*>(PyArray_DATA(lhs)),
                               static_cast<int*>(PyArray_DATA(ind_ptr)),
                               static_cast<float*>(PyArray_DATA(rhs)),
                               seg_num,
                               nnz,
                               &ret_d);
        alloc_npy_from_ptr(ret_d, nnz, &ret);
        delete[] ret_d;
    } else {
        PyErr_SetString(GraphSamplerError, "UnImplemented!");
        return NULL;
    }
    return Py_BuildValue("N", ret);
}

/*
Inputs:
NDArray(DType) lhs
NDArray(DType) ind_ptr
NDArray(DType) rhs
---------------------------------
Outputs:
NDArray(DType) ret
*/
static PyObject* seg_add(PyObject* self, PyObject* args) {
    PyArrayObject* lhs;
    PyArrayObject* ind_ptr;
    PyArrayObject* rhs;
    PyObject* ret = NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          &PyArray_Type, &lhs,
                          &PyArray_Type, &ind_ptr,
                          &PyArray_Type, &rhs)) return NULL;
    PY_CHECK_CONTIGUOUS(lhs);
    PY_CHECK_CONTIGUOUS(ind_ptr);
    PY_CHECK_CONTIGUOUS(rhs);
    PY_CHECK_EQUAL(PyArray_TYPE(ind_ptr), NPY_INT32);
    int seg_num = PyArray_SIZE(ind_ptr) - 1;
    int nnz = PyArray_SIZE(lhs);
    if(PyArray_TYPE(lhs) == NPY_INT32) {
        PY_CHECK_EQUAL(PyArray_TYPE(rhs), NPY_INT32);
        int* ret_d;
        graph_sampler::seg_add(static_cast<int*>(PyArray_DATA(lhs)),
                               static_cast<int*>(PyArray_DATA(ind_ptr)),
                               static_cast<int*>(PyArray_DATA(rhs)),
                               seg_num,
                               nnz,
                               &ret_d);
        alloc_npy_from_ptr(ret_d, nnz, &ret);
        delete[] ret_d;
    } else if(PyArray_TYPE(lhs) == NPY_FLOAT32) {
        PY_CHECK_EQUAL(PyArray_TYPE(rhs), NPY_FLOAT32);
        float* ret_d;
        graph_sampler::seg_add(static_cast<float*>(PyArray_DATA(lhs)),
                               static_cast<int*>(PyArray_DATA(ind_ptr)),
                               static_cast<float*>(PyArray_DATA(rhs)),
                               seg_num,
                               nnz,
                               &ret_d);
        alloc_npy_from_ptr(ret_d, nnz, &ret);
        delete[] ret_d;
    } else {
        PyErr_SetString(GraphSamplerError, "UnImplemented!");
        return NULL;
    }
    return Py_BuildValue("N", ret);
}

/*
Inputs:
NDArray(DType) data
NDArray(DType) ind_ptr
---------------------------------
Outputs:
NDArray(DType) ret
*/
static PyObject* seg_sum(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyArrayObject* ind_ptr;
    PyObject* ret = NULL;
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &data,
                          &PyArray_Type, &ind_ptr)) return NULL;
    PY_CHECK_CONTIGUOUS(data);
    PY_CHECK_CONTIGUOUS(ind_ptr);
    PY_CHECK_EQUAL(PyArray_TYPE(ind_ptr), NPY_INT32);
    int seg_num = PyArray_SIZE(ind_ptr) - 1;
    int nnz = PyArray_SIZE(data);
    if(PyArray_TYPE(data) == NPY_INT32) {
        int* ret_d;
        graph_sampler::seg_sum(static_cast<int*>(PyArray_DATA(data)),
                               static_cast<int*>(PyArray_DATA(ind_ptr)),
                               seg_num,
                               nnz,
                               &ret_d);
        alloc_npy_from_ptr(ret_d, seg_num, &ret);
        delete[] ret_d;
    } else if(PyArray_TYPE(data) == NPY_FLOAT32) {
        float* ret_d;
        graph_sampler::seg_sum(static_cast<float*>(PyArray_DATA(data)),
                               static_cast<int*>(PyArray_DATA(ind_ptr)),
                               seg_num,
                               nnz,
                               &ret_d);
        alloc_npy_from_ptr(ret_d, seg_num, &ret);
        delete[] ret_d;
    } else {
        PyErr_SetString(GraphSamplerError, "UnImplemented!");
        return NULL;
    }
    return Py_BuildValue("N", ret);
}

/*
Inputs:
NDArray(int) data
---------------------------------
Outputs:
NDArray(int) unique_value
NDArray(int) counts
*/
static PyObject* unique_cnt(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyObject* unique_data = NULL;
    PyObject* cnt = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &data)) return NULL;
    PY_CHECK_CONTIGUOUS(data);
    PY_CHECK_EQUAL(PyArray_TYPE(data), NPY_INT32);
    int num = PyArray_SIZE(data);
    std::vector<int> unique_data_vec;
    std::vector<int> cnt_vec;

    graph_sampler::unique_cnt(reinterpret_cast<int*>(data->data),
                              num,
                              &unique_data_vec,
                              &cnt_vec);
    alloc_npy_from_vector(unique_data_vec, &unique_data);
    alloc_npy_from_vector(cnt_vec, &cnt);
    return Py_BuildValue("(NN)", unique_data, cnt);
}

/*
Inputs:
NDArray(int) data
---------------------------------
Outputs:
NDArray(int) unique_value
NDArray(int) reverse_idx
*/
static PyObject* unique_inverse(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyObject* unique_data = NULL;
    PyObject* idx = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &data)) return NULL;
    PY_CHECK_CONTIGUOUS(data);
    PY_CHECK_EQUAL(PyArray_TYPE(data), NPY_INT32);
    int num = PyArray_SIZE(data);
    std::vector<int> unique_data_vec;
    std::vector<int> idx_vec;

    graph_sampler::unique_inverse(reinterpret_cast<int*>(data->data),
                                  num,
                                  &unique_data_vec,
                                  &idx_vec);
    alloc_npy_from_vector(unique_data_vec, &unique_data);
    alloc_npy_from_vector(idx_vec, &idx);
    return Py_BuildValue("(NN)", unique_data, idx);
}

/*
Inputs:
NDArray(int) end_points
NDArray(float) values
NDArray(int) ind_ptr
NDArray(int) row_indices
NDArray(int) col_indices
---------------------------------
Outputs:
NDArray(int) dst_end_points
NDArray(float) dst_values
NDArray(int) dst_ind_ptr
*/
static PyObject* remove_edges_by_indices(PyObject* self, PyObject* args) {
    PyArrayObject* end_points;
    PyArrayObject* values;
    PyArrayObject* ind_ptr;
    PyArrayObject* row_indices;
    PyArrayObject* col_indices;
    PyObject* dst_end_points = NULL;
    PyObject* dst_values = NULL;
    PyObject* dst_ind_ptr = NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
                          &PyArray_Type, &end_points,
                          &PyArray_Type, &values,
                          &PyArray_Type, &ind_ptr,
                          &PyArray_Type, &row_indices,
                          &PyArray_Type, &col_indices)) return NULL;
    PY_CHECK_CONTIGUOUS(end_points);
    PY_CHECK_CONTIGUOUS(values);
    PY_CHECK_CONTIGUOUS(ind_ptr);
    PY_CHECK_CONTIGUOUS(row_indices);
    PY_CHECK_CONTIGUOUS(col_indices);
    PY_CHECK_EQUAL(PyArray_TYPE(end_points), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(values), NPY_FLOAT32);
    PY_CHECK_EQUAL(PyArray_TYPE(ind_ptr), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(row_indices), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(col_indices), NPY_INT32);
    int row_num = PyArray_SIZE(ind_ptr) - 1;
    int nnz = PyArray_SIZE(end_points);
    int edge_num = PyArray_SIZE(row_indices);
    PY_CHECK_EQUAL(edge_num, PyArray_SIZE(col_indices));
    std::vector<int> dst_end_points_vec;
    std::vector<float> dst_values_vec;
    std::vector<int> dst_ind_ptr_vec;
    graph_sampler::remove_edges(static_cast<int*>(PyArray_DATA(end_points)),
                                static_cast<float*>(PyArray_DATA(values)),
                                static_cast<int*>(PyArray_DATA(ind_ptr)),
                                static_cast<int*>(PyArray_DATA(row_indices)),
                                static_cast<int*>(PyArray_DATA(col_indices)),
                                row_num, nnz, edge_num,
                                &dst_end_points_vec,
                                &dst_values_vec,
                                &dst_ind_ptr_vec);
    alloc_npy_from_vector(dst_end_points_vec, &dst_end_points);
    alloc_npy_from_vector(dst_values_vec, &dst_values);
    alloc_npy_from_vector(dst_ind_ptr_vec, &dst_ind_ptr);
    return Py_BuildValue("(NNN)", dst_end_points, dst_values, dst_ind_ptr);
}


/*
Inputs:
NDArray(float) edge_values
NDArray(int) ind_ptr
NDArray(float) possible_edge_values
---------------------------------
List[NDArray(int)] split_indices
List[NDArray(int)] split_ind_ptrs
*/
static PyObject* multi_link_split(PyObject* self, PyObject* args) {
    PyArrayObject* edge_values;
    PyArrayObject* ind_ptr;
    PyArrayObject* possible_edge_values;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          &PyArray_Type, &edge_values,
                          &PyArray_Type, &ind_ptr,
                          &PyArray_Type, &possible_edge_values)) return NULL;
    PY_CHECK_CONTIGUOUS(edge_values);
    PY_CHECK_CONTIGUOUS(ind_ptr);
    PY_CHECK_CONTIGUOUS(possible_edge_values);
    PY_CHECK_EQUAL(PyArray_TYPE(edge_values), NPY_FLOAT32);
    PY_CHECK_EQUAL(PyArray_TYPE(ind_ptr), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(possible_edge_values), NPY_FLOAT32);
    int node_num = PyArray_SIZE(ind_ptr) - 1;
    int nnz = PyArray_SIZE(edge_values);
    int val_num = PyArray_SIZE(possible_edge_values);
    std::vector<std::vector<int>> split_indices_vec;
    std::vector<std::vector<int>> split_ind_ptrs_vec;
    graph_sampler::multi_link_split_by_value(static_cast<float*>(PyArray_DATA(edge_values)),
                                             static_cast<int*>(PyArray_DATA(ind_ptr)),
                                             static_cast<float*>(PyArray_DATA(possible_edge_values)),
                                             node_num, nnz, val_num,
                                             &split_indices_vec, &split_ind_ptrs_vec);
    PyObject* split_indices = PyList_New(val_num);
    PyObject* split_ind_ptrs = PyList_New(val_num);
    for (int i = 0; i < val_num; i++) {
      PyObject* ele_indices;
      PyObject* ele_ind_ptr;
      alloc_npy_from_vector(split_indices_vec[i], &ele_indices);
      alloc_npy_from_vector(split_ind_ptrs_vec[i], &ele_ind_ptr);
      PyList_SetItem(split_indices, i, ele_indices);
      PyList_SetItem(split_ind_ptrs, i, ele_ind_ptr);
    }
    return Py_BuildValue("(NN)", split_indices, split_ind_ptrs);
}

static PyObject* take_1d_omp(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyArrayObject* sel;
    PyObject* ret = NULL;
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &data, &PyArray_Type, &sel)) return NULL;
    PY_CHECK_CONTIGUOUS(data);
    PY_CHECK_CONTIGUOUS(sel);
    int data_num = PyArray_SIZE(data);
    int sel_num = PyArray_SIZE(sel);
    PY_CHECK_EQUAL(PyArray_TYPE(sel), NPY_INT32);
    if (PyArray_TYPE(data) == NPY_INT32) {
        std::vector<int> ret_vec;
        graph_sampler::take_1d_omp(static_cast<int*>(PyArray_DATA(data)),
                                   static_cast<int*>(PyArray_DATA(sel)),
                                   data_num, sel_num, &ret_vec);
        alloc_npy_from_vector(ret_vec, &ret);
    } else if (PyArray_TYPE(data) == NPY_FLOAT32) {
        std::vector<float> ret_vec;
        graph_sampler::take_1d_omp(static_cast<float*>(PyArray_DATA(data)),
                                   static_cast<int*>(PyArray_DATA(sel)),
                                   data_num, sel_num, &ret_vec);
        alloc_npy_from_vector(ret_vec, &ret);
    } else {
        PyErr_SetString(GraphSamplerError, "UnImplemented!");
        return NULL;
    }
    return Py_BuildValue("N", ret);
}

static PyObject* gen_row_indices_by_indptr(PyObject* self, PyObject* args) {
    PyArrayObject* ind_ptr;
    int nnz;
    PyObject* row_indices;
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &ind_ptr, &nnz)) return NULL;
    PY_CHECK_CONTIGUOUS(ind_ptr);
    PY_CHECK_EQUAL(PyArray_TYPE(ind_ptr), NPY_INT32);
    int num = PyArray_SIZE(ind_ptr) - 1;
    std::vector<int> row_indices_vec;
    graph_sampler::gen_row_indices_by_indptr(static_cast<int*>(PyArray_DATA(ind_ptr)), num, nnz,
                                             &row_indices_vec);
    alloc_npy_from_vector(row_indices_vec, &row_indices);
    return Py_BuildValue("N", row_indices);
}

static PyObject* get_support(PyObject* self, PyObject* args) {
    PyArrayObject* row_degrees;
    PyArrayObject* col_degrees;
    PyArrayObject* ind_ptr;
    PyArrayObject* end_points;
    int symm;
    PyObject* support = NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!O!i",
                          &PyArray_Type, &row_degrees,
                          &PyArray_Type, &col_degrees,
                          &PyArray_Type, &ind_ptr,
                          &PyArray_Type, &end_points, &symm)) return NULL;
    PY_CHECK_CONTIGUOUS(row_degrees);
    PY_CHECK_CONTIGUOUS(col_degrees);
    PY_CHECK_CONTIGUOUS(ind_ptr);
    PY_CHECK_CONTIGUOUS(end_points);
    PY_CHECK_EQUAL(PyArray_TYPE(row_degrees), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(col_degrees), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(ind_ptr), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(end_points), NPY_INT32);
    int num = PyArray_SIZE(ind_ptr) - 1;
    int nnz = PyArray_SIZE(end_points);
    std::vector<float> support_vec;
    graph_sampler::get_support(static_cast<int*>(PyArray_DATA(row_degrees)),
                               static_cast<int*>(PyArray_DATA(col_degrees)),
                               static_cast<int*>(PyArray_DATA(ind_ptr)),
                               static_cast<int*>(PyArray_DATA(end_points)),
                               num, nnz, symm, &support_vec);
    alloc_npy_from_vector(support_vec, &support);
    return Py_BuildValue("N", support);
}

static PyMethodDef myextension_methods[] = {
    {"random_sample_fix_neighbor", (PyCFunction)random_sample_fix_neighbor, METH_VARARGS, NULL},
    {"set_seed", (PyCFunction)set_seed, METH_VARARGS, NULL},
    {"csr_submat", (PyCFunction)csr_submat, METH_VARARGS, NULL},
    {"seg_mul", (PyCFunction)seg_mul, METH_VARARGS, NULL},
    {"seg_add", (PyCFunction)seg_add, METH_VARARGS, NULL},
    {"seg_sum", (PyCFunction)seg_sum, METH_VARARGS, NULL},
    {"unique_cnt", (PyCFunction)unique_cnt, METH_VARARGS, NULL},
    {"unique_inverse", (PyCFunction)unique_inverse, METH_VARARGS, NULL},
    {"remove_edges_by_indices", (PyCFunction)remove_edges_by_indices, METH_VARARGS, NULL},
    {"multi_link_split", (PyCFunction)multi_link_split, METH_VARARGS, NULL},
    {"take_1d_omp", (PyCFunction)take_1d_omp, METH_VARARGS, NULL},
    {"gen_row_indices_by_indptr", (PyCFunction)gen_row_indices_by_indptr, METH_VARARGS, NULL},
    {"get_support", (PyCFunction)get_support, METH_VARARGS, NULL},
    {NULL, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_graph_sampler",
        NULL,
        -1,
        myextension_methods
};


PyMODINIT_FUNC
PyInit__graph_sampler(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL)
      return NULL;
    import_array();
    GraphSamplerError = PyErr_NewException("graph_sampler.error", NULL, NULL);
    Py_INCREF(GraphSamplerError);
    PyModule_AddObject(m, "graph_sampler.error", GraphSamplerError);
    return m;
}
#endif

