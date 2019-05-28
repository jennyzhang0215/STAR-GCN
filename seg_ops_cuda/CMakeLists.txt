cmake_minimum_required(VERSION 2.8)
project(seg_ops)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -Wall")
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)

find_package(CUDA REQUIRED)
include_directories(cub)
# Specify binary name and source file to build it from

cuda_add_executable(
    seg_ops_test
    seg_ops.cu)
