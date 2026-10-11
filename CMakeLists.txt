cmake_minimum_required(VERSION 3.18)
project(parallel_computing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find OpenMP and Intel TBB
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

# ==============================================================================
# CPU Targets
# ==============================================================================
add_executable(cpu_matmul cpu_matmul.cpp)
target_link_libraries(cpu_matmul PRIVATE OpenMP::OpenMP_CXX)

add_executable(cpu_primitives cpu_primitives.cpp)
target_link_libraries(cpu_primitives PRIVATE OpenMP::OpenMP_CXX TBB::tbb)

# ==============================================================================
# GPU Targets
# ==============================================================================
add_executable(vector_add vector_add.cu)
add_executable(matrix_add matrix_add.cu)
add_executable(gpu_matmul gpu_matmul.cu)
add_executable(blur blur.cu)
