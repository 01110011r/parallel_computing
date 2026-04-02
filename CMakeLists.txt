cmake_minimum_required(VERSION 4.1)
project(parallel_computing)

set(CMAKE_CXX_STANDARD 14)

add_executable(parallel_computing
        main.cpp
        cpu_matmul.cpp)
