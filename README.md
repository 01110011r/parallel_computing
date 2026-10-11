# Parallel Computing: CPU vs GPU Benchmarks & Kernels

A comprehensive repository exploring parallel computing paradigms on modern heterogeneous hardware. This project demonstrates and compares CPU multi-threading (using **OpenMP** and **Intel oneTBB**) with GPU massively parallel computing (using **NVIDIA CUDA**), covering foundational parallel algorithms, thread indexing, memory architectures, modular build pipelines, and benchmark analyses.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Parallel Paradigms & Architecture](#parallel-paradigms--architecture)
  - [CPU Multithreading: OpenMP & Intel oneTBB](#cpu-multithreading-openmp--intel-onetbb)
  - [GPU Acceleration: NVIDIA CUDA (SIMT)](#gpu-acceleration-nvidia-cuda-simt)
- [Detailed Module Walkthrough](#detailed-module-walkthrough)
  - [1. Vector Addition Benchmark (`vector_add.cu`)](#1-vector-addition-benchmark-vector_addcu)
  - [2. Matrix Multiplication Comparison (`cpu_matmul.cpp` vs `gpu_matmul.cu`)](#2-matrix-multiplication-comparison-cpu_matmulcpp-vs-gpu_matmulcu)
  - [3. 2D Matrix Addition (`matrix_add.cu`)](#3-2d-matrix-addition-matrix_addcu)
  - [4. 2D 5-Point Stencil Image Blur (`blur.cu`)](#4-2d-5-point-stencil-image-blur-blurcu)
  - [5. CPU Parallel Primitives & Memory Layout (`cpu_primitives.cpp`)](#5-cpu-parallel-primitives--memory-layout-cpu_primitivescpp)
- [Benchmark Results & Performance Analysis](#benchmark-results--performance-analysis)
- [Prerequisites & System Requirements](#prerequisites--system-requirements)
- [Build and Execution Guide](#build-and-execution-guide)
  - [Option A: Unified CMake Build (Recommended)](#option-a-unified-cmake-build-recommended)
  - [Option B: Direct Compilation (NVCC & G++)](#option-b-direct-compilation-nvcc--g)
- [Key Takeaways & Optimization Insights](#key-takeaways--optimization-insights)

---

## Overview

Modern high-performance computing (HPC) relies heavily on two primary execution models:
1. **Multi-Core CPU Parallelism**: Latency-optimized cores with deep cache hierarchies and wide vector units (SIMD), orchestrated via compiler directives (**OpenMP**) or runtime work-stealing schedulers (**Intel oneTBB**).
2. **Massively Parallel GPU Computing**: Throughput-optimized architectures containing thousands of ALUs executing in lockstep (**SIMT - Single Instruction, Multiple Threads**) via **NVIDIA CUDA**.

This repository contains concrete implementations of both paradigms across classic computational kernels:
- Linear 1D vector addition ($10^6$ elements)
- 2D matrix addition ($1024 \times 1024$)
- Dense square matrix multiplication ($N \in \{16, 32, 64, 128, 256, 512\}$)
- 2D 5-point stencil convolution (image blur)
- Cache-conscious memory layout patterns for 2D grids

---

## Repository Structure

```
parallel_computing/
├── blur.cu             # CUDA 2D 5-point stencil image blur kernel & runner
├── cpu_matmul.cpp      # CPU Matrix multiplication with OpenMP parallelization
├── cpu_primitives.cpp  # CPU multi-threading examples (OpenMP and Intel oneTBB)
├── gpu_matmul.cu       # GPU Matrix multiplication with 2D thread block decomposition
├── matrix_add.cu       # GPU 2D element-wise matrix addition kernel & runner
├── vector_add.cu       # 1D Vector Addition benchmark (CPU vs GPU with CUDA events)
├── CMakeLists.txt      # Multi-target CMake build configuration
└── README.md           # Documentation and benchmark analysis
```

---

## Parallel Paradigms & Architecture

### CPU Multithreading: OpenMP & Intel oneTBB

- **OpenMP (Open Multi-Processing)**:
  - Pragmas (`#pragma omp parallel for`) allow fine-grained loop-level parallelism.
  - Employs a fork-join threading model over shared memory.
  - Automatically partitions loop iterations across available CPU hardware threads (e.g., via static or dynamic scheduling).
- **Intel oneTBB (Threading Building Blocks)**:
  - Template-based C++ library replacing raw thread management with task-based parallelism (`tbb::parallel_for`).
  - Utilizes a work-stealing task scheduler to ensure balanced core saturation without thread oversubscription.
  - Emphasizes cache-conscious contiguous buffer structures over fragmented heap allocations (e.g. flat `vector<T>` over `vector<vector<T>>`).

### GPU Acceleration: NVIDIA CUDA (SIMT)

In CUDA, execution is offloaded to the GPU across a hierarchical structure:
- **Grid $\to$ Thread Blocks $\to$ Threads (organized in 32-thread Warps)**.
- Each thread executes the kernel with unique built-in spatial coordinates:
  - **1D Thread Indexing**:
    $$\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
  - **2D Thread Indexing**:
    $$\text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}$$
    $$\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
    $$\text{linear\_index} = \text{row} \times N + \text{col}$$
- **Memory Pipeline**:
  $$\text{Host (CPU) RAM} \xrightarrow{\text{cudaMemcpy(HtoD)}} \text{Device (GPU) Global Memory} \xrightarrow{\text{Kernel Launch}} \text{Device Memory} \xrightarrow{\text{cudaMemcpy(DtoH)}} \text{Host RAM}$$

---

## Detailed Module Walkthrough

### 1. Vector Addition Benchmark (`vector_add.cu`)

Computes $C[i] = A[i] + B[i]$ over $N = 2^{20} \approx 1{,}048{,}576$ elements.

- **CPU Implementation**: Measures sequential processing time using `std::chrono::high_resolution_clock`.
- **GPU Implementation**:
  - Launches kernel `vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N)`.
  - Block size: 256 threads (`threads = 256`).
  - Grid size: $\lceil N / \text{threads} \rceil = (N + 256 - 1) / 256$.
  - Precision timing measured via hardware GPU events (`cudaEventRecord` / `cudaEventElapsedTime`).

```cuda
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

---

### 2. Matrix Multiplication Comparison (`cpu_matmul.cpp` vs `gpu_matmul.cu`)

Computes dense square matrix multiplication $C = A \times B$ where $A, B, C \in \mathbb{R}^{N \times N}$:
$$C_{i, j} = \sum_{k=0}^{N-1} A_{i, k} \cdot B_{k, j}$$
Computational complexity is $\mathcal{O}(N^3)$, making this an ideal arithmetic-intensity test across matrix sizes $N \in \{16, 32, 64, 128, 256, 512\}$.

#### CPU: `cpu_matmul.cpp`
- Employs OpenMP directive `#pragma omp parallel for` on the outermost row loop $i$.
- Distributes rows of $C$ across threads while sharing reads from $A$ and $B$.
- Wall-clock runtime measured with `omp_get_wtime()`.

#### GPU: `gpu_matmul.cu`
- Maps each output element $(row, col)$ to a distinct CUDA thread.
- Thread blocks configured in 2D tiles of $16 \times 16 = 256$ threads:
  ```cuda
  dim3 threads(16, 16);
  dim3 blocks((N + 15) / 16, (N + 15) / 16);
  ```
- Evaluates raw compute time on device via CUDA events.

```cuda
__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

---

### 3. 2D Matrix Addition (`matrix_add.cu`)

Computes element-wise matrix addition $C = A + B$ for a $1024 \times 1024$ matrix ($1{,}048{,}576$ elements).

- Illustrates 2D thread grid decomposition:
  - Block dimensions: `dim3 threads(16, 16)`
  - Grid dimensions: `dim3 blocks((1024 + 15) / 16, (1024 + 15) / 16) = (64, 64)` blocks (4,096 total thread blocks).
- Demonstrates row-major mapping from 2D coordinates `(i, j)` to flattened 1D array index `idx = i * N + j`.

```cuda
__global__ void matAdd(int *A, int *B, int *C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        int idx = i * N + j;
        C[idx] = A[idx] + B[idx];
    }
}
```

---

### 4. 2D 5-Point Stencil Image Blur (`blur.cu`)

A 2D spatial convolution kernel implementing a 5-point cross-stencil filter (Von Neumann neighborhood averaging) commonly used in image smoothing and finite-difference PDE solvers:

$$\text{out}[i, j] = \frac{\text{in}[i, j] + \text{in}[i, j-1] + \text{in}[i, j+1] + \text{in}[i-1, j] + \text{in}[i+1, j]}{5}$$

- Explicit boundary protection (`i > 0 && j > 0 && i < N-1 && j < N-1`) avoids out-of-bounds reads on border pixels.
- Demonstrates spatial neighbor indexing in flattened row-major buffers (`idx - 1`, `idx + 1`, `idx - N`, `idx + N`).
- Includes a standalone `main()` test harness executing over a $1024 \times 1024$ grid.

```cuda
__global__ void blur(int *in, int *out, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && j > 0 && i < N - 1 && j < N - 1) {
        int idx = i * N + j;
        out[idx] = (
            in[idx] +
            in[idx - 1] +
            in[idx + 1] +
            in[idx - N] +
            in[idx + N]
        ) / 5;
    }
}
```

---

### 5. CPU Parallel Primitives & Memory Layout (`cpu_primitives.cpp`)

Showcases host-side multi-core programming techniques:

1. **OpenMP Thread Inspection & Benchmarking**:
   - Querying runtime environment: `omp_get_thread_num()`, `omp_get_num_threads()`.
   - Setting thread counts explicitly (`omp_set_num_threads`).
   - Comparing serial vs parallel loop throughput.
2. **Intel oneTBB Parallelism**:
   - `tbb::parallel_for` lambda interface for automatic load balancing across CPU worker threads.
3. **Contiguous Memory Architecture for Multi-Dimensional Data**:
   - Highlights the performance cost of fragmented memory allocations like `vector<vector<T>>` (pointer indirection, cache misses, heap overhead).
   - Replaces fragmented arrays with a single contiguous allocation `vector<T>(rows * cols)` and 2D index arithmetic `mat[i * cols + j]`, yielding superior cache locality and vectorization potential.

---

## Benchmark Results & Performance Analysis

Measured on an **NVIDIA GeForce GTX 1650** paired with a modern multi-core host CPU:

### Matrix Multiplication ($N \times N$)

| Matrix Size ($N \times N$) | Total Multiplications ($N^3$) | CPU (OpenMP Multithreaded) | GPU (CUDA Kernel) | Speedup (GPU vs CPU) |
|:---------------------------|:------------------------------|:---------------------------|:------------------|:---------------------|
| **$16 \times 16$**         | 4,096                         | 2.47 ms                    | 0.115 ms          | **~21.5×**           |
| **$32 \times 32$**         | 32,768                        | 4.27 ms                    | 0.007 ms          | **~600×**            |
| **$64 \times 64$**         | 262,144                       | 4.98 ms                    | 0.008 ms          | **~620×**            |
| **$128 \times 128$**       | 2,097,152                     | 6.06 ms                    | 0.030 ms          | **~200×**            |
| **$256 \times 256$**       | 16,777,216                    | 10.50 ms                   | 0.194 ms          | **~54×**             |
| **$512 \times 512$**       | 134,217,728                   | 29.40 ms                   | 1.478 ms          | **~20×**             |

### Vector Addition ($N = 1{,}048{,}576$ elements)

| Execution Target | Execution Time | Notes |
|:-----------------|:---------------|:------|
| **CPU Serial**   | ~3.25 ms       | Sequential loop traversal |
| **CUDA Kernel**  | ~0.08 ms       | Excludes PCIe host-device transfer |

### Analysis

1. **Kernel Launch & Warmup Overhead**: At very small dimensions ($N = 16$), initial CUDA runtime invocation overhead is visible. Once initialized, the GPU sustains orders of magnitude higher throughput.
2. **Compute Intensity vs Memory Bandwidth**: Vector addition is memory-bandwidth bound, whereas matrix multiplication is compute bound ($\mathcal{O}(N^3)$ ops vs $\mathcal{O}(N^2)$ data). The GPU's massive core count yields substantial speedup on compute-heavy operations.
3. **Data Transfer (PCIe)**: Real-world GPU speedup must account for `cudaMemcpy` transfer costs. For short-lived, low-arithmetic-intensity tasks, transfer costs can negate GPU gains; for large datasets or multi-step pipelines, GPU computation overwhelmingly dominates.

---

## Prerequisites & System Requirements

To build and run all targets in this repository, ensure you have:

- **Operating System**: Linux (Ubuntu 20.04/22.04/24.04 recommended)
- **C++ Compiler**: GCC / G++ (v9.0+ supporting C++14/17)
- **CUDA Toolkit & Compiler**: NVIDIA CUDA Toolkit (v11.x, v12.x, or v13.x) with `nvcc`
- **NVIDIA GPU**: Compute capability 5.0+ with proprietary NVIDIA drivers installed
- **Libraries**:
  - OpenMP (`libomp-dev`)
  - Intel oneTBB (`libtbb-dev`)
  - CMake (3.18+)

### Installing Dependencies on Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y build-essential cmake libomp-dev libtbb-dev
```

Verify your NVIDIA GPU and CUDA driver installation:

```bash
nvidia-smi
nvcc --version
```

---

## Build and Execution Guide

### Option A: Unified CMake Build (Recommended)

Each experiment is configured as an independent CMake target, eliminating multiple `main()` symbol collisions.

```bash
# 1. Configure and generate build files
cmake -B build

# 2. Build all targets
cmake --build build

# Or build a specific target individually:
# cmake --build build --target vector_add
# cmake --build build --target cpu_matmul

# 3. Run any experiment from the build folder
./build/vector_add
./build/matrix_add
./build/gpu_matmul
./build/blur
./build/cpu_matmul
./build/cpu_primitives
```

---

### Option B: Direct Compilation (NVCC & G++)

You can also compile each file individually using the native compiler CLI:

#### Compiling CUDA Programs (NVCC)

```bash
# Vector Addition benchmark
nvcc -O3 vector_add.cu -o vector_add && ./vector_add

# 2D Matrix Addition
nvcc -O3 matrix_add.cu -o matrix_add && ./matrix_add

# GPU Matrix Multiplication benchmark
nvcc -O3 gpu_matmul.cu -o gpu_matmul && ./gpu_matmul

# 2D Stencil Image Blur
nvcc -O3 blur.cu -o blur && ./blur
```

#### Compiling CPU Programs (G++)

```bash
# OpenMP Matrix Multiplication benchmark
g++ -O3 -fopenmp cpu_matmul.cpp -o cpu_matmul && ./cpu_matmul

# OpenMP & Intel oneTBB primitives
g++ -O3 -fopenmp cpu_primitives.cpp -ltbb -o cpu_primitives && ./cpu_primitives
```

---

## Key Takeaways & Optimization Insights

- **Thread Divergence**: In conditional kernels like `blur.cu` and boundary checks (`if (row < N && col < N)`), threads within the same warp that take different branch paths serialize execution. Designing grids that align with warp size multiples ($32$) minimizes divergence.
- **Memory Coalescing**: Consecutive threads in a warp should access contiguous memory addresses in global memory. The 2D indexing pattern `row * N + col` maps consecutive `col` threads to consecutive memory addresses, ensuring coalesced DRAM bursts.
- **Shared Memory Tiling**: The naive matrix multiplication kernel accesses global memory on every inner-loop iteration. Further optimization can be achieved by loading tiles into `__shared__` memory, drastically reducing global memory bandwidth pressure.
- **Contiguous Layouts on CPU**: Vector-of-vectors (`vector<vector<T>>`) introduces heap pointer fragmentation and cache misses. Linear 1D vectors with `(i * N + j)` indexing allow CPU prefetchers and SIMD vectorizers to operate at peak efficiency.
