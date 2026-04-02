#include <iostream>
#include <cuda.h>
#include <chrono>

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 1 << 20; // ~1M
    size_t size = N * sizeof(int);

    // host memory
    int *h_a = new int[N];
    int *h_b = new int[N];
    int *h_c = new int[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    std::cout<<"############ Benchmark ##############"<<std::endl;
    // with cpu
    auto startCPU = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        h_c[i] = h_a[i] + h_b[i];
    }

    auto endCPU = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(endCPU - startCPU).count();
    std::cout<<"CPU duration: "<<duration<<std::endl;

    // device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // copy to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "GPU time: " << ms << " ms\n";

    // copy back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Result: " << h_c[0] << " ... " << h_c[N-1] << std::endl;

    // cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}