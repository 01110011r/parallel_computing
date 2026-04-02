#include <iostream>

__global__ void matAdd(int *A, int *B, int *C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        int idx = i * N + j;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(int);

    int *h_A = new int[N*N];
    int *h_B = new int[N*N];
    int *h_C = new int[N*N];

    for (int i = 0; i < N*N; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15)/16, (N + 15)/16);

    matAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Done: " << h_C[0] << " ... " << h_C[N*N-1] << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
}