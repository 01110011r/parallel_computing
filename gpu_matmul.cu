#include <iostream>

__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row*N + k] * B[k*N + col];
        }
        C[row*N + col] = sum;
    }
}

int main() {
    int sizes[] = {16,32,64,128,256,512};

    for (int N : sizes) {
        size_t size = N*N*sizeof(float);

        float *h_A = new float[N*N];
        float *h_B = new float[N*N];
        float *h_C = new float[N*N];

        for (int i = 0; i < N*N; i++) {
            h_A[i] = 1.0f;
            h_B[i] = 1.0f;
        }

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        dim3 threads(16,16);
        dim3 blocks((N+15)/16, (N+15)/16);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        matmul<<<blocks, threads>>>(d_A, d_B, d_C, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        std::cout << "GPU N=" << N << " Time: " << ms << " ms\n";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        delete[] h_A; delete[] h_B; delete[] h_C;
    }
}