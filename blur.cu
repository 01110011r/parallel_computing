#include <iostream>

__global__ void blur(int *in, int *out, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && j > 0 && i < N-1 && j < N-1) {
        int idx = i * N + j;

        out[idx] = (
            in[idx] +
            in[idx-1] +
            in[idx+1] +
            in[idx-N] +
            in[idx+N]
        ) / 5;
    }
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(int);

    int *h_in = new int[N * N];
    int *h_out = new int[N * N]();

    for (int i = 0; i < N * N; i++) {
        h_in[i] = 100;
    }

    int *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    blur<<<blocks, threads>>>(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    std::cout << "Blur completed for " << N << "x" << N << " grid. Interior cell value: "
              << h_out[N + 1] << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}