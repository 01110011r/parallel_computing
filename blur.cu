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