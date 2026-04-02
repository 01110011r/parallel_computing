//
// Created by rss on 4/1/26.
//
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

void matmul(vector<float>& A, vector<float>& B, vector<float>& C, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

int main() {
    int sizes[] = {16,32,64,128,256,512};

    for (int N : sizes) {
        vector<float> A(N*N), B(N*N), C(N*N);

        for (int i = 0; i < N*N; i++) {
            A[i] = 1.0f;
            B[i] = 1.0f;
        }

        double start = omp_get_wtime();

        matmul(A, B, C, N);

        double end = omp_get_wtime();

        cout << "CPU N=" << N << " Time: " << (end - start) << " sec\n";
    }
}