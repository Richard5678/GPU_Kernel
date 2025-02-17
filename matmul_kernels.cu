#include "matmul_kernels.h"

// implementation of naive kernel with column major storage format
__global__ void matmul_naive_column_major(float* A, float* B, float* C, int m, int n, int s) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < m && c < n) {
        float temp = 0.0f;
        for (int i = 0; i < s; i++) {
            temp += A[r * s + i] * B[c + i * n];
        }
        C[r * n + c] = temp;
    }
}


__global__ void matmul_naive_row_major(float* A, float* B, float* C, int m, int n, int s) {
    const unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int c = blockIdx.y * blockDim.y + threadIdx.y; 

    if (r < m && c < n) {
        float temp = 0.0f;
        for (int i = 0; i < s; i++) {
            temp += A[r * s + i] * B[c + i * n];
        }
        C[r * n + c] = temp;
    }
}