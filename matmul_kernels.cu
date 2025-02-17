#include "matmul_kernels.h"

// implementation of naive kernel
__global__ void matmul(float* A, float* B, float* C, int m, int n, int s) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < m && x < n) {
        float temp = 0;
        for (int i = 0; i < s; i++) {
            temp += A[y * s + i] * B[x + i * n];
        }
        C[y * n + x] = temp;
    }
}