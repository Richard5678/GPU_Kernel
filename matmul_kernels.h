#include <cuda_runtime.h>

// naive kernel with column major storage format
__global__ void matmul_naive_column_major(float *A, float *B, float *C, int m, int n, int s);

// naive kernel row major storage format
__global__ void matmul_naive_row_major(float *A, float *B, float *C, int m, int n, int s);