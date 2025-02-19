#include <stdio.h>

// Implementation of naive kernel with column major storage format
__global__ void matmul_naive_column_major(float *A, float *B, float *C, int m, int n, int s)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n)
    {
        float temp = 0.0f;
        for (int i = 0; i < s; i++)
        {
            temp += A[row * s + i] * B[col + i * n];
        }
        C[row * n + col] = temp;
    }
}

// Implementation of naive kernel with row major storage format
__global__ void matmul_naive_row_major(float *A, float *B, float *C, int m, int n, int s)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n)
    {
        float temp = 0.0f;
        for (int i = 0; i < s; i++)
        {
            temp += A[row * s + i] * B[col + i * n];
        }
        C[row * n + col] = temp;
    }
}

// Shared memory block caching (smbc)
//      - blockDim(BLOCK_SIZE, BLOCK_SIZE)
//      - gridDim(n / blockDim.x, m / blockDim.y)
template <const int BLOCK_SIZE>
__global__ void matmul_smbc(float *A, float *B, float *C, int m, int n, int s)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    float temp = 0.0f;

    // start_idx: starting index of the block along the tile direction
    //      - Horizontal for A
    //      - Vertical for B
    for (int start_idx = 0; start_idx < s; start_idx += BLOCK_SIZE)
    {
        // 1D index of the current thread in the block
        const int idx_in_block = threadIdx.y * BLOCK_SIZE + threadIdx.x;

        // For matrix A:
        //      - row index is 'row'
        //      - col index is 'start_idx + threadIdx.x'
        if (row < m && (start_idx + threadIdx.x) < s)
            As[idx_in_block] = A[row * s + (start_idx + threadIdx.x)];
        else
            As[idx_in_block] = 0.0f;

        // For matrix B:
        //      - row index is 'start_idx + threadIdx.y
        //      - col index is 'col'
        if (start_idx + threadIdx.y < s && col < n)
            Bs[idx_in_block] = B[(start_idx + threadIdx.y) * n + col];
        else
            Bs[idx_in_block] = 0.0f;

        __syncthreads();

        // Accumulate partial dot product for the current block
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            temp += As[threadIdx.y * BLOCK_SIZE + i] * Bs[threadIdx.x + i * BLOCK_SIZE];
        }

        __syncthreads();
    }

    // Update output if within bound
    if (row < m && col < n)
        C[row * n + col] = temp;
}

// 1D block tiling
//      - blockDim(BN, BM / TM)
//      - gridDim(N / BN, M / BM)
// requires:
//      - BM = BN = BS * TM
template <const int BM, const int BS, const int BN, const int TM>
__global__ void matmul_1D_block_tiling(float *A, float *B, float *C, int m, int n, int s)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * BM + threadIdx.y * TM; // row of the first element of the vertical tile

    __shared__ float As[BM * BS];
    __shared__ float Bs[BS * BN];

    float output[TM] = {0.0f};

    for (int start_idx = 0; start_idx < s; start_idx += BS)
    {
        // 1D index of the thread in block As
        const int idx_in_block_a = threadIdx.x * BS + threadIdx.y;

        // For matrix A:
        //      - row index is 'blockIdx.y * BM + threadIdx.x'
        //          - 'blockIdx.y * BM' -> row index of the frist row of the block
        //          - 'threadIdx.x' -> row index of the thread in the block
        //      - col index is 'start_idx + threadIdx.y'
        //          - 'start_idx' -> col index of the first column in the block
        //          - 'threadIdx.y' -> col index of the thread in the block
        if ((blockIdx.y * BM + threadIdx.x) < m && (start_idx + threadIdx.y) < s)
        {
            As[idx_in_block_a] = A[(blockIdx.y * BM + threadIdx.x) * s + (start_idx + threadIdx.y)];
        }
        else
        {
            As[idx_in_block_a] = 0.0f;
        }

        // 1D index of the thread in block Bs
        const int idx_in_block_b = threadIdx.y * BN + threadIdx.x;

        // For matrix B:
        //      - row index is 'start_idx + threadIdx.y'
        //          - 'start_idx' -> row index of the frist row of the block
        //          - 'threadIdx.y' -> row index of the thread in the block
        //      - col index is 'col'
        if ((start_idx + threadIdx.y) < s && col < n)
        {
            Bs[idx_in_block_b] = B[(start_idx + threadIdx.y) * n + col];
        }
        else
        {
            Bs[idx_in_block_b] = 0.0f;
        }

        __syncthreads();

        // accumulate partial dot product for the current 1D vertile block tile
        for (int i = 0; i < BS; i++)
        {
            const float B_val = Bs[i * BN + threadIdx.x];
            for (int j = 0; j < TM; j++)
            {
                output[j] += As[(threadIdx.y * TM + j) * BS + i] * B_val;
            }
        }

        __syncthreads();
    }

    // update output of the 1D vertical block tile if within bound
    for (int j = 0; j < TM; j++)
    {
        if (row + j < m && col < n) // bounds check here
        {
            C[(row + j) * n + col] = output[j];
        }
    }
}