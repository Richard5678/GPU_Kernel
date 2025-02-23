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
    const int row = blockIdx.y * BM + threadIdx.y * TM;

    __shared__ float As[BM * BS];
    __shared__ float Bs[BS * BN];

    float output[TM] = {0.0f};

    for (int start_idx = 0; start_idx < s; start_idx += BS)
    {
        const int idx_in_block_a = threadIdx.x * BS + threadIdx.y;
        if ((blockIdx.y * BM + threadIdx.x) < m && (start_idx + threadIdx.y) < s)
        {
            As[idx_in_block_a] = A[(blockIdx.y * BM + threadIdx.x) * s + (start_idx + threadIdx.y)];
        }
        else
        {
            As[idx_in_block_a] = 0.0f;
        }

        const int idx_in_block_b = threadIdx.y * BN + threadIdx.x;
        if ((start_idx + threadIdx.y) < s && col < n)
        {
            Bs[idx_in_block_b] = B[(start_idx + threadIdx.y) * n + col];
        }
        else
        {
            Bs[idx_in_block_b] = 0.0f;
        }

        __syncthreads();

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

    // Add bounds checking when writing output
    for (int j = 0; j < TM; j++)
    {
        if (row + j < m && col < n) // Add bounds check here
        {
            C[(row + j) * n + col] = output[j];
        }
    }
}

// 1D block tiling
//      - blockDim(BN, BM / TM)
//      - gridDim(N / BN, M / BM)
// requires:
//      - BN * BM / TM >= min(BM * BS, BS * BN)
template <const int BM, const int BS, const int BN, const int TM>
__global__ void matmul_1D_block_tiling_optimized(float *A, float *B, float *C, int m, int n, int s)
{
    // load threads with adjacent tid to allow memory coalescing
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int threadRowA = tid / BS;
    const int threadColA = tid % BS;
    const int blockRowA = blockIdx.y * BM;

    const int threadRowB = tid / BN;
    const int threadColB = tid % BN;
    const int blockColB = blockIdx.x * BN;

    __shared__ float As[BM * BS];
    __shared__ float Bs[BS * BN];

    float tileOutput[TM] = {0.0f};

    for (int start_idx = 0; start_idx < s; start_idx += BS)
    {
        // check bounds
        if (blockRowA + threadRowA < m && start_idx + threadColA < s)
        {
            As[threadRowA * BS + threadColA] = A[(blockRowA + threadRowA) * s + (start_idx + threadColA)];
        }
        else
        {
            As[threadRowA * BS + threadColA] = 0.0f;
        }

        // check bounds
        if ((start_idx + threadRowB) < s && (blockColB + threadColB) < n)
        {
            Bs[threadRowB * BN + threadColB] = B[(start_idx + threadRowB) * n + (blockColB + threadColB)];
        }
        else
        {
            Bs[threadRowB * BN + threadColB] = 0.0f;
        }

        __syncthreads();

        // accumulate partial dot product for the current 1D vertile block tile
        for (int bsOffset = 0; bsOffset < BS; bsOffset++)
        {
            const float B_val = Bs[bsOffset * BN + threadColB];
            for (int tmOffset = 0; tmOffset < TM; tmOffset++)
            {
                // only update tile output if current row is at the start of a vertical tile
                if (threadRowA % TM == 0)
                    tileOutput[tmOffset] += As[(threadRowA + tmOffset) * BS + bsOffset] * B_val;
            }
        }

        __syncthreads();
    }

    // update output of the 1D vertical block tile if within bound
    for (int tmOffset = 0; tmOffset < TM; tmOffset++)
    {
        const int rowC = blockRowA + threadRowA + tmOffset;
        const int colC = blockColB + threadColB;
        if (threadRowA % TM == 0 && rowC < m && colC < n) // bounds check here
        {
            C[rowC * n + colC] = tileOutput[tmOffset];
        }
    }
}



// // 2D block tiling
// //      - blockDim(BN / TN, BM / TM)
// //      - gridDim(N / BN, M / BM)
// // requires:
// //      - BS <= min(BN / TN, BM / TM)
// template <const int BM, const int BS, const int BN, const int TM, const int TN>
// __global__ void matmul_2D_block_tiling(float *A, float *B, float *C, int m, int n, int s)
// {
//     const int blockRow_a = blockIdx.y * BM;
//     const int threadCol_a = threadIdx.x;
//     const int threadRow_a = threadIdx.y * TM;

//     const int blockCol_b = blockIdx.x * BN;
//     const int threadRow_b = threadIdx.y;
//     const int threadCol_b = threadIdx.x * TN;

//     __shared__ float As[BM * BS];
//     __shared__ float Bs[BS * BN];

//     float output[TM * TN] = {0.0f};

//     for (int start_idx = 0; start_idx < s; start_idx += BS)
//     {
//         // load block tiles into shared memory
//         const int blockCol_a = start_idx;
//         for (int tmOffset = 0; tmOffset < TM; tmOffset++)
//         {
//             const int row_a = blockRow_a + threadRow_a + tmOffset;
//             const int col_a = blockCol_a + threadCol_a;
//             if (threadCol_a < BS && row_a < m && col_a < s)
//             {
//                 As[(threadRow_a + tmOffset) * BS + threadCol_a] = A[row_a * s + col_a];
//             }
//             else if (threadCol_a < BS && threadRow_a + tmOffset < BM && threadCol_a < BS)
//             {
//                 As[(threadRow_a + tmOffset) * BS + threadCol_a] = 0.0f;
//             }
//         }

//         const int blockRow_b = start_idx;
//         for (int tnOffset = 0; tnOffset < TN; tnOffset++)
//         {
//             const int row_b = blockRow_b + threadRow_b;
//             const int col_b = blockCol_b + threadCol_b + tnOffset;
//             if (threadRow_b < BS && row_b < s && col_b < n)
//             {
//                 Bs[threadRow_b * BN + (threadCol_b + tnOffset)] = B[row_b * n + col_b];
//             }
//             else if (threadRow_b < BS && threadRow_b < BS && threadCol_b + tnOffset < BN)
//             {
//                 Bs[threadRow_b * BN + (threadCol_b + tnOffset)] = 0.0f;
//             }
//         }

//         __syncthreads();

// // Optimized computation loop
// #pragma unroll
//         for (int bsOffset = 0; bsOffset < BS; bsOffset++)
//         {
//             // Load B values into registers once
//             float regB[TN];

// #pragma unroll
//             for (int tn = 0; tn < TN; tn++)
//             {
//                 regB[tn] = Bs[bsOffset * BN + (threadCol_b + tn)];
//             }

// // Process one row of A at a time
// #pragma unroll
//             for (int tm = 0; tm < TM; tm++)
//             {
//                 float regA = As[(threadRow_a + tm) * BS + bsOffset];

// // Multiply with all columns of B
// #pragma unroll
//                 for (int tn = 0; tn < TN; tn++)
//                 {
//                     output[tm * TN + tn] += regA * regB[tn];
//                 }
//             }
//         }

//         __syncthreads();
//     }

//     // move tile output to global output
//     for (int tmOffset = 0; tmOffset < TM; tmOffset++)
//     {
//         for (int tnOffset = 0; tnOffset < TN; tnOffset++)
//         {
//             const int row_c = blockRow_a + threadRow_a + tmOffset;
//             const int col_c = blockCol_b + threadCol_b + tnOffset;
//             if (row_c < m && col_c < n)
//             {
//                 C[row_c * n + col_c] = output[tmOffset * TN + tnOffset];
//             }
//         }
//     }
// }

// 2D block tiling
//      - blockDim(BN / TN, BM / TM)
//      - gridDim(N / BN, M / BM)
// requires:
//      - BS <= min(BN / TN, BM / TM)
template <const int BM, const int BS, const int BN, const int TM, const int TN>
__global__ void matmul_2D_block_tiling(float *A, float *B, float *C, int m, int n, int s)
{
    const int blockRow_a = blockIdx.y * BM;
    const int threadCol_a = threadIdx.x;
    const int threadRow_a = threadIdx.y * TM;

    const int blockCol_b = blockIdx.x * BN;
    const int threadRow_b = threadIdx.y;
    const int threadCol_b = threadIdx.x * TN;

    __shared__ float As[BM * BS];
    __shared__ float Bs[BS * BN];

    float output[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int start_idx = 0; start_idx < s; start_idx += BS)
    {
        // load block tiles into shared memory
        const int blockCol_a = start_idx;
        for (int tmOffset = 0; tmOffset < TM; tmOffset++)
        {
            const int row_a = blockRow_a + threadRow_a + tmOffset;
            const int col_a = blockCol_a + threadCol_a;
            if (threadCol_a < BS && row_a < m && col_a < s)
            {
                As[(threadRow_a + tmOffset) * BS + threadCol_a] = A[row_a * s + col_a];
            }
            else if (threadCol_a < BS && threadRow_a + tmOffset < BM && threadCol_a < BS)
            {
                As[(threadRow_a + tmOffset) * BS + threadCol_a] = 0.0f;
            }
        }

        const int blockRow_b = start_idx;
        for (int tnOffset = 0; tnOffset < TN; tnOffset++)
        {
            const int row_b = blockRow_b + threadRow_b;
            const int col_b = blockCol_b + threadCol_b + tnOffset;
            if (threadRow_b < BS && row_b < s && col_b < n)
            {
                Bs[threadRow_b * BN + (threadCol_b + tnOffset)] = B[row_b * n + col_b];
            }
            else if (threadRow_b < BS && threadRow_b < BS && threadCol_b + tnOffset < BN)
            {
                Bs[threadRow_b * BN + (threadCol_b + tnOffset)] = 0.0f;
            }
        }

        __syncthreads();

        // accumulate partial dot product for 2D block tile
        for (int bsOffset = 0; bsOffset < BS; bsOffset++)
        {
            for (int tmOffset = 0; tmOffset < TM; tmOffset++)
            {
                if (threadRow_a + tmOffset < BM)
                {
                    regM[tmOffset] = As[(threadRow_a + tmOffset) * BS + bsOffset];
                }
            }

            for (int tnOffset = 0; tnOffset < TN; tnOffset++)
            {
                if (threadCol_b + tnOffset < BN)
                {
                    regN[tnOffset] = Bs[bsOffset * BN + (threadCol_b + tnOffset)];
                }
            }

            for (int tnOffset = 0; tnOffset < TN; tnOffset++)
            {
                for (int tmOffset = 0; tmOffset < TM; tmOffset++)
                {
                    output[tmOffset * TN + tnOffset] += regM[tmOffset] * regN[tnOffset];
                }
            }
        }

        __syncthreads();
    }

    // move tile output to global output
    for (int tmOffset = 0; tmOffset < TM; tmOffset++)
    {
        for (int tnOffset = 0; tnOffset < TN; tnOffset++)
        {
            const int row_c = blockRow_a + threadRow_a + tmOffset;
            const int col_c = blockCol_b + threadCol_b + tnOffset;
            if (row_c < m && col_c < n)
            {
                C[row_c * n + col_c] = output[tmOffset * TN + tnOffset];
            }
        }
    }
}


// 2D block tiling
//      - blockDim(BN / TN, BM / TM)
//      - gridDim(N / BN, M / BM)
// requires:
//      - blockDim.x * blockDim.y * TM >= max(BM * BS, BS * BN) 
//      - BS <= min(BN / TN, BM / TM)
template <const int BM, const int BS, const int BN, const int TM, const int TN>
__global__ void matmul_2D_block_tiling_optimized(float *A, float *B, float *C, int m, int n, int s)
{
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    const int blockRowA = blockIdx.y * BM;
    const int threadRowA = (tid / BS) * TM;
    const int threadColA = tid % BS;

    const int blockColB = blockIdx.x * BN;
    const int threadRowB = tid / (BN / TN);
    const int threadColB = (tid % (BN / TM)) * TN;
    
    __shared__ float As[BM * BS];
    __shared__ float Bs[BS * BN];

    float output[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int start_idx = 0; start_idx < s; start_idx += BS)
    {
        // load block tiles from A into shared memory As
        //      - each thread is responsible for 1 vertical tile of size TM
        const int blockColA = start_idx;
        for (int tmOffset = 0; tmOffset < TM; tmOffset++)
        {
            const int rowA = blockRowA + threadRowA + tmOffset;
            const int colA = blockColA + threadColA;
            if (threadRowA < BM && rowA < m && colA < s)
            {
                As[(threadRowA + tmOffset) * BS + threadColA] = A[rowA * s + colA];
            }
            else if (threadRowA + tmOffset < BM && threadColA < BS)
            {
                As[(threadRowA + tmOffset) * BS + threadColA] = 0.0f;
            }
        }

        // load block tiles from B into shared memory Bs
        //      - each thread is responsible for a 1 horizontal tile of size TN
        const int blockRowB = start_idx;
        for (int tnOffset = 0; tnOffset < TN; tnOffset++)
        {
            const int rowB = blockRowB + threadRowB;
            const int colB = blockColB + threadColB + tnOffset;
            if (threadRowB < BS && rowB < s && colB < n)
            {
                Bs[threadRowB * BN + (threadColB + tnOffset)] = B[rowB * n + colB];
            }
            else if (threadRowB < BS && threadColB + tnOffset < BN)
            {
                Bs[threadRowB * BN + (threadColB + tnOffset)] = 0.0f;
            }
        }

        __syncthreads();

        // accumulate partial dot product for 2D block tile
        for (int bsOffset = 0; bsOffset < BS; bsOffset++)
        {
            for (int tmOffset = 0; tmOffset < TM; tmOffset++)
            {
                if (threadRowA + tmOffset < BM)
                {
                    regM[tmOffset] = As[(threadRowA + tmOffset) * BS + bsOffset];
                }
            }

            for (int tnOffset = 0; tnOffset < TN; tnOffset++)
            {
                if (threadColB + tnOffset < BN)
                {
                    regN[tnOffset] = Bs[bsOffset * BN + (threadColB + tnOffset)];
                }
            }

            for (int tnOffset = 0; tnOffset < TN; tnOffset++)
            {
                for (int tmOffset = 0; tmOffset < TM; tmOffset++)
                {
                    output[tmOffset * TN + tnOffset] += regM[tmOffset] * regN[tnOffset];
                }
            }
        }

        __syncthreads();
    }

    // move tile output to global output
    for (int tmOffset = 0; tmOffset < TM; tmOffset++)
    {
        for (int tnOffset = 0; tnOffset < TN; tnOffset++)
        {
            const int rowC = blockRowA + threadRowA + tmOffset;
            const int colC = blockColB + threadColB + tnOffset;
            if (rowC < m && colC < n)
            {
                C[rowC * n + colC] = output[tmOffset * TN + tnOffset];
            }
        }
    }
}