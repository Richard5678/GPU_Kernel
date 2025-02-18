// Implementation of naive kernel with column major storage format
__global__ void matmul_naive_column_major(float *A, float *B, float *C, int m, int n, int s)
{
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

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
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

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
template <const int BLOCK_SIZE>
__global__ void matmul_smbc(float* A, float* B, float* C, int m, int n, int s) 
{
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    float temp = 0.0f;

    // start_idx: starting index of the block along the tile direction
    //      - Horizontal for A
    //      - Vertical for B
    for (int start_idx = 0; start_idx < s; start_idx += BLOCK_SIZE) 
    {
        // 1D index of the current thread in the block
        const uint idx_in_block = threadIdx.y * BLOCK_SIZE + threadIdx.x;

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
            Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[(start_idx + threadIdx.y) * n + col];
        else
            Bs[idx_in_block] = 0.0f;

        __syncthreads();

        // Accumulate partial dot product for the current tile
        for (int i = 0; i < BLOCK_SIZE; i++) 
        {
            temp += As[threadIdx.y * BLOCK_SIZE + i] * Bs[threadIdx.x + i * BLOCK_SIZE];
        }

        __syncthreads();
    }

    // Update output if within bound
    if (row < m && col < n) C[row * n + col] = temp;
}
