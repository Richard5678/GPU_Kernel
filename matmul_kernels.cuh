// implementation of naive kernel with column major storage format
__global__ void matmul_naive_column_major(float *A, float *B, float *C, int m, int n, int s)
{
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < m && c < n)
    {
        float temp = 0.0f;
        for (int i = 0; i < s; i++)
        {
            temp += A[r * s + i] * B[c + i * n];
        }
        C[r * n + c] = temp;
    }
}

// implementation of naive kernel with row major storage format
__global__ void matmul_naive_row_major(float *A, float *B, float *C, int m, int n, int s)
{
    const unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < m && c < n)
    {
        float temp = 0.0f;
        for (int i = 0; i < s; i++)
        {
            temp += A[r * s + i] * B[c + i * n];
        }
        C[r * n + c] = temp;
    }
}

// share memory block caching
template <const int BLOCK_SIZE>
__global__ void matmul_smbc(float *A, float *B, float *C, int m, int n, int s)
{
    // Compute the global row and column indices for the output matrix C.
    const unsigned int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    float temp = 0.0f;

    // Loop over tiles that cover the K dimension (the common dimension between A and B).
    for (int i = 0; i < s; i += BLOCK_SIZE)
    {
        // For matrix A:
        //   - row index is 'row'
        //   - column index is 'i + threadIdx.x'
        // Check boundaries: row must be less than m and (i + threadIdx.x) less than s.
        if (row < m && (i + threadIdx.x) < s)
            As[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[row * s + (i + threadIdx.x)];
        else
            As[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;

        // For matrix B:
        //   - row index is 'i + threadIdx.y'
        //   - column index is 'col'
        // Check boundaries: (i + threadIdx.y) must be less than s and col less than n.
        if ((i + threadIdx.y) < s && col < n)
            Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[(i + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together.
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            temp += As[threadIdx.y * BLOCK_SIZE + j] * Bs[j * BLOCK_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    // Write the computed value back to global memory if within bounds.
    if (row < m && col < n)
    {
        C[row * n + col] = temp;
    }
}

// // the output block that we want to compute in this threadblock
//   int M = m, N = n, K = s;
//   const uint cRow = blockIdx.x;
//   const uint cCol = blockIdx.y;

//   // allocate buffer for current block in fast shared mem
//   // shared mem is shared between all threads in a block
//   __shared__ float As[BLOCKSIZE * BLOCKSIZE];
//   __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

//   // the inner row & col that we're accessing in this thread
// //   const uint threadCol = threadIdx.x % BLOCKSIZE;
// //   const uint threadRow = threadIdx.x / BLOCKSIZE;
//   const uint threadCol = threadIdx.x;
//   const uint threadRow = threadIdx.y;

//   // advance pointers to the starting positions
//   A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
//   B += cCol * BLOCKSIZE;                        // row=0, col=cCol
//   C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

//   float tmp = 0.0;
//   for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
//     // Have each thread load one of the elements in A & B
//     // Make the threadCol (=threadIdx.x) the consecutive index
//     // to allow global memory access coalescing
//     As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
//     Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

//     // block threads in this block until cache is fully populated
//     __syncthreads();
//     A += BLOCKSIZE;
//     B += BLOCKSIZE * N;

//     // execute the dotproduct on the currently cached block
//     for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
//       tmp += As[threadRow * BLOCKSIZE + dotIdx] *
//              Bs[dotIdx * BLOCKSIZE + threadCol];
//     }
//     // need to sync again at the end, to avoid faster threads
//     // fetching the next block into the cache before slower threads are done
//     __syncthreads();
//   }
//   C[threadRow * N + threadCol] = tmp;
//     //   alpha * tmp + beta * C[threadRow * N + threadCol];
// }