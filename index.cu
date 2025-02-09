#include <stdio.h>

__global__ void func() {
    printf("block index: (%u, %u, %u), thread index: (%u, %u, %u)", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    dim3 gridDim(2, 2, 2);
    dim3 blockDim(2, 2, 2);

    func<<<gridDim, blockDim>>>();

    cudaDeviceSynchronize();
}