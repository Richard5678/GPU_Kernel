#include <vector>
#include <cassert>
#include <iostream>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <cublas_v2.h>
#include <random> 

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


template <typename Func>
float benchmarkKernel(Func kernelLaunch, const int iterations=100, const int warmupRuns=5, const bool printTime=true) {
    for (int i = 0; i < warmupRuns; i++) {
        kernelLaunch();
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
        kernelLaunch();
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float msElapsed = 0;
    cudaEventElapsedTime(&msElapsed, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    if (printTime) {
        std::cout << "Matmul took " << msElapsed << " ms on cuda averaged over " 
            << iterations << " iterations" << std::endl; 
    }

    return msElapsed / iterations;
}


int main() {
    // int m = 1000, s = 500, n = 700;
    const int m = 10, s = 5, n = 7;
    const unsigned int SEED = 42;

    std::vector<float> A(m * s);
    std::vector<float> B(s * n);

    std::mt19937 gen(SEED);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);


    for(int i = 0; i < A.size(); i++) {
        A[i] = dist(gen);
    }
    for(int i = 0; i < B.size(); i++) {
        B[i] = dist(gen);
    }
    
    // allocate memory on device
    float *d_a = nullptr, *d_b = nullptr, *d_expected_c = nullptr, *d_c = nullptr;
    cudaMalloc((void **)&d_a, sizeof(float) * m * s);
    cudaMalloc(&d_b, sizeof(float) * s * n);
    cudaMalloc(&d_expected_c, sizeof(float) * m * n);
    cudaMalloc(&d_c, sizeof(float) * m * n);

    // cpy input matrix from host to device
    cudaMemcpy(d_a, A.data(), sizeof(float) * m * s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B.data(), sizeof(float) * s * n, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaDeviceSynchronize();

    float alpha = 1.0f;
    float beta = 0.0f;

    // compute expected matrix product using cuBLAS
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, s,
                &alpha,
                d_b, n,
                d_a, s,
                &beta,
                d_expected_c, n);

    cudaDeviceSynchronize();

    // copy expected matrix product from device to host
    std::vector<float> h_expected_c(m * n);
    cudaMemcpy(h_expected_c.data(), d_expected_c, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    dim3 blockDim(32, 32);
    dim3 gridDim(
        (n + blockDim.x - 1) / blockDim.x,
        (m + blockDim.y - 1) / blockDim.y
    );

    // naive Kernel
    auto kernel_1 = [&]() {
        matmul<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, s);
    };

    const float msElapsed_1 = benchmarkKernel(kernel_1);




    // cpy output from device to host
    std::vector<float> h_c(m * n);
    cudaMemcpy(h_c.data(), d_c, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    bool correctness = true;
    // check correctness
    for (int i = 0; i < m * n; i++) {
        try {
            if (abs(h_expected_c[i] - h_c[i]) >= 1e-3) {
                throw std::runtime_error("Matrix multiplication results don't match!");
            }
        }
        catch (const std::runtime_error& e) {
            std::cout << i << " " << h_expected_c[i] << " " << h_c[i] << std::endl;
            correctness = false;
        }
    }
    if (correctness) {
        std::cout << "Implementation was correct!" << std::endl;
    } else {
        std::cout << "Results dont match" << std::endl;
    } 

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}