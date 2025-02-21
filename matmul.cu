#include <vector>
#include <cassert>
#include <iostream>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <cublas_v2.h>
#include <random>
#include "matmul_kernels.cuh"
#include "utils.cc"

const int ITERATIONS = 10;
const int WARMUPS = 3;

enum class KernelImpl
{
    NAIVE_ROW_MAJOR,
    NAIVE_COLUMN_MAJOR,
    SMBC,
    ONE_D_BLOCK_TILING,
    TWO_D_BLOCK_TILING,
};

template <typename Func>
std::pair<const float, const float> benchmarkKernel(
    Func kernelLaunch,
    const float m,
    const float n,
    const float s,
    const int iterations = ITERATIONS,
    const int warmupRuns = WARMUPS,
    const bool printMetrics = false)
{
    // Warmup runs
    for (int i = 0; i < warmupRuns; i++) {
        kernelLaunch();
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Warmup failed: %s\n", cudaGetErrorString(err));
        return std::make_pair(-1.0f, -1.0f);
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Timing runs
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernelLaunch();
        // Check for errors after each launch
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
            return std::make_pair(-1.0f, -1.0f);
        }
    }
    cudaEventRecord(end);
    
    err = cudaEventSynchronize(end);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to synchronize: %s\n", cudaGetErrorString(err));
        return std::make_pair(-1.0f, -1.0f);
    }

    float msElapsed;
    cudaEventElapsedTime(&msElapsed, start, end);
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    float avgMsElapsed = msElapsed / iterations;
    float gflops = calculateGFLOPS(m, n, s, avgMsElapsed);

    return std::make_pair(avgMsElapsed, gflops);
}

void runKernel(KernelImpl kernel)
{
    int m = 5120, s = 5120, n = 5120;
    // int m = 1000, s = 500, n = 700;
    // int m = 73, s = 150, n = 351;
    // int m = 130, s = 130, n = 130;
    // const int m = 10, s = 5, n = 7;
    const unsigned int SEED = 42;

    std::vector<float> A(m * s);
    std::vector<float> B(s * n);

    std::mt19937 gen(SEED);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < A.size(); i++)
    {
        A[i] = dist(gen);
    }
    for (int i = 0; i < B.size(); i++)
    {
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

    auto kernel_cublas = [&]()
    {
        // cublasHandle_t handle;
        // cublasCreate(&handle);
        // float alpha = 1.0f;
        // float beta = 0.0f;

        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, s,
                    &alpha,
                    d_b, n,
                    d_a, s,
                    &beta,
                    d_c, n);
    };

    auto [ms_elapsed_cublas, gflops_cublas] = benchmarkKernel(kernel_cublas, m, n, s);

    std::cout << "cuBLAS matmul took " << ms_elapsed_cublas << " ms on cuda averaged over "
              << ITERATIONS << " iterations" << std::endl;
    std::cout << "Performance: " << gflops_cublas << " GFLOPS" << std::endl;

    dim3 blockDim(32, 32);
    dim3 gridDimRowMajor(
        ceil_div(m, blockDim.x),
        ceil_div(n, blockDim.y));
    dim3 gridDimColumnMajor(
        ceil_div(n, blockDim.x),
        ceil_div(m, blockDim.y));

    float ms_elapsed = 0, gflops = 0;
    // // naive kernel - row major
    switch (kernel)
    {
    case KernelImpl::NAIVE_ROW_MAJOR:
    {
        auto kernel_naive_row_major = [&]()
        {
            matmul_naive_row_major<<<gridDimRowMajor, blockDim>>>(d_a, d_b, d_c, m, n, s);
        };

        auto metrics = benchmarkKernel(kernel_naive_row_major, m, n, s);
        ms_elapsed = metrics.first, gflops = metrics.second;
        break;
    }
    case KernelImpl::NAIVE_COLUMN_MAJOR:
    {
        auto kernel_naive_column_major = [&]()
        {
            matmul_naive_column_major<<<gridDimColumnMajor, blockDim>>>(d_a, d_b, d_c, m, n, s);
        };

        auto metrics = benchmarkKernel(kernel_naive_column_major, m, n, s);
        ms_elapsed = metrics.first, gflops = metrics.second;
        break;
    }
    case KernelImpl::SMBC:
    {
        auto kernel_smbc = [&]()
        {
            matmul_smbc<32><<<gridDimColumnMajor, blockDim>>>(d_a, d_b, d_c, m, n, s);
        };
        auto metrics = benchmarkKernel(kernel_smbc, m, n, s);
        ms_elapsed = metrics.first, gflops = metrics.second;
        break;
    }
    case KernelImpl::ONE_D_BLOCK_TILING:
    {
        auto kernel_1D_block_tiling = [&]()
        {
            const int BM = 64, BS = 8, BN = 64, TM = 8;
            dim3 blockDim(BN, BM / TM);
            dim3 gridDim(
                ceil_div(n, BN),
                ceil_div(m, BM));

            matmul_1D_block_tiling<BM, BS, BN, TM><<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, s);
        };
        auto metrics = benchmarkKernel(kernel_1D_block_tiling, m, n, s);
        ms_elapsed = metrics.first, gflops = metrics.second;
        break;
    }
    case KernelImpl::TWO_D_BLOCK_TILING:
    {
        auto kernel_2D_block_tiling = [&]()
        {
            // const int BM = 64, BS = 8, BN = 64, TM = 8, TN = 8;
            // const int BM = 128, BN = 128, BS = 16, TM = 8, TN = 8;
            const int BM = 256, BN = 128, BS = 8, TM = 8, TN = 8;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim(
                ceil_div(n, BN),
                ceil_div(m, BM));

            matmul_2D_block_tiling<BM, BS, BN, TM, TN><<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, s);
        };

        auto metrics = benchmarkKernel(kernel_2D_block_tiling, m, n, s);
        ms_elapsed = metrics.first, gflops = metrics.second;
        break;
    }
    default:
    {
        std::cerr << "Error: Unknown kernel implementation specified" << std::endl;
        return;
    }
    }

    std::cout << "Matmul took " << ms_elapsed << " ms on cuda averaged over "
              << ITERATIONS << " iterations" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    const float gflops_percent = gflops / gflops_cublas * 100;
    std::cout << "gflop percetage achieved: " << gflops_percent << "%" << std::endl;

    // cpy output from device to host
    std::vector<float> h_c(m * n);
    cudaMemcpy(h_c.data(), d_c, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    bool correctness = true;
    int num_mismatch_entries = 0;
    // check correctness
    for (int i = 0; i < m * n; i++)
    {
        try
        {
            if (abs(h_expected_c[i] - h_c[i]) >= 1e-3)
            {
                throw std::runtime_error("Matrix multiplication results don't match!");
            }
        }
        catch (const std::runtime_error &e)
        {
            std::cout << i << " " << h_expected_c[i] << " " << h_c[i] << std::endl;
            correctness = false;
            num_mismatch_entries += 1;
        }
    }
    if (correctness)
    {
        std::cout << "Implementation was correct!" << std::endl;
    }
    else
    {
        std::cout << "Results dont match!" << std::endl;
        std::cout << num_mismatch_entries << " mismatch entries found" << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main()
{
    // Pass enum value to function
    runKernel(KernelImpl::TWO_D_BLOCK_TILING);
    // runKernel(KernelImpl::ONE_D_BLOCK_TILING);
    return 0;
}