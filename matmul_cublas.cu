#include <vector>
#include <cassert>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cublas_v2.h>
#include <random>

using namespace std;

vector<vector<float>> cpu_matmul(vector<vector<float>>& A, vector<vector<float>>& B) {
    auto start = std::chrono::high_resolution_clock::now();
    int m = A.size(), n = B[0].size(), s = B.size();
    vector<vector<float>> output(m, vector<float>(n, 0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < s; k++) {
                output[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "CPU Matmul took " << duration.count() << " ms" << std::endl;

    return output;
}

int main() {
    // Use larger matrices to better demonstrate cuBLAS performance
    int m = 1000, s = 500, n = 700;

    // Initialize random number generator with seed for reproducibility
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // Initialize matrices A and B with random values
    vector<vector<float>> A(m, vector<float>(s));
    vector<vector<float>> B(s, vector<float>(n));

    // Fill matrices with random float values
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < s; j++) {
            A[i][j] = dis(gen);
        }
    }
    for(int i = 0; i < s; i++) {
        for(int j = 0; j < n; j++) {
            B[i][j] = dis(gen);
        }
    }

    // Compute reference result on CPU
    vector<vector<float>> C = cpu_matmul(A, B);

    // Flatten matrices for GPU computation
    vector<float> A_flat(m * s);
    vector<float> B_flat(s * n);
    vector<float> C_flat(m * n);

    // Flatten A (row-major)
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < s; j++) {
            A_flat[i * s + j] = A[i][j];
        }
    }
    // Flatten B (row-major)
    for(int i = 0; i < s; i++) {
        for(int j = 0; j < n; j++) {
            B_flat[i * n + j] = B[i][j];
        }
    }
    // Flatten C for comparison
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            C_flat[i * n + j] = C[i][j];
        }
    }

    // Allocate device memory
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc((void **)&d_a, sizeof(float) * m * s);
    cudaMalloc((void **)&d_b, sizeof(float) * s * n);
    cudaMalloc((void **)&d_c, sizeof(float) * m * n);

    // Copy input matrices from host to device
    cudaMemcpy(d_a, A_flat.data(), sizeof(float) * m * s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B_flat.data(), sizeof(float) * s * n, cudaMemcpyHostToDevice);

    // Create cuBLAS handle and warm up GPU
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaDeviceSynchronize();  // Ensure GPU is ready

    // Set up and run cuBLAS SGEMM
    float alpha = 1.0f;
    float beta = 0.0f;

    // Warm-up run to avoid first-run overhead
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, s,
                &alpha,
                d_b, n,
                d_a, s,
                &beta,
                d_c, n);

    // Time multiple iterations for more accurate measurement
    const int NUM_ITERATIONS = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < NUM_ITERATIONS; i++) {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, s,
                    &alpha,
                    d_b, n,
                    d_a, s,
                    &beta,
                    d_c, n);
    }
    cudaDeviceSynchronize();  // Ensure all operations are complete

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "cuBLAS Matmul took " << (duration.count() / NUM_ITERATIONS) << " ms (averaged over " 
              << NUM_ITERATIONS << " iterations)" << std::endl;

    // Copy result back to host
    vector<float> gpu_result(m * n);
    cudaMemcpy(gpu_result.data(), d_c, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    // Verify results
    float max_error = 0.0f;
    bool correct = true;
    for (int i = 0; i < m * n; i++) {
        float error = abs(C_flat[i] - gpu_result[i]);
        max_error = max(max_error, error);
        if (error > 1e-3) {
            correct = false;
            cout << "Mismatch at " << i << ": CPU = " << C_flat[i] 
                 << ", GPU = " << gpu_result[i] << ", error = " << error << endl;
            break;
        }
    }

    if (correct) {
        cout << "Results match! Maximum error: " << max_error << endl;
    } else {
        cout << "Results don't match! Maximum error: " << max_error << endl;
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
} 