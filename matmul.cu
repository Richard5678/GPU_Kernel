#include <vector>
#include <cassert>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cublas_v2.h>

using namespace std;

__global__ void matmul(float* A, float* B, float* output, int m, int n, int s) {
    // int r = blockIdx.x;
    // int c = threadIdx.x;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < m && x < n) {
        float temp = 0;
        for (int i = 0; i < s; i++) {
            temp += A[y * s + i] * B[x + i * n];
        }
        output[y * n + x] = temp;
    }
}


int main() {
    // int m = 1000, s = 500, n = 700;
    int m = 10, s = 5, n = 7;
    const unsigned int SEED = 42;

    vector<float> A(m * s);
    vector<float> B(s * n);

    std::mt19937 gen(SEED);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);


    for(int i = 0; i < A.size(); i++) {
        A[i] = dist(gen);
    }
    for(int i = 0; i < B.size(); i++) {
        B[i] = dist(gen);
    }
    
    // allocate memory on device
    float *d_a = nullptr, *d_b = nullptr, *d_expected_c = nullptr;
    cudaMalloc((void **)&d_a, sizeof(float) * m * s);
    cudaMalloc(&d_b, sizeof(float) * s * n);
    cudaMalloc(&d_expected_c, sizeof(float) * m * n);

    // cpy input matrix from host to device
    cudaMemcpy(d_a, A.data(), sizeof(float) * m * s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B.data(), sizeof(float) * s * n, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaDeviceSynchronize();

    float alpha = 1.0f;
    float beta = 0.0f;

    // Compute expected matrix product using cuBLAS
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_N,
                n, m, s,
                &alpha,
                d_b, n,
                d_a, s,
                &beta,
                d_expected_c, n);


    auto start = std::chrono::high_resolution_clock::now();
    const int NUM_ITERATIONS = 100;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        dim3 blockDim(32, 32);
        dim3 gridDim(
            (n + blockDim.x - 1) / blockDim.x,
            (m + blockDim.y - 1) / blockDim.y
        );
        // call the cuda kernel to perform the operation
        matmul<<<gridDim, blockDim>>>(d_a, d_b, d_output, m, n, s);

        // matmul<<<m, n>>>(d_a, d_b, d_output, m, n, s);

    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Matmul took " << (duration.count() / NUM_ITERATIONS) 
        << " ms on cuda" << std::endl;

    // cpy output from device to host
    vector<float> h_output(m * n);
    cudaMemcpy(h_output.data(), d_output, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    bool correctness = true;
    // check correctness
    for (int i = 0; i < m * n; i++) {
        try {
            if (abs(C_flat[i] - h_output[i]) >= 1e-3) {
                throw runtime_error("Matrix multiplication results don't match!");
            }
        }
        catch (const runtime_error& e) {
            cout << i << " " << C_flat[i] << " " << h_output[i] << endl;
            correctness = false;
        }
    }
    if (correctness) {
        cout << "Implementation was correct!" << endl;
    } else {
        cout << "Results dont match" << endl;
    } 

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_output);
}