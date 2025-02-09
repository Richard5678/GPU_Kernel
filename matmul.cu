#include <vector>
#include <cassert>
#include <iostream>
#include <chrono>
#include <stdexcept>

using namespace std;

__global__ void matmul(float* A, float* B, float* output, int m, int n, int s) {
    int r = blockIdx.x;
    int c = threadIdx.x;

    // int r = blockIdx.y * blockDim.y + threadIdx.y;
    // int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < m && c < n) {
        // output[r * n + c] = 0;
        for (int i = 0; i < s; i++) {
            output[r * n + c] += A[r * s + i] * B[c + i * n];
        }
    }
}

vector<vector<float>> matmul(vector<vector<float>>& A, vector<vector<float>>& B) {
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
    std::chrono::duration<double> duration = end - start;
    std::cout << "Matmul took " << duration.count() << " seconds" << std::endl;

    return output;
}


int main() {
    int m = 1000, s = 500, n = 700;
    // int m = 10, s = 5, n = 7;

    // Initialize random 2D matrices A and B with correct dimensions
    vector<vector<float>> A(m, vector<float>(s));
    vector<vector<float>> B(s, vector<float>(n));

    // Fill matrices with random float values
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < s; j++) {
            A[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    for(int i = 0; i < s; i++) {
        for(int j = 0; j < n; j++) {
            B[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Flatten 2D vectors into 1D arrays
    vector<float> A_flat(m * s);
    vector<float> B_flat(s * n);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < s; j++) {
            A_flat[i * s + j] = A[i][j];
        }
    }
    for(int i = 0; i < s; i++) {
        for(int j = 0; j < n; j++) {
            B_flat[i * n + j] = B[i][j];
        }
    }
    
    vector<vector<float>> C = matmul(A, B); 

    // Flatten 2D vectors into 1D arrays
    vector<float> C_flat(m * n);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            C_flat[i * n + j] = C[i][j];
        }
    }

    // allocate memory on device
    float *d_a = nullptr, *d_b = nullptr, *d_output = nullptr;
    cudaMalloc((void **)&d_a, sizeof(float) * m * s);
    cudaMalloc(&d_b, sizeof(float) * s * n);
    cudaMalloc(&d_output, sizeof(float) * m * n);

    // cpy input matrix from host to device
    cudaMemcpy(d_a, A_flat.data(), sizeof(float) * m * s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B_flat.data(), sizeof(float) * s * n, cudaMemcpyHostToDevice);


    auto start = std::chrono::high_resolution_clock::now();

    // dim3 blockDim(32, 32);
    // dim3 gridDim(
    //     (n + blockDim.x - 1) / blockDim.x,
    //     (m + blockDim.y - 1) / blockDim.y
    // );
    // // call the cuda kernel to perform the operation
    // matmul<<<gridDim, blockDim>>>(d_a, d_b, d_output, m, n, s);

    matmul<<<m, n>>>(d_a, d_b, d_output, m, n, s);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Matmul took " << duration.count() << " seconds on cuda" << std::endl;

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