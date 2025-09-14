#include <iostream>
#include <vector>

// A simple CUDA kernel that runs on the GPU.
__global__ void addKernel(const int *in, int *out, std::size_t n) {
    std::size_t index{ static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x) };
    if (index < n) {
        out[index] = in[index] + 1;
    }
}

// Manages the memory on the device and launches the kernel.
extern "C" void runCudaKernel(const int* input, int* output, std::size_t n) {
    int* d_in{}, int* d_out{};

    // Allocate memory on the GPU
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_in, input, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for the kernel
    int threadsPerBlock{256};
    int blocksPerGrid{(static_cast<int>(n) + threadsPerBlock - 1) / threadsPerBlock};

    // Launch the kernel
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);

    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up: free GPU memory
    cudaFree(d_in);
    cudaFree(d_out);
}
