#include <cuda_runtime.h>

__global__ void add_kernel(const float* input1, const float* input2, float* output, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n) {
    // 1D + 1D
    output[i] = input1[i] + input2[i];
  }
}

// For any operation with two input tensors and one output tensor
extern "C" void runKernel(const float* input1, const float* input2, float* output, std::size_t n1, std::size_t n2, std::size_t n3) {
  float *d_in1, *d_in2, *d_out;
  std::size_t size1{n1 * sizeof(float)};
  std::size_t size2{n2 * sizeof(float)};
  std::size_t size3{n3 * sizeof(float)};

  // Allocate device memory
  cudaMalloc(&d_in1, size1);
  cudaMalloc(&d_in2, size2);
  cudaMalloc(&d_out, size3);

  // Copy input data from host to device
  cudaMemcpy(d_in1, input1, size1, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, input2, size2, cudaMemcpyHostToDevice);

  // Launch kernel with enough blocks to cover all elements
  int threadsPerBlock{256};
  int blocksPerGrid{(static_cast<int>(n3) + threadsPerBlock - 1) / threadsPerBlock};
  add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in1, d_in2, d_out, n3);

  // Copy result from device to host
  cudaMemcpy(output, d_out, size3, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_out);
}
