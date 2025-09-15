#include <cuda_runtime.h>
#include <string_view>
#include <stdio.h>
#include <iostream>

__global__ void add_kernel(const float* input1, const float* input2, float* output, std::size_t n1, std::size_t n2, std::size_t n3) {
  std::size_t i{static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x)};

   if (i >= n3) {
      return;
   }

  if (n1 == 1) {
    // 1D + scalar
      output[i] = input1[0] + input2[i];
  }
  else if (n2 == 1) {
    // scalar + 1D 
      output[i] = input1[i] + input2[0];
  }
  else if (n1 == n2 && n1 == n3) {
    // 1D + 1D 
      output[i] = input1[i] + input2[i];
  }
}

__global__ void sum_kernel(const float* input, float* output, std::size_t n) {
  std::size_t i{static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x)};

   if (i >= n) {
      return;
   }
   (*output) += input[i];
}

// 1D x 2D (most relevant operation)
__global__ void multiply_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ out,
                                std::size_t n1, std::size_t n2, std::size_t subSize, std::size_t curr) {
    extern __shared__ float s[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;

    if (i < n1) {
        int j = i * subSize + curr;
        if (j < n2) val = a[i] * b[j];
    }
    s[threadIdx.x] = val;
    __syncthreads();

    // tree reduce within block
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, s[0]);
}

// For any operation with two input tensors and one output tensor
extern "C" void runKernel(const float* input1, const float* input2, float* output, std::size_t n1, std::size_t n2, std::size_t n3, std::string_view operation, std::size_t subSize = 0, std::size_t curr = 0) {
  float *d_in1, *d_in2, *d_out;
  std::size_t size1{n1 * sizeof(float)};
  std::size_t size2{n2 * sizeof(float)};
  std::size_t size3{n3 * sizeof(float)};

  // Allocate device memory
  cudaMalloc(&d_in1, size1);
  cudaMalloc(&d_in2, size2);
  cudaMalloc(&d_out, size3);
  if (n3 == 1) {
    cudaMemset(d_out, 0, sizeof(float));
  }

  // Copy input data from host to device
  if (input1 != nullptr) {
    cudaMemcpy(d_in1, input1, size1, cudaMemcpyHostToDevice);
  }
  if (input2 != nullptr) {
    cudaMemcpy(d_in2, input2, size2, cudaMemcpyHostToDevice);
  }

  // Launch kernel with enough blocks to cover all elements
  int threadsPerBlock{256};
  int blocksPerGrid{(static_cast<int>(n3) + threadsPerBlock - 1) / threadsPerBlock};
  if (operation == "add") {
      add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in1, d_in2, d_out, n1, n2, n3);
  } else if (operation == "multiply") {
      // Only does 1D x 2D since that is the main operation for training
      blocksPerGrid = (static_cast<int>(n1) + threadsPerBlock - 1) / threadsPerBlock;
      multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in1, d_in2, d_out, n1, n2, subSize, curr);
  } else if (operation == "sum") {
    if (n1 == 0) {
      // Sum all of the second array and return to output
      sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in2, d_out, n2);
    } else {
      // Sum all of the first array and return to output
      sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in1, d_out, n1);
    }
  }

  cudaDeviceSynchronize();


  // Copy result from device to host
  cudaMemcpy(output, d_out, size3, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_out);
}
