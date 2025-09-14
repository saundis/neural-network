cmake_minimum_required(VERSION 3.20)
project(CudaCombined LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Explicitly find the CUDA Toolkit package.
# This ensures that CMake creates the `CUDA::cudart` target.
find_package(CUDAToolkit REQUIRED)

add_executable(CudaCombined 
main.cpp 
neural_network/tensors.cpp 
neural_network/serialization.cpp 
neural_network/sgd.cpp 
neural_network/modules/flatten.cpp 
neural_network/modules/linear.cpp 
neural_network/modules/module.cpp 
neural_network/modules/relu.cpp 
neural_network/modules/softmax.cpp 
neural_network/modules/loss.cpp 
neural_network/data/datasets.cpp 
neural_network/data/dataloader.cpp 
cuda_kernel.cu 
neural_network/tensors.cu 
)

# Link to the `CUDA::cudart` imported target.
target_link_libraries(CudaCombined PRIVATE CUDA::cudart)
