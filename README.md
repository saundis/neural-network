# Neural Network

A minimal PyTorch-inspired deep-learning framework written from scratch in **C++**.

## Features

- **Custom tensor library** with automatic differentiation (`backward()`, gradient accumulation, `zero_grad()`)
- **PyTorch-style `Module` API** — subclass `Module`, register children with `register_module`, override `forward`
- **Layers**: `Linear`, `Flatten`, `ReLU`, `Softmax`
- **Loss**: `CrossEntropyLoss`
- **Optimizer**: stochastic gradient descent (`SGD`)
- **Data pipeline**: `DataLoader` with batching, plus built-in `MNIST` and `FashionMNIST` dataset loaders that read the original IDX file format
- **Model serialization**: `save` / `load` a model's `state_dict` to disk
- **CUDA acceleration** via custom `.cu` kernels alongside the CPU code paths

## Project structure

```
neural-network/
├── main.cpp                    # Training + inference entry point
├── cuda_kernel.cu              # Example CUDA kernel
├── CMakeLists.txt              # Build configuration
├── neural_network/
│   ├── tensors.cpp / .cu / .h  # Tensor class + CUDA ops
│   ├── serialization.cpp       # save / load state_dict
│   ├── sgd.cpp                 # SGD optimizer
│   ├── modules/
│   │   ├── module.cpp          # Base Module class
│   │   ├── linear.cpp          # Fully-connected layer
│   │   ├── flatten.cpp         # Flatten layer
│   │   ├── relu.cpp            # ReLU activation
│   │   ├── softmax.cpp         # Softmax activation
│   │   └── loss.cpp            # CrossEntropyLoss
│   └── data/
│       ├── dataloader.cpp      # Batched iterable over a dataset
│       └── datasets.cpp        # MNIST / FashionMNIST loaders
├── raw_data/                   # Place MNIST / Fashion-MNIST IDX files here
└── models/                     # Saved .nn checkpoints
```

## Requirements

- **CMake** ≥ 3.20
- A **C++17** compiler
- **CUDA Toolkit** (for `nvcc` and `CUDA::cudart`) — CUDA standard is set to 17
- An NVIDIA GPU with CUDA support

## Building

```bash
git clone https://github.com/saundis/neural-network.git
cd neural-network

cmake -S . -B build
cmake --build build
```

This produces an executable named `CudaCombined` inside the `build/` directory.

## Datasets

The example expects the original IDX-format files from the MNIST and Fashion-MNIST archives. Place them like so:

```
raw_data/
├── MNIST/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── FashionMNIST/
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
```

- MNIST: http://yann.lecun.com/exdb/mnist/
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist

## Running

From the project root (so the relative paths to `raw_data/` and `models/` resolve):

```bash
./build/CudaCombined
```

By default, `main()` calls `train_new_mnist_model()`, which:

1. Loads the Fashion-MNIST training and test sets,
2. Builds a 3-layer MLP (`784 → 512 → 512 → 10`) with ReLU activations,
3. Trains for 1 epoch with `SGD` (`lr = 0.001`, batch size 10),
4. Evaluates on the test set,
5. Saves the trained weights to `./models/fashion-mnist.nn`.

To run inference on a saved checkpoint instead, swap which function is called in `main()`:

```cpp
int main() {
    // train_new_mnist_model();
    inference_on_saved_model();
    return 0;
}
```

`inference_on_saved_model()` loads `models/mnist.nn`, picks 10 random samples from the MNIST test set, and prints the predicted vs. actual class for each.

## Example: defining a model

The framework's `Module` API will look familiar if you've used PyTorch:

```cpp
class NeuralNetwork : public Module {
public:
    NeuralNetwork() {
        register_module("linear_1", m_linear_1);
        register_module("linear_2", m_linear_2);
        register_module("linear_3", m_linear_3);
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) {
        auto x = (*m_flatten)(input);
        x = (*m_relu)((*m_linear_1)(x));
        x = (*m_relu)((*m_linear_2)(x));
        return (*m_linear_3)(x);
    }

private:
    std::shared_ptr<Flatten> m_flatten{ std::make_shared<Flatten>() };
    std::shared_ptr<Linear>  m_linear_1{ std::make_shared<Linear>(28 * 28, 512) };
    std::shared_ptr<Linear>  m_linear_2{ std::make_shared<Linear>(512, 512) };
    std::shared_ptr<Linear>  m_linear_3{ std::make_shared<Linear>(512, 10) };
    std::shared_ptr<Relu>    m_relu    { std::make_shared<Relu>() };
};
```

And a training step:

```cpp
auto output = model(tensor);
auto loss   = loss_fn(output, label);

loss->backward();      // populate gradients
optimizer.step();      // update parameters
optimizer.zero_grad(); // clear gradients for next batch
```

## Roadmap / ideas

- More layers (Conv2D, BatchNorm, Dropout)
- Additional optimizers (Adam, RMSProp)
- Full GPU training path (currently mixed CPU/CUDA)
- Mini-batch parallelism in the training loop
