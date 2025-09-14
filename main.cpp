#include "neural_network/data/dataloader.h"
#include "neural_network/data/datasets.h"
#include "neural_network/modules/flatten.h"
#include "neural_network/modules/linear.h"
#include "neural_network/modules/module.h"
#include "neural_network/modules/relu.h"
#include "neural_network/modules/loss.h"
#include "neural_network/serialization.h"
#include "neural_network/sgd.h"
#include "neural_network/tensors.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

extern "C" void runCudaKernel(const int* input, int* output, std::size_t n);

class NeuralNetwork : public Module
{
public:
    NeuralNetwork()
    {
        register_module("linear_1", m_linear_1);
        register_module("linear_2", m_linear_2);
        register_module("linear_3", m_linear_3);
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input)
    {
        std::shared_ptr<Tensor> flat = (*m_flatten)(input);
        std::shared_ptr<Tensor> linear_1 = (*m_linear_1)(flat);
        std::shared_ptr<Tensor> relu_1 = (*m_relu)(linear_1);
        std::shared_ptr<Tensor> linear_2 = (*m_linear_2)(relu_1);
        std::shared_ptr<Tensor> relu_2 = (*m_relu)(linear_2);
        std::shared_ptr<Tensor> linear_3 = (*m_linear_3)(relu_2);
        return linear_3;
    }

private:
    // Layers
    std::shared_ptr<Flatten> m_flatten{ std::make_shared<Flatten>() };
    std::shared_ptr<Linear> m_linear_1{ std::make_shared<Linear>(28 * 28, 512) };
    std::shared_ptr<Linear> m_linear_2{ std::make_shared<Linear>(512, 512) };
    std::shared_ptr<Linear> m_linear_3{ std::make_shared<Linear>(512, 10) };
    // Activation
    std::shared_ptr<Relu> m_relu = std::make_shared<Relu>();
};

// Training loop
void train(DataLoader &dataloader, NeuralNetwork &model, CrossEntropyLoss &loss_fn, SGD &optimizer) {
    // For logging
    std::size_t log_interval{100};
    std::size_t batch_n{0};
    std::size_t seen_samples{0};

    for (const auto &batch : dataloader) {
        std::shared_ptr<Tensor> total_loss{nullptr};
        std::size_t batch_size{batch.size()};

        // can be optimized to do in parallel
        // Iterate over each label one at a time
        for (const auto &[label, tensor] : batch) {
            // Pass image through neural network and get output
            auto output{model(tensor)};
            // Calculated loss by passing in output and true label
            auto loss{loss_fn(output, label)};
            if (total_loss == nullptr) {
                total_loss = loss;
            }
            else {
                total_loss = (*total_loss) + loss;
            }
            seen_samples += 1;
        }
        total_loss->item() /= batch_size;

        if (batch_n % log_interval == 0) {
            std::cout << "loss: " << std::fixed << std::setprecision(6) << total_loss->item()
                      << "  [" << seen_samples << "/" << dataloader.n_samples() << "]" << std::endl;
        }

        // Calculates gradient
        total_loss->backward();
        // Optimizes
        optimizer.step();
        // Resets grads
        optimizer.zero_grad();
        batch_n += 1;
    }
}

void test(DataLoader &dataloader, NeuralNetwork &model, CrossEntropyLoss &loss_fn) {
    float running_loss{0.0f};
    std::size_t correct{0};
    std::size_t n_samples{0};

    for (const auto &batch : dataloader) {
        for (const auto &[label, tensor] : batch) {
            auto output = model(tensor);
            // Checks if it's accurate
            if (output->argmax() == label) {
                correct += 1;
            }
            running_loss += loss_fn(output, label)->item();
            n_samples += 1;
        }
    }

    std::cout << correct;

    float accuracy{static_cast<float>(correct) / static_cast<float>(n_samples)};
    float avg_loss = running_loss / n_samples;

    std::cout << std::fixed << std::setprecision(6)
              << "Test error:\n  accuracy: " << std::setprecision(1) << accuracy * 100.0 << "%\n"
              << "  avg loss: " << std::setprecision(6) << avg_loss << "\n";
}

void train_new_mnist_model() {
    std::cout << "Loading dataset..." << std::endl;

    // Loading data into tensors
    //MNIST mnist_train{
    //    MNIST("./raw_data/MNIST/train-images-idx3-ubyte", "./raw_data/MNIST/train-labels-idx1-ubyte")};
    //MNIST mnist_test{
    //    MNIST("./raw_data/MNIST/t10k-images-idx3-ubyte", "./raw_data/MNIST/t10k-labels-idx1-ubyte")};

    FashionMNIST mnist_train{FashionMNIST("./raw_data/FashionMNIST/train-images-idx3-ubyte",
                              "./raw_data/FashionMNIST/train-labels-idx1-ubyte")};
    FashionMNIST mnist_test{FashionMNIST("./raw_data/FashionMNIST/t10k-images-idx3-ubyte",
                            "./raw_data/FashionMNIST/t10k-labels-idx1-ubyte")};
    std::cout << "Dataset loaded." << std::endl;

    // Lads data into batches
    int batch_size{10};
    DataLoader train_dataloader(&mnist_train, batch_size);
    DataLoader test_dataloader(&mnist_test, batch_size);

    NeuralNetwork model{};
    CrossEntropyLoss loss_fn{};

    // Amount that will be multiplied by parameters for training
    float learning_rate = 0.001f;
    SGD optimizer(model.parameters(), learning_rate);

    // Amount of times iterating through dataset
    int n_epochs{1};
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        std::cout << "[Epoch ]" << epoch << "/" << n_epochs << "] Training ..." << std::endl;
        train(train_dataloader, model, loss_fn, optimizer);
        std::cout << "[Epoch ]" << epoch << "/" << n_epochs << "] Testing ..." << std::endl;
        test(test_dataloader, model, loss_fn);
    }

    auto state_dict{model.state_dict()};
    //save(state_dict, "./models/mnist.nn");
    save(state_dict, "./models/fashion-mnist.nn");
}

void inference_on_saved_model() {
    NeuralNetwork model;
    std::cout << "Loading model..." << std::endl;
    auto loaded_state_dict = load("models/mnist.nn");
    model.load_state_dict(loaded_state_dict);

    std::cout << "Loading test set..." << std::endl;
    MNIST mnist_test{
        MNIST("./raw_data/MNIST/t10k-images-idx3-ubyte", "./raw_data/MNIST/t10k-labels-idx1-ubyte")};
    //FashionMNIST mnist_test{FashionMNIST("data/FashionMNIST/raw/t10k-images-idx3-ubyte",
    //                                       "data/FashionMNIST/raw/t10k-labels-idx1-ubyte")};
    int n_samples{10};

    std::vector<int> all_indices(mnist_test.get_length());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all_indices.begin(), all_indices.end(), g);
    std::vector<int> indices(all_indices.begin(), all_indices.begin() + n_samples);

    for (int i = 0; i < n_samples; i++)
    {
        std::cout << "Sample " << i << " of " << n_samples << std::endl;
        std::pair<int, std::shared_ptr<Tensor>> sample_image = mnist_test.get_item(indices[i]);
        visualize_image(sample_image.second);
        auto output = model(sample_image.second);
        int predicted_class = output->argmax();
        std::cout << "Predicted class: " << mnist_test.label_to_class(predicted_class) << std::endl;
        std::cout << "Actual class: " << mnist_test.label_to_class(sample_image.first) << std::endl;
        std::cout << "----------------------------" << std::endl;
    }
}

int main() {
    std::vector<int> test{1, 2, 3};
    std::vector<int> result(test.size());

    runCudaKernel(test.data(), result.data(), test.size());

    for (int i{0}; i < 3; ++i) {
        std::cout << "c[" << i << "] = " << result[i] << '\n';
    }

    train_new_mnist_model();
    // inference_on_saved_model();
    return 0;
}
