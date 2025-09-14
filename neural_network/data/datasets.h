#pragma once
#include "../tensors.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Dataset
{
public:
    virtual std::pair<int, std::shared_ptr<Tensor>> get_item(int index) = 0;
    virtual int get_length() = 0;
};

class MNIST : public Dataset
{
private:
    // Vector of 2D vectors that represent the image
    std::vector<std::vector<std::vector<float>>> m_images{};
    // Vector of correct labels
    std::vector<int> m_labels{};
    std::vector<std::string> m_classes{"zero", "one", "two",   "three", "four",
                                        "five", "six", "seven", "eight", "nine"};

public:
    MNIST(std::string data_path, std::string labels_path);
    // Return label and image represented in tensor form given an index
    std::pair<int, std::shared_ptr<Tensor>> get_item(int index) override;
    // Returns amount of samples
    int get_length() override;
    std::string label_to_class(int label);
};

// Same as MNIST but with different classes
class FashionMNIST : public Dataset
{
private:
    std::vector<std::vector<std::vector<float>>> m_images{};
    std::vector<int> m_labels{};
    std::vector<std::string> m_classes{
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
    };

public:
    FashionMNIST(std::string data_path, std::string labels_path);
    std::pair<int, std::shared_ptr<Tensor>> get_item(int index) override;
    int get_length() override;
    std::string label_to_class(int label);
};

void visualize_image(std::shared_ptr<Tensor> image);
