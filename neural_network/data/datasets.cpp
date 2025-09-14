#include "./datasets.h"
#include "../tensors.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float convert_to_float(unsigned char px) { return static_cast<float>(px) / 255.0f; }

// Loads in image and returns a vector of image with corresponding grayscale values
std::vector<std::vector<std::vector<float>>> read_mnist(std::string path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<std::vector<std::vector<float>>> dataset{};
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        // Mnist files are big endian so need to reverse int
        magic_number = reverse_int(magic_number);
        if (magic_number != 2051) {
            throw std::runtime_error("Invalid MNIST image file!");
        }
        file.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);
        std::cout<< "Rows: " << n_rows << " Columns: " << n_cols << " Images: "<< number_of_images << '\n';
        for (int i{0}; i < number_of_images; ++i) {
            std::vector<std::vector<float>> image{};
            for (int r{0}; r < n_rows; ++r) {
                std::vector<float> row{};
                for (int c{0}; c < n_cols; ++c) {
                    unsigned char temp=0;
                    file.read((char *)&temp, sizeof(temp));
                    row.push_back(convert_to_float(temp));
                }
                image.push_back(row);
            }
            dataset.push_back(image);
        }
    }
    return dataset;
}

// Parses the label data to get correct label data (0-9) as a vector
std::vector<int> read_mnist_labels(std::string path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<int> labels{};
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_items = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        if (magic_number != 2049) {
            throw std::runtime_error("Invalid MNIST label file!");
        }
        file.read((char *)&number_of_items, sizeof(number_of_items));
        // mnist files are big endian so need to reverse int
        number_of_items = reverse_int(number_of_items);
        for (int i = 0; i < number_of_items; ++i) {
            unsigned char label = 0;
            file.read((char *)&label, sizeof(label));
            labels.push_back(label);
        }
    }
    return labels;
}

MNIST::MNIST(std::string data_path, std::string labels_path) {
    m_images = read_mnist(data_path);
    m_labels = read_mnist_labels(labels_path);
}

// Given an index, return associated label and image
std::pair<int, std::shared_ptr<Tensor>> MNIST::get_item(int index) {
    return std::make_pair(m_labels[index], std::make_shared<Tensor>(m_images[index]));
}

int MNIST::get_length() { return m_images.size(); }

std::string MNIST::label_to_class(int label) { return m_classes[label]; }

FashionMNIST::FashionMNIST(std::string data_path, std::string labels_path) {
    m_images = read_mnist(data_path);
    m_labels = read_mnist_labels(labels_path);
}

std::pair<int, std::shared_ptr<Tensor>> FashionMNIST::get_item(int index) {
    return std::make_pair(m_labels[index], std::make_shared<Tensor>(m_images[index]));
}

int FashionMNIST::get_length() { return m_images.size(); }

std::string FashionMNIST::label_to_class(int label) { return m_classes[label]; }

// For visualizing the images in ASCII
void visualize_image(std::shared_ptr<Tensor> image) {
    for (int i = 0; i < image->shape()[0]; ++i) {
        for (int j = 0; j < image->shape()[1]; ++j) {
            float px{(*image)(i, j)};
            std::cout << (px > 0.75   ? '@'
                          : px > 0.5  ? '#'
                          : px > 0.25 ? '+'
                          : px > 0.1  ? '.'
                                      : ' ');
        }
        std::cout << '\n';
    }
}
