#include "tensors.h"
#include <iostream>

Tensor::Tensor(float data) 
    : m_data{data}
    , m_shape{}
    , m_stride{}
{
}

// Possible optimization using C style arrays with move semantics?
Tensor::Tensor(std::vector<float> data)
    : m_shape{data.size()}
    , m_data{data}
    , m_stride{1}
{
}

// 2D tensors
Tensor::Tensor(std::vector<std::vector<float>> data)
    : m_shape{data.size(), data[0].size()}
    , m_stride{data[0].size(), 1}
{
    std::size_t expectedColumnAmount = data[0].size();

    if (data.size() != expectedColumnAmount) {
        throw std::invalid_argument("Dimensions are inconsisted.");
    }

    for (std::size_t i{0}; i < data.size(); ++i) {
        if (data[i].size() != expectedColumnAmount) {
            throw std::invalid_argument("Dimensions are inconsisted.");
        }
    }

    // Store in row major order (row by row contiguously)
    for (std::size_t i{0}; i < data.size(); ++i) {
        for (std::size_t j{0}; j < data[0].size(); ++j) {
            m_data.push_back(data[i][j]);
        }
    }
}

// Only works with scalars and 1D tensors
const float& Tensor::item() const {
    if (m_data.size() == 1) {
        return m_data[0];
    } else {
        throw std::runtime_error("item() must be called on tensor with a single element");
    }
}

// Only works with scalars
float& Tensor::item() {
    if (m_data.size() == 1) {
        return m_data[0];
    } else {
        throw std::runtime_error("item() must be called on tensor with a single element");
    }
}

// Indexing into 1D tensors
const float& Tensor::operator()(std::size_t i) const {
    if (m_shape.size() == 0) {
        throw std::invalid_argument("Can't index into scalar. Use item() instead.");
    } else if (m_shape.size() == 1) {
        if (i >= m_shape[0]) {
            std::invalid_argument("Indice out of bounds");
        }
        return m_data[i];
    }
    throw std::invalid_argument("Use two indices for 2D tensors.");
}

// Indexing into 1D tensors
float & Tensor::operator()(std::size_t i) {
    if (m_shape.size() == 0) {
        throw std::invalid_argument("Can't index into scalar. Use item() instead.");
    } else if (m_shape.size() == 1) {
        if (i >= m_shape[0]) {
            throw std::invalid_argument("Indice out of bounds");
        }
        return m_data[i];
    }
    throw std::invalid_argument("Use two indices for 2D tensors.");
}

// Indexing into 2D tensors (Row, column)
const float& Tensor::operator()(std::size_t i, std::size_t j) const {
    if (m_shape.size() == 2) {
        if (i >= m_shape[0]) {
            throw std::invalid_argument("Row index out of range");
        } else if (j >= m_shape[1]) {
            throw std::invalid_argument("Column index out of range");
        }
        return m_data[i * m_stride[0] + j * m_stride[1]];
    }
    throw std::invalid_argument("2 indices only work with 2D items.");
}

// Indexing into 2D tensors (Row, column)
float& Tensor::operator()(std::size_t i, std::size_t j) {
    if (m_shape.size() == 2) {
        if (i >= m_shape[0]) {
            throw std::invalid_argument("Row index out of range");
        } else if (j >= m_shape[1]) {
            throw std::invalid_argument("Column index out of range");
        }
        return m_data[i * m_stride[0] + j * m_stride[1]];
    }
    throw std::invalid_argument("2 indices only work with 2D items.");
}

int main() {
    Tensor test{ std::vector<std::vector<float>> { std::vector<float>{1, 2, 3}, std::vector<float>{1, 2, 3}, std::vector<float>{1, 2, 3}} };

    return 0;
}
