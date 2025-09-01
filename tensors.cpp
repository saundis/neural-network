#include "tensors.h"
#include <iostream>
#include <string>

int main() {
    //Tensor test{ std::vector<std::vector<float>> { std::vector<float>{1, 2, 3}, std::vector<float>{1, 2, 3}, std::vector<float>{1, 2, 3}} };
    Tensor test{{1,2,3}};
    test(1) = 3;

    std::cout << test << '\n';
    
    return 0;
}

Tensor::Tensor(float data) 
    : m_data{data}
    , m_shape{}
    , m_stride{}
{
}

// 1D tensors
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

// Tensor multiplication
std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {
    if (m_shape.size() == 0 || other->shape().size() == 0)
    {
        throw std::invalid_argument("Both arguments needs to be at least 1D for matmul.");
    }
    if (m_shape[m_shape.size() - 1] != other->shape()[0])
    {
        throw std::invalid_argument(
            "Last dimension of first tensor doesn't have same size as first dimension of second.");
    }
    // 1D x 1D -> float
    if (m_shape.size() == 1 && other->shape().size() == 1)
    {
        float result = 0.0f;
        for (std::size_t i = 0; i < m_shape[0]; i++)
        {
            result += operator()(i) * (*other)(i);
        }
        return std::make_shared<Tensor>(result);
    }
    // 2D x 1D -> 1D
    else if (m_shape.size() == 2 && other->shape().size() == 1)
    {
        std::vector<float> result{};
        for (std::size_t i = 0; i < m_shape[0]; i++)
        {
            float result_i = 0.0f;
            for (std::size_t j = 0; j < m_shape[1]; j++)
            {
                result_i += operator()(i, j) * (*other)(j);
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }
    // 1D x 2D -> 1D
    else if (m_shape.size() == 1 && other->shape().size() == 2)
    {
        std::vector<float> result{};
        for (std::size_t i = 0; i < other->shape()[1]; i++)
        {
            float result_i = 0.0f;
            for (std::size_t j = 0; j < other->shape()[0]; j++)
            {
                result_i += operator()(j) * (*other)(j, i);
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }
    // 2D x 2D
    else
    {
        if (other->shape().size() < 2)
        {
            throw std::invalid_argument(
                "Expected second tensor to have at least 2 dimensions for this operation");
        }
        std::vector<std::vector<float>> result{};
        for (std::size_t i = 0; i < shape()[0]; i++)
        {
            std::vector<float> result_i{};
            for (std::size_t j = 0; j < other->shape()[1]; j++)
            {
                float result_i_j = 0.0f;
                for (std::size_t k = 0; k < shape()[1]; k++)
                {
                    result_i_j += operator()(i, k) * (*other)(k, j);
                }
                result_i.push_back(result_i_j);
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }
}

// Tensor addition
std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
    // Scalar + scalar
    if (m_shape.size() == 0 && other -> shape().size() == 0) {
        float result = item() + other->item();
        return std::make_shared<Tensor>(result);
    }
    // Scalar + 1D
    if (m_shape.size() == 0 && other -> shape().size() == 1) {
        std::vector<float> result{};
        for (std::size_t i{0}; i < other->shape()[0]; ++i) {
            result.push_back(item()+((*other)(i)));
        }
        return std::make_shared<Tensor>(result);
    }
    // scalar + 2D
    if (m_shape.size() == 0 && other -> shape().size() == 2) {
        std::vector<std::vector<float>> result{};
        for (std::size_t i{0}; i < other->shape()[0]; ++i) {
            std::vector<float> result_i{};
            for (std::size_t j{0}; j < other->shape()[1]; ++j) {
                result_i.push_back(item() + (*other)(i, j));
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }
    // 1D + Scalar
    if (other -> shape().size() == 0 && m_shape.size() == 1) {
        std::vector<float> result{};
        for (std::size_t i{0}; i < shape()[0]; ++i) {
            result.push_back(operator()(i)+(other -> item()));
        }
        return std::make_shared<Tensor>(result);
    }
    // 2D + Scalar
    if (m_shape.size() == 2 && other -> shape().size() == 0) {
        std::vector<std::vector<float>> result{};
        for (std::size_t i{0}; i < shape()[0]; ++i) {
            std::vector<float> result_i{};
            for (std::size_t j{0}; j < shape()[1]; ++j) {
                result_i.push_back(operator()(i, j) + other -> item());
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }
    // 1D + 1D
    if (m_shape[0] != other -> shape()[0]) {
        throw std::invalid_argument("First dimensions not equal.");
    }
    if (m_shape.size() == 1) {
        std::vector<float> result{};
        for (std::size_t i{0}; i < shape()[0]; ++i) {
            result.push_back(operator()(i) + (*other)(i));
        }

        return std::make_shared<Tensor>(result);
    }
    else {
        if (shape()[1] != other -> shape()[1]) {
            throw std::invalid_argument("Second dimensions are not equal.");
        }
        std::vector<std::vector<float>> result{};
        for (std::size_t i{0}; i < shape()[0]; ++i) {
            std::vector<float> result_i{};
            for (std::size_t j{0}; j < shape()[1]; ++j) {
                result_i.push_back(operator()(i, j) + (*other)(i, j));
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }
}

std::ostream& operator<<(std::ostream& os, const Tensor& obj) {
    std::string string_repr = "[";
    if (obj.shape().size() == 0)
    {
        os << obj.item();
        return os;
    }
    else if (obj.shape().size() == 1)
    {
        for (std::size_t i = 0; i < obj.shape()[0]; i++)
        {
            string_repr += std::to_string(obj(i));
            if (i != obj.shape()[0] - 1)
            {
                string_repr += ", ";
            }
        }
        string_repr += "]";
    }
    else
    {
        for (std::size_t i = 0; i < obj.shape()[0]; i++)
        {
            string_repr += "[";
            for (std::size_t j = 0; j < obj.shape()[1]; j++)
            {
                string_repr += std::to_string(obj(i, j));
                if (j != obj.shape()[1] - 1)
                {
                    string_repr += ", ";
                }
            }
            string_repr += "]";
            if (i != obj.shape()[0] - 1)
            {
                string_repr += ", ";
            }
        }
        string_repr += "]";
    }
    os << string_repr;
    return os;
}
