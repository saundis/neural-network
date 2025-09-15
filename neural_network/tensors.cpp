#include "tensors.h"
#include <iostream>
#include <string>
#include <string_view>
#include <algorithm>

extern "C" void runKernel(const float* input1, const float* input2, float* output, 
    std::size_t n1, std::size_t n2, std::size_t n3, std::string_view operation, std::size_t subSize = 0, std::size_t curr = 0);

// Scalar
Tensor::Tensor(float data, bool requires_grad,
            std::function<void(const std::vector<float>&)> gradfn,
            std::vector<std::shared_ptr<Tensor>> parents) 
        : m_data{data}, m_requires_grad{requires_grad}
        , m_gradfn{gradfn}, m_parents{parents}
        , m_shape{}, m_stride{}
{
    if (m_requires_grad) {
        zeroGrad();
    }
}

// 1D tensor
Tensor::Tensor(std::vector<float> data, bool requires_grad,
            std::function<void(const std::vector<float>&)> gradfn,
            std::vector<std::shared_ptr<Tensor>> parents)
        : m_data{data}, m_requires_grad{requires_grad}
        , m_gradfn{gradfn}, m_parents{parents}
        , m_shape{data.size()}, m_stride{1}
{
    if (m_requires_grad) {
        zeroGrad();
    }
}

// 2D tensor
Tensor::Tensor(std::vector<std::vector<float>> data, bool requires_grad,
            std::function<void(const std::vector<float>&)> gradfn,
            std::vector<std::shared_ptr<Tensor>> parents)
        : m_data{}, m_requires_grad{requires_grad}
        , m_gradfn{gradfn}, m_parents{parents}
        , m_shape{data.size(), data[0].size()}, m_stride{data[0].size(), 1}
{
    std::size_t expectedColumnAmount{data[0].size()};

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
    if (m_requires_grad) {
        zeroGrad();
    }
}

// Only called on scalar output that requires grad
void Tensor::backward() {
    // public interface to set initial gradient
    if (!m_requires_grad) {
        throw std::runtime_error("Element does not require grad.");
    }
    if (m_shape.size() != 0) {
        throw std::runtime_error("Grad can only be calculated for scalar outputs.");
    }
    m_reset_graph_visit();
    m_grad = {1.0f};
    m_backward();
}

// Start from final output and iterate over computation graph in backwards order (backwards pass)
void Tensor::m_backward() {
    if (!m_requires_grad) {
        return;
    }
    if (m_visited) {
        return;
    }
    m_visited = true;
    if (m_gradfn) {
        m_gradfn(m_grad);
    }
    for (std::size_t i{0}; i < m_parents.size(); ++i) {
        m_parents[i] -> m_backward();
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

std::size_t Tensor::argmax() const {
    return std::distance(m_data.begin(), std::max_element(m_data.begin(), m_data.end()));
}

// Indexing into 1D tensors
const float& Tensor::operator()(std::size_t i) const {
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

// Adds gradient update to currently stored gradient
void Tensor::addToGrad(const std::vector<float>& gradUpdate) {
    if (!m_requires_grad) {
        return;
    }
    if (m_grad.size() != gradUpdate.size()) {
        throw std::runtime_error("Gradient shape mismatch during accumlation.");
    }
    for (std::size_t i{0}; i < m_grad.size(); ++i) {
        m_grad[i] += gradUpdate[i];
    }
}


// Tensor multiplication
std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {
    if (m_shape.size() == 0 || other->shape().size() == 0) {
        throw std::invalid_argument("Both arguments needs to be at least 1D for matmul.");
    }
    if (m_shape[m_shape.size() - 1] != other->shape()[0]) {
        throw std::invalid_argument(
            "Last dimension of first tensor doesn't have same size as first dimension of second.");
    }
    // 1D x 1D -> float
    if (m_shape.size() == 1 && other->shape().size() == 1) {
        float result = 0.0f;
        for (std::size_t i = 0; i < m_shape[0]; i++) {
            result += operator()(i) * (*other)(i);
        } 
        if (m_requires_grad || other -> requiresGrad()) {
            std::shared_ptr<Tensor> self{ shared_from_this() };
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn{
                [self, other](const std::vector<float>& grad_output){
                    std::vector<float> grad_self{};
                    std::vector<float> grad_other{};
                    for (std::size_t i{0}; i < self -> numel(); ++i) {
                        // Output gradients is a scalar and gets propagated back to the local gradients
                        grad_self.push_back((*other)(i) * grad_output[0]);
                        grad_other.push_back((*self)(i) * grad_output[0]);
                    }
                    self -> addToGrad(grad_self);
                    other -> addToGrad(grad_other);
                }
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 2D x 1D -> 1D
    else if (m_shape.size() == 2 && other->shape().size() == 1) {
        std::vector<float> result{};
        for (std::size_t i = 0; i < m_shape[0]; i++) {
            float result_i = 0.0f;
            for (std::size_t j = 0; j < m_shape[1]; j++)
            {
                result_i += operator()(i, j) * (*other)(j);
            }
            result.push_back(result_i);
        }
        if (m_requires_grad || other -> requiresGrad()) {
            std::shared_ptr<Tensor> self{ shared_from_this() };
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn{
                [self, other](const std::vector<float>& grad_output){
                    std::vector<float> grad_self{};
                    // Iterate using row major order
                    for (std::size_t i = 0; i < self -> shape()[0]; i++) {
                        for (std::size_t j = 0; j < self -> shape()[1]; j++)
                        {
                            // Propagate child gradient according to row (i) multiplied by other tensor
                            grad_self.push_back((*other)(j) * grad_output[i]);
                        }
                    }
                    std::vector<float> grad_other{};
                    for (std::size_t i = 0; i < self -> shape()[0]; i++) {
                        float result{0.0f};
                        // Iterate through rows
                        for (std::size_t j = 0; j < self -> shape()[0]; j++)
                        {
                            // Sum all the indices where other[i] is used by iterating through rows and propagating output[j]
                            result += (*self)(j, i) * grad_output[j];
                        }
                        grad_other.push_back(result);
                    }
                    self -> addToGrad(grad_self);
                    other -> addToGrad(grad_other);
                }
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 1D x 2D -> 1D
    else if (m_shape.size() == 1 && other->shape().size() == 2) {
        std::vector<float> result{};
        for (std::size_t i{0}; i < other->shape()[1]; ++i)
        {
            float result_i = 0.0f;
            for (std::size_t j{0}; j < other->shape()[0]; ++j)
            {
                result_i += operator()(j) * (*other)(j, i);
            }
            float result2{0.0f};
            // Turns out this is a lot slower
            // runKernel(m_data.data(), ((other->data()).data()), &result2,
            //           m_data.size(), (other->data()).size(), 1, "multiply", static_cast<std::size_t>(other->shape()[1]), i);
            std::cout << "REAL: " << result_i << " NEW: " << result2 << '\n';
            result.push_back(result_i);
        }
        if (m_requires_grad || other -> requiresGrad()) {
            std::shared_ptr<Tensor> self{ shared_from_this() };
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn{
                [self, other](const std::vector<float>& grad_output){
                    std::vector<float> grad_self{};
                    for (std::size_t i{0}; i < other -> shape()[0]; i++) {
                        float result{0.0f};
                        for (std::size_t j{0}; j < other -> shape()[1]; j++)
                        {
                            // Sum all the indices where self[i] is used by iterating through cols and propagating output[j]
                            result += (*other)(i, j) * grad_output[j];
                        }
                        grad_self.push_back(result);
                    }
                    std::vector<float> grad_other{};
                    for (std::size_t i = 0; i < other -> shape()[0]; ++i) {
                        for (std::size_t j = 0; j < other -> shape()[1]; ++j)
                        {
                            grad_other.push_back((*self)(i) * grad_output[j]);
                        }
                    }
                    self -> addToGrad(grad_self);
                    other -> addToGrad(grad_other);
                }
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
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
            for (std::size_t j = 0; j < other->shape()[1]; ++j)
            {
                float result_i_j = 0.0f;
                for (std::size_t k = 0; k < shape()[1]; ++k)
                {
                    result_i_j += operator()(i, k) * (*other)(k, j);
                }
                result_i.push_back(result_i_j);
            }
            result.push_back(result_i);
        }
        if (m_requires_grad || other -> requiresGrad()) {
            std::shared_ptr<Tensor> self{ shared_from_this() };
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn{
                [self, other](const std::vector<float>& grad_output){
                    std::vector<float> grad_self{};
                    // Iterate over rows self
                    for (std::size_t i{0}; i < self -> shape()[0]; ++i) {
                        // Iterate over columns self
                        for (std::size_t j{0}; j < self -> shape()[1]; ++j)
                        {
                            float grad_self_i_j{0.0f};
                            // Iterate over row other
                            for (std::size_t k{0}; j < other -> shape()[1]; ++k)
                            {
                                // propagate corresponding child grad
                                grad_self_i_j += 
                                    (*other)(j, k) * grad_output[i * other->shape()[1] + k];
                            }
                            grad_self.push_back(grad_self_i_j);
                        }
                    }
                    std::vector<float> grad_other{};
                    // Iterate over rows self
                    for (std::size_t i{0}; i < other -> shape()[0]; ++i) {
                        // Iterate over columns self
                        for (std::size_t j{0}; j < other -> shape()[1]; ++j)
                        {
                            float grad_self_i_j{0.0f};
                            // Iterate over row other
                            for (std::size_t k{0}; j < self -> shape()[0]; ++k)
                            {
                                // propagate corresponding child grad
                                grad_self_i_j += (*self)(k, i) * grad_output[k * other->shape()[1] + k];
                            }
                            grad_self.push_back(grad_self_i_j);
                        }
                    }
                    self -> addToGrad(grad_self);
                    other -> addToGrad(grad_other);
                }
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
}

// Tensor addition
std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
    // Scalar + scalar
    if (m_shape.size() == 0 && other -> shape().size() == 0) {
        float result{item() + other->item()};
        if (m_requires_grad || other -> requiresGrad()) {
            std::shared_ptr<Tensor> self{ shared_from_this() };
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float>&)> gradfn{
                [self, other](const std::vector<float>& grad_output) {
                    // Propagate parent gradient
                    self -> addToGrad({grad_output[0]});
                    other -> addToGrad({grad_output[0]});
                }
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // Scalar + 1D
    if (m_shape.size() == 0 && other -> shape().size() == 1) {
        std::vector<float> result(other->shape()[0]);
        runKernel(m_data.data(), (other->data()).data(), result.data(),
                  m_data.size(), (other->data()).size(), result.size(), "add");
        if (m_requires_grad || other -> requiresGrad()) {
            std::shared_ptr<Tensor> self{ shared_from_this() };
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn{
                [self, other](const std::vector<float>& grad_output){
                    // Broadcast in forward = sum in backward
                    float grad_self{0.0f};
                    float test_grad{0.0f};
                    for (std::size_t i{0}; i < grad_output.size(); ++i) {
                        test_grad += grad_output[i];
                    }
                    runKernel((self->data()).data(), nullptr, &grad_self,
                            (self->data()).size(), 0, 1, "sum");
                    std::cout << "Real: " << test_grad << " New: " << grad_self << '\n';
                    self -> addToGrad({grad_self});
                    other -> addToGrad(grad_output);
                }
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // scalar + 2D
    if (m_shape.size() == 0 && other -> shape().size() == 2) {
        std::vector<std::vector<float>> result{};
        for (std::size_t i{0}; i < other->shape()[0]; ++i) {
            std::vector<float> result_i(other->shape()[1]);
            runKernel(m_data.data(), ((other->data()).data()) + (i * (other->shape()[1])), result_i.data(),
                  m_data.size(), static_cast<std::size_t>(other->shape()[1]), result_i.size(), "add");
            result.push_back(result_i);
        }
        if (m_requires_grad || other -> requiresGrad()) {
            std::shared_ptr<Tensor> self{ shared_from_this() };
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn{
                [self, other](const std::vector<float>& grad_output){
                    // Broadcast in forward = sum in backward
                    float grad_self{0.0f};
                    for (std::size_t i{0}; i < grad_output.size(); ++i) {
                        grad_self += grad_output[i];
                    }
                    self -> addToGrad({grad_self});
                    other -> addToGrad(grad_output);
                }
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 1D + Scalar
    if (other -> shape().size() == 0 && m_shape.size() == 1) {
        std::vector<float> result(shape()[0]);
        runKernel(m_data.data(), (other->data()).data(), result.data(),
                  m_data.size(), (other->data()).size(), result.size(), "add");
        if (m_requires_grad || other -> requiresGrad()) {
            std::shared_ptr<Tensor> self{ shared_from_this() };
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn{
                [self, other](const std::vector<float>& grad_output){
                    // Broadcast in forward = sum in backward
                    float grad_other{0.0f};
                    for (std::size_t i{0}; i < grad_output.size(); ++i) {
                        grad_other += grad_output[i];
                    }
                    self -> addToGrad(grad_output);
                    other -> addToGrad({grad_other});
                }
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
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
        if (m_requires_grad || other -> requiresGrad()) {
            std::shared_ptr<Tensor> self{ shared_from_this() };
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn{
                [self, other](const std::vector<float>& grad_output){
                    // Broadcast in forward = sum in backward
                    float grad_other{0.0f};
                    for (std::size_t i{0}; i < grad_output.size(); ++i) {
                        grad_other += grad_output[i];
                    }
                    self -> addToGrad(grad_output);
                    other -> addToGrad({grad_other});
                }
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 1D + 1D
    if (m_shape[0] != other -> shape()[0]) {
        throw std::invalid_argument("First dimensions not equal.");
    }
    if (m_shape.size() == 1) {
        std::vector<float> result(shape()[0]);
        runKernel(m_data.data(), (other->data()).data(), result.data(),
                  m_data.size(), (other->data()).size(), result.size(), "add");
        if (m_requires_grad || other -> requiresGrad()) {
            std::shared_ptr<Tensor> self{ shared_from_this() };
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn{
                [self, other](const std::vector<float>& grad_output){
                    // Propagate child gradient
                    self -> addToGrad(grad_output);
                    other -> addToGrad(grad_output);
                }
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else {
        // 2D + 2D
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
        if (m_requires_grad || other -> requiresGrad()) {
            std::shared_ptr<Tensor> self{ shared_from_this() };
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn{
                [self, other](const std::vector<float>& grad_output){
                    // Propagate child gradient
                    self -> addToGrad(grad_output);
                    other -> addToGrad(grad_output);
                }
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
}

void Tensor::m_reset_graph_visit() {
    if (!m_visited) {
        return;
    }
    m_visited = false;
    for (std::size_t i{0}; i < m_parents.size(); ++i) {
        m_parents[i]->m_reset_graph_visit();
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
