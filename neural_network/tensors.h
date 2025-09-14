# ifndef TENSORS_H
# define TENSORS_H
#include <vector>
#include <iostream>
#include <memory>
#include <functional>
#include <utility>

class Tensor : public std::enable_shared_from_this<Tensor>
{
public:
    Tensor() = delete;
    Tensor(float data, bool requires_grad = false,
            std::function<void(const std::vector<float>&)> gradfn = nullptr,
            std::vector<std::shared_ptr<Tensor>> parents = {});
    Tensor(std::vector<float> data, bool requires_grad = false,
            std::function<void(const std::vector<float>&)> gradfn = nullptr,
            std::vector<std::shared_ptr<Tensor>> parents = {});
    Tensor(std::vector<std::vector<float>> data, bool requires_grad = false,
            std::function<void(const std::vector<float>&)> gradfn = nullptr,
            std::vector<std::shared_ptr<Tensor>> parents = {});
    
    const float& operator()(std::size_t i) const;
    float & operator()(std::size_t i);
    const float& operator()(std::size_t i, std::size_t j) const;
    float& operator()(std::size_t i, std::size_t j); 
    const float& item() const;
    float& item();
    // Returns total data size 
    std::size_t numel() const { return m_data.size(); }
    // Returns index of max element
    std::size_t argmax() const;
    void backward();

    void addToGrad(const std::vector<float>& gradUpdate);
    void zeroGrad() { m_grad = std::vector<float>(m_data.size(), 0.0f); }
    
    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);
    friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);

    std::vector<float>& data() { return m_data; }
    const std::vector<std::size_t>& shape() const { return m_shape; }
    const bool& requiresGrad() const { return m_requires_grad; }
    const std::vector<float>& grad() const { return m_grad; }

private:
    std::vector<float> m_data{};
    // 0 for scalar, (col) for 1D, (row, col) for 2D
    std::vector<std::size_t> m_shape{};
    // Used to convert 2D to 1D (ex. [i][j] = i * stride[0] + j * stride[1])
    std::vector<std::size_t> m_stride{};
    std::vector<float> m_grad{};
    // Updates gradient of parents based off of gradient of child, takes in child gradient
    std::function<void(const std::vector<float>&)> m_gradfn{};
    // Gradient tensors
    std::vector<std::shared_ptr<Tensor>> m_parents{};
    bool m_requires_grad{};
    void m_backward();
    bool m_visited{false};
    void m_reset_graph_visit();

};

# endif
