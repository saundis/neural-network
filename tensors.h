# ifndef TENSORS_H
# define TENSORS_H
#include <vector>
#include <iostream>
#include <memory>

class Tensor{
public:
    Tensor() = delete;
    Tensor(float data);
    Tensor(std::vector<float> data);
    Tensor(std::vector<std::vector<float>> data);
    
    const float& operator()(std::size_t i) const;
    float & operator()(std::size_t i);
    const float& operator()(std::size_t i, std::size_t j) const;
    float& operator()(std::size_t i, std::size_t j); 
    const float& item() const;
    float& item();

    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);
    friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);

    const std::vector<float>& data() const { return m_data; }
    const std::vector<std::size_t>& shape() const { return m_shape; }
    
private:
    std::vector<float> m_data{};
    std::vector<std::size_t> m_shape{};
    std::vector<std::size_t> m_stride{};
};

# endif
