# ifndef TENSORS_H
# define TENSORS_H
#include <vector>
#include <initializer_list>

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

    std::vector<float>& getData() { return m_data;  } 
    
private:
    std::vector<float> m_data{};
    std::vector<std::size_t> m_shape{};
    std::vector<std::size_t> m_stride{};
};

# endif
