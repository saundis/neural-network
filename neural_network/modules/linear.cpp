#include "./linear.h"
#include "../tensors.h"
#include <memory>
#include <random>

Linear::Linear(std::size_t in_features, std::size_t out_features, std::size_t seed)
    : m_in_features{in_features}, m_out_features{out_features}
    , m_weight{ std::make_shared<Tensor>(
          // input features is rows, output features is columns
          std::vector<std::vector<float>> (in_features, std::vector<float>(out_features, 0.0f)),
          true) }
    , m_bias{ std::make_shared<Tensor>(std::vector<float> (out_features, 0.0f), true) }
    , m_seed{seed}
{
    // register parameters
    register_parameter("weight", m_weight);
    register_parameter("bias", m_bias);
    // initatilizing
    reset_parameters();
}

// Defaulting the weights
void Linear::reset_parameters()
{
    // Relu gain (same as pytorch)
    float gain = std::sqrt(2.0f);
    std::size_t fan_in = m_in_features;
    float bound = gain * std::sqrt(3.0f / fan_in);
    std::mt19937 generator(m_seed);

    for (std::size_t i = 0; i < m_weight->shape()[0]; i++)
    {
        for (std::size_t j = 0; j < m_weight->shape()[1]; j++)
        {
            (*m_weight)(i, j) = std::uniform_real_distribution<float>(-bound, bound)(generator);
        }
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input)
{
    std::shared_ptr<Tensor> xW{(*input) * m_weight};
    return (*xW) + m_bias;
}
