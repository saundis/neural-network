#pragma once
#include "./module.h"
#include "../tensors.h"
#include <memory>

// Module that contains weights and bias for each layer
class Linear : public Module
{
public:
    Linear(std::size_t in_features, std::size_t out_features, std::size_t seed = 7);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    void reset_parameters();

private:
    std::shared_ptr<Tensor> m_weight{};
    std::shared_ptr<Tensor> m_bias{};
    // Size of input tensor
    std::size_t m_in_features{};
    // How many elements we want output to have
    std::size_t m_out_features{};
    std::size_t m_seed{};
};
