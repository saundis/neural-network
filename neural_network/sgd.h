#pragma once
#include "./tensors.h"
#include <memory>
#include <string>
#include <vector>

class SGD
{
public:
    SGD(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> params, float lr = 0.001);
    void step();
    void zero_grad();

private:
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> m_params{};
    // Determine how much parameters are being changed each step
    float m_learning_rate{};
};
