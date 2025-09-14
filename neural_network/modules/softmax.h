#pragma once
#include "module.h"
#include "../tensors.h"
#include <memory>

// Turns raw output into probabilities
class Softmax : public Module
{
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
