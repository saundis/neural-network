#pragma once
#include "./module.h"
#include "../tensors.h"
#include <memory>

// Module to flatten input into tensors (scalar, 1D, 2D)
class Flatten : public Module
{
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
