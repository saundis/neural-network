#pragma once
#include "./module.h"
#include "../tensors.h"

// Module that breaks linearity; input <= 0 : 0 ? input
class Relu : public Module
{
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
