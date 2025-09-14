#pragma once
#include "../tensors.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Building block for higher level operations
class Module
{
public:
    // Does operation on tnesor and outpujts resulting tensor (must be called on subclass)
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);

    void register_parameter(std::string name, std::shared_ptr<Tensor> param);
    void register_module(std::string name, std::shared_ptr<Module> module);
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> parameters() const;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict() const;
    void load_state_dict(std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict);

    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input) { return forward(input); }

private:
    // Parameters are tensors that we wish to train when using the neural network (weights, etc.)
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> m_parameters{};
    // For if modules have children modules
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> m_modules{};
};
