#include "./sgd.h"
#include "./tensors.h"
#include <memory>
#include <string>
#include <vector>

SGD::SGD(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> params, float lr)
    : m_params(params), m_learning_rate(lr)
{
}

// Iterates over each parameter and trains them
void SGD::step() {
    for (auto &param : m_params){
        for (std::size_t i{0}; i < param.second->numel(); i++) {
            param.second->data()[i] -= m_learning_rate * param.second->grad()[i];
        }
    }
}

// Resets all graidents to 0
void SGD::zero_grad() {
    for (auto &param : m_params) {
        param.second-> zeroGrad();
    }
}
