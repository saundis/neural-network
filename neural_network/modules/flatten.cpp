#include "./flatten.h"
#include "../tensors.h"
#include <functional>
#include <memory>

// input to 1D tensor
std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> input)
{
    std::vector<float> result{};
    // Scalar
    if (input->shape().size() == 0) {
        result.push_back(input->item());
    }
    // 1D vector
    else if (input->shape().size() == 1) {
        for (std::size_t i = 0; i < input->shape()[0]; i++) {
            result.push_back((*input)(i));
        }
    }
    // 2D vector
    else {
        for (std::size_t i = 0; i < input->shape()[0]; i++) {
            for (std::size_t j = 0; j < input->shape()[1]; j++) {
                result.push_back((*input)(i, j));
            }
        }
    }
    if (input->requiresGrad()) {
        std::vector<std::shared_ptr<Tensor>> parents{input};

        std::function<void(const std::vector<float> &)> gradfn{
            [input](const std::vector<float> &grad_output){
                // Propagate grad of the output
                input -> addToGrad(grad_output);
            }};
        return std::make_shared<Tensor>(result, true, gradfn, parents);
    }
    return std::make_shared<Tensor>(result);
}
