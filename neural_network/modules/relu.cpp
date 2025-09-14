#include "./relu.h"
#include "../tensors.h"
#include <functional>
#include <memory>

// https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
std::shared_ptr<Tensor> Relu::forward(std::shared_ptr<Tensor> input) {
    // Scalar
    if (input->shape().size() == 0) {
        float result{};
        if (input->item() > 0) {
            result = input->item();
        }
        else {
            result = 0.0f;
        }
        if (input->requiresGrad()) {
            std::vector<std::shared_ptr<Tensor>> parents{input};

            std::function<void(const std::vector<float> &)> gradfn {
                [input](const std::vector<float> &grad_output) {
                        std::vector<float> grad_input{};
                        if (input->item() > 0) {
                            grad_input.push_back(grad_output[0]);
                        }
                        else {
                            grad_input.push_back(0.0f);
                        }
                        input->addToGrad(grad_input);
                    }};
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }

    // 1D
    if (input->shape().size() == 1)
    {
        std::vector<float> result{};
        for (std::size_t i = 0; i < input->shape()[0]; i++) {
            if ((*input)(i) > 0) {
                result.push_back((*input)(i));
            }
            else {
                result.push_back(0.0f);
            }
        }
        if (input-> requiresGrad()) {
            std::vector<std::shared_ptr<Tensor>> parents{input};

            std::function<void(const std::vector<float> &)> gradfn {
                [input](const std::vector<float> &grad_output)
                    {
                        std::vector<float> grad_input;
                        for (std::size_t i = 0; i < input->numel(); i++) {
                            if ((*input)(i) > 0){
                                grad_input.push_back(grad_output[i]);
                            }
                            else {
                                grad_input.push_back(0);
                            }
                        }
                        input->addToGrad(grad_input);
                    }};
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 2D
    else {
        std::vector<std::vector<float>> result {};
        for (std::size_t i = 0; i < input->shape()[0]; i++) {
            std::vector<float> result_i;
            for (std::size_t j = 0; j < input->shape()[1]; j++) {
                if ((*input)(i, j) > 0) {
                    result_i.push_back((*input)(i, j));
                }
                else {
                    result_i.push_back(0.0f);
                }
            }
            result.push_back(result_i);
        }
        if (input->requiresGrad()) {
            std::vector<std::shared_ptr<Tensor>> parents{input};

            std::function<void(const std::vector<float> &)> gradfn {
                [input](const std::vector<float> &grad_output)
                    {
                        // All grads are stored in row major order
                        std::vector<float> grad_input;
                        for (std::size_t i = 0; i < input->numel(); i++) {
                            if ((*input)(i) > 0) {
                                grad_input.push_back(grad_output[i]);
                            }
                            else {
                                grad_input.push_back(0);
                            }
                        }
                        input->addToGrad(grad_input);
                    }};
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
}
