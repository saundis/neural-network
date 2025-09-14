#include "./softmax.h"
#include "../tensors.h"
#include <memory>
#include <random>

// https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
std::shared_ptr<Tensor> Softmax::forward(std::shared_ptr<Tensor> input) {
    // Scalar
    if (input->shape().size() == 0) {
        float result = 1.0f;
        if (input->requiresGrad()) {
            std::vector<std::shared_ptr<Tensor>> parents{input};

            std::function<void(const std::vector<float> &)> gradfn =
                [input](const std::vector<float> &grad_output)
                    {
                        // For scalar grad of softmax is 0
                        std::vector<float> grad_input = {0.0f};
                        input -> addToGrad(grad_input);
                    };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 1D
    if (input->shape().size() == 1)
    {
        // Search for max (to later normalize each value to avoid overflow)
        float max_val = (*input)(0);
        for (int i = 0; i < input->numel(); i++) {
            if ((*input)(i) > max_val) {
                max_val = (*input)(i);
            }
        }
        // Softmax
        std::vector<float> s;
        float sum_exp = 0.0f;
        for (int i = 0; i < input->numel(); i++) {
            sum_exp += std::exp((*input)(i)-max_val);
        }
        for (int i = 0; i < input->numel(); i++) {
            s.push_back((std::exp((*input)(i)-max_val) / sum_exp));
        }
        if (input->requiresGrad()) {
            std::vector<std::shared_ptr<Tensor>> parents{input};
            std::function<void(const std::vector<float> &)> gradfn =
                [input, s](const std::vector<float> &grad_output) {
                        std::vector<float> grad_input;
                        // For each input value, gradient is sum of loss grad * derivative of 
                        // softmax with respect to initial value which is the derivation of softmax function
                        for (int j = 0; j < input->numel(); j++) {
                            float grad_j = 0.0f;
                            for (int i = 0; i < grad_output.size(); i++) {
                                if (i == j) { 
                                    grad_j += (grad_output[i] * (s[i] * (1 - s[i])));
                                }
                                else {
                                    grad_j += (grad_output[i] * (-s[i] * s[j]));
                                }
                            }
                            grad_input.push_back(grad_j);
                        }
                        input->addToGrad(grad_input);
                    };
            return std::make_shared<Tensor>(s, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(s);
    }
    else {
        throw std::runtime_error("Softmax is only allowed for 1d vectors.");
    }
}
