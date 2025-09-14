#include "./module.h"
#include "../tensors.h"
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

std::shared_ptr<Tensor> Module::forward(std::shared_ptr<Tensor> input) {
    throw std::runtime_error("Forward not implemented");
}

// Registers modules into stored parameters given a name and pointer
void Module::register_parameter(std::string name, std::shared_ptr<Tensor> param) {
    for (const auto& p : m_parameters) {
        if (p.first == name) {
            throw std::runtime_error("Parameter '" + name + "' already registered");
        }
    }

    m_parameters.push_back({name, param});
}

// Registers modules into stored modules given a name and pointer
void Module::register_module(std::string name, std::shared_ptr<Module> module) {
    for (const auto& m: m_modules) {
        if (m.first == name) {
            throw std::runtime_error("Module '" + name + "' already registered");
        }
    }
    m_modules.push_back({name, module});
}

// Collects all parameters of module and children modules into vector and returns them
std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> Module::parameters() const {
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> params{};
    for (const auto& p : m_parameters) {
        params.push_back(p);
    }
    for (const auto& m : m_modules) {
        for (const auto& p : m.second -> parameters()) {
            std::string full_name{ m.first.empty() ? p.first : m.first + '.' + p.first };
            params.push_back({full_name, p.second});
        }
    }
    return params;
}

// Returns dictionary of parameters to be saved
std::unordered_map<std::string, std::shared_ptr<Tensor>> Module::state_dict() const {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict{};

    for (const auto& p : parameters()) {
        state_dict[p.first] = p.second;
    }
    return state_dict;
}

// Loads dictionary of parameters into module 
void Module::load_state_dict(std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict) {
    for (const auto& p : parameters()) {
        auto it{ state_dict.find(p.first) };
        if (it == state_dict.end()) {
            std::cerr << "Warning: Paramters '" << p.first << "' not found in state_dict"
                      << '\n';
            continue;
        }
        std::shared_ptr<Tensor> stored_param{ it -> second };
        if (p.second -> shape() != stored_param -> shape()) {
            throw std::runtime_error("Parameter '" + p.first + "' has different shape in state_dict");
        }
        p.second -> data() = stored_param -> data();
    }
}
