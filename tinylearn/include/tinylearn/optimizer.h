#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <cmath>
#include <iostream>
#include "tensor.h"

class SGD {
public:
    float lr;
    float momentum;
    float weight_decay; 
    std::vector<std::vector<float>> velocity; 

    SGD(float lr_=1e-2f, float momentum_=0.9f, float weight_decay_=5e-4f)
        : lr(lr_), momentum(momentum_), weight_decay(weight_decay_) {}

    void init(const std::vector<std::pair<std::vector<float>*, std::vector<float>*>> &params) {
        velocity.clear();
        velocity.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            velocity[i].assign(params[i].first->size(), 0.0f);
        }
    }

    void step(const std::vector<std::pair<std::vector<float>*, std::vector<float>*>> &params) {
        for (size_t i = 0; i < params.size(); ++i) {
            auto &p = *params[i].first;
            auto &g = *params[i].second;
            auto &v = velocity[i];
            for (size_t j = 0; j < p.size(); ++j) {

                float grad = g[j] + weight_decay * p[j];

                v[j] = momentum * v[j] + (1.0f - momentum) * grad;

                p[j] -= lr * v[j];

                g[j] = 0.0f;
            }
        }
    }
};

#endif