#ifndef UTILS_H
#define UTILS_H

#include <vector>

#include <algorithm>
#include "tensor.h"

inline int num_params(const std::vector<float>& weights,
    const std::vector<float>& bias) {

    return static_cast<int>(weights.size() + bias.size());
}

inline long macs(int out_channels, int in_channels, int kernel, int stride,
    int pad, int H, int W) {

    int H_out = (H - kernel + 2 * pad) / stride + 1;
    int W_out = (W - kernel + 2 * pad) / stride + 1;

    return static_cast<long>(out_channels) * H_out * W_out * in_channels
        * kernel * kernel;
}

inline long macs_linear(int in_features, int out_features, int stride,
    int pad) {

    int effective_in = in_features + 2 * pad;
    return static_cast<long>(effective_in / stride) * out_features;
}

inline int print_argmax(const Tensor1D &tensor) {
    if (tensor.data.empty()) return -1;
    auto max_it = std::max_element(tensor.data.begin(), tensor.data.end());
    return std::distance(tensor.data.begin(), max_it);
}

#endif