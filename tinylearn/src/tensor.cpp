#include "../include/tensor.hpp"
#include <random>
#include <algorithm>

Tensor::Tensor() : b(0), c(0), h(0), w(0) {}

Tensor::Tensor(int b, int c, int h, int w, float init_val)
: b(b), c(c), h(h), w(w) {
    data.resize(b * c * h * w, init_val);
    grad.resize(b * c * h * w, 0.0f);
}

float Tensor::get(int ib, int ic, int ih, int iw) const {
    if (ih < 0 || ih >= h || iw < 0 || iw >= w) return 0.0f;
    return data[ib * (c * h * w) + ic * (h * w) + ih * w + iw];
}

float& Tensor::operator()(int ib, int ic, int ih, int iw) {
    return data[ib * (c * h * w) + ic * (h * w) + ih * w + iw];
}

void Tensor::randomize(float scale) {
    std::mt19937 gen(1234);
    std::normal_distribution<float> d(0, scale);
    for (auto& val : data) val = d(gen);
}

void Tensor::zero_grad() {
    std::fill(grad.begin(), grad.end(), 0.0f);
}

int Tensor::size() const {
    return b * c * h * w;
}

void Tensor::flatten() {
    // Used before Fully Connected layers
    int flat_dim = c * h * w;
    c = 1; h = 1; w = flat_dim;
}
