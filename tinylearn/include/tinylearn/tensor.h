#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <cassert>

struct Tensor
{
    int C, H, W;
    std::vector<float> data;
    std::vector<float> grad; 

    Tensor() : C(0), H(0), W(0) {}
    Tensor(int c, int h, int w) : C(c), H(h), W(w), data(c * h * w), grad(c * h * w, 0.0f) {}

    void resize(int c, int h, int w) {
        C = c; H = h; W = w;
        data.assign(c * h * w, 0.0f);
        grad.assign(c * h * w, 0.0f);
    }

    float &operator()(int c, int h, int w)
    {
        assert(c >= 0 && c < C);
        assert(h >= 0 && h < H);
        assert(w >= 0 && w < W);
        return data[c * H * W + h * W + w];
    }

    float &grad_at(int c, int h, int w) {
        return grad[c * H * W + h * W + w];
    }

    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
};

struct Tensor1D
{
    int size;
    std::vector<float> data;
    std::vector<float> grad;

    Tensor1D() : size(0) {}
    Tensor1D(int s) : size(s), data(s, 0.0f), grad(s, 0.0f) {}

    void resize(int s) {
        size = s;
        data.assign(s, 0.0f);
        grad.assign(s, 0.0f);
    }

    float &operator()(int i)
    {
        assert(i >= 0 && i < size);
        return data[i];
    }

    float &grad_at(int i) {
        return grad[i];
    }

    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
};

std::ostream& operator<<(std::ostream& os, const Tensor1D& t) {
    os << "[";
    for (size_t i = 0; i < t.data.size(); ++i) {
        os << t.data[i];
        if (i + 1 < t.data.size()) os << ", ";
    }
    os << "]";
    return os;
}

#endif