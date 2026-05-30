#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor.hpp"
#include <string>
#include <vector>
#include <fstream> 


class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void update(float lr) {};

    virtual long long get_params() const { return 0; }
    virtual long long get_macs() const { return 0; }
    virtual std::string get_name() const = 0;

    virtual void save(std::ofstream& out) const {}
    virtual void load(std::ifstream& in) {}

    virtual ~Layer() = default;
};

class Conv2D : public Layer {
    Tensor weights, bias;
    Tensor input_cache;
    int in_c, out_c, k_size, stride, padding;
    long long params = 0, macs = 0;

public:
    Conv2D(int in_c, int out_c, int k_size, int stride=1, int padding=1);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update(float lr) override;

    long long get_params() const override { return params; }
    long long get_macs() const override { return macs; }
    std::string get_name() const override { return "Conv2D"; }

    void save(std::ofstream& out) const override;
    void load(std::ifstream& in) override;
};

class MaxPool2D : public Layer {
    int pool_size, stride;
    Tensor input_cache;
    std::vector<int> max_indices;

public:
    MaxPool2D(int size=2, int stride=2);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    std::string get_name() const override { return "MaxPool"; }
};

class ReLU : public Layer {
    Tensor input_cache;
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    std::string get_name() const override { return "ReLU"; }
};

class Linear : public Layer {
    Tensor weights, bias, input_cache;
    int in_features, out_features;
    long long params = 0, macs = 0;

public:
    Linear(int in_feat, int out_feat);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update(float lr) override;

    long long get_params() const override { return params; }
    long long get_macs() const override { return macs; }
    std::string get_name() const override { return "Linear"; }

    void save(std::ofstream& out) const override;
    void load(std::ifstream& in) override;
};

class CrossEntropyLoss {
    Tensor preds;
public:
    float forward(const Tensor& logits, const Tensor& targets);
    Tensor backward(const Tensor& targets);
};

#endif
