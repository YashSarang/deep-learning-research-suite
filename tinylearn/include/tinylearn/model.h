#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream> 
#include <algorithm>
#include <cassert>
#include <random>
#include <omp.h>
#include "tensor.h"
#include "utils.h"

inline void kaiming_normal(std::vector<float> &weights, int fan_in, unsigned int seed = 0) {
    float std = std::sqrt(2.0f / static_cast<float>(fan_in));
    std::mt19937 rng;
    if (seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(seed);
    }
    std::normal_distribution<float> dist(0.0f, std);
    for (auto &w : weights) w = dist(rng);
}

class Conv2D
{
public:
    int in_channels, out_channels;
    int kernel;
    int stride, pad;
    int H_out, W_out;

    std::vector<float> weights; 
    std::vector<float> bias;    

    std::vector<float> grad_w;
    std::vector<float> grad_b;

    Tensor last_input;

    Conv2D(int in_c, int out_c, int k, int s = 1, int p = 0) :
        in_channels(in_c), out_channels(out_c), kernel(k), stride(s), pad(p),
        weights(out_c * in_c * k * k), bias(out_c, 0.0f),
        grad_w(out_c * in_c * k * k, 0.0f), grad_b(out_c, 0.0f)
    {
        int fan_in = in_channels * kernel * kernel;
        kaiming_normal(weights, fan_in, 1234u);
    }

    Tensor forward(Tensor &input) {
        last_input = input; 
        H_out = (input.H - kernel + 2 * pad) / stride + 1;
        W_out = (input.W - kernel + 2 * pad) / stride + 1;
        Tensor output(out_channels, H_out, W_out);

        #pragma omp parallel for collapse(3)
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = bias[oc];
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel; ++kh) {
                            for (int kw = 0; kw < kernel; ++kw) {
                                int in_h = h * stride + kh - pad;
                                int in_w = w * stride + kw - pad;
                                if (in_h >= 0 && in_h < input.H && in_w >= 0 && in_w < input.W) {
                                    int idx = oc * in_channels * kernel * kernel + ic * kernel * kernel + kh * kernel + kw;
                                    sum += weights[idx] * input(ic, in_h, in_w);
                                }
                            }
                        }
                    }
                    output(oc, h, w) = sum;
                }
            }
        }
        return output;
    }

    void zero_grad() {
        std::fill(grad_w.begin(), grad_w.end(), 0.0f);
        std::fill(grad_b.begin(), grad_b.end(), 0.0f);
    }

    void backward(const Tensor &d_out) {

         #pragma omp parallel for
        for (int oc = 0; oc < out_channels; ++oc) {
            float gb = 0.0f;
            for (int h = 0; h < d_out.H; ++h)
                for (int w = 0; w < d_out.W; ++w)
                    gb += d_out.data[oc * d_out.H * d_out.W + h * d_out.W + w];
            #pragma omp atomic
            grad_b[oc] += gb;
        }

        int Wsize = static_cast<int>(grad_w.size());
        int nthreads = omp_get_max_threads();
        std::vector<std::vector<float>> local_grad_w(nthreads, std::vector<float>(Wsize, 0.0f));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &gw_local = local_grad_w[tid];

            #pragma omp for collapse(4)
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel; ++kh) {
                        for (int kw = 0; kw < kernel; ++kw) {
                            int w_idx = oc * in_channels * kernel * kernel + ic * kernel * kernel + kh * kernel + kw;
                            float gw = 0.0f;
                            for (int h = 0; h < d_out.H; ++h) {
                                for (int w = 0; w < d_out.W; ++w) {
                                    int in_h = h * stride + kh - pad;
                                    int in_w = w * stride + kw - pad;
                                    if (in_h >= 0 && in_h < last_input.H && in_w >= 0 && in_w < last_input.W) {
                                        float inp = last_input.data[ic * last_input.H * last_input.W + in_h * last_input.W + in_w];
                                        float dout = d_out.data[oc * d_out.H * d_out.W + h * d_out.W + w];
                                        gw += dout * inp;
                                    }
                                }
                            }
                            gw_local[w_idx] += gw;
                        }
                    }
                }
            }
        }

        for (int t = 0; t < nthreads; ++t) {
            for (int i = 0; i < Wsize; ++i) {
                grad_w[i] += local_grad_w[t][i];
            }
        }

        if (last_input.grad.empty()) last_input.grad.assign(last_input.data.size(), 0.0f);

        #pragma omp parallel for collapse(3)
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int h = 0; h < last_input.H; ++h) {
                for (int w = 0; w < last_input.W; ++w) {
                    float g = 0.0f;
                    for (int oc = 0; oc < out_channels; ++oc) {
                        for (int kh = 0; kh < kernel; ++kh) {
                            for (int kw = 0; kw < kernel; ++kw) {
                                int out_h = h - kh + pad;
                                int out_w = w - kw + pad;
                                if (out_h % stride == 0 && out_w % stride == 0) {
                                    out_h /= stride; out_w /= stride;
                                    if (out_h >= 0 && out_h < d_out.H && out_w >= 0 && out_w < d_out.W) {
                                        int w_idx = oc * in_channels * kernel * kernel + ic * kernel * kernel + kh * kernel + kw;
                                        float wval = weights[w_idx];
                                        float dout = d_out.data[oc * d_out.H * d_out.W + out_h * d_out.W + out_w];
                                        g += wval * dout;
                                    }
                                }
                            }
                        }
                    }
                    #pragma omp atomic
                    last_input.grad[ic * last_input.H * last_input.W + h * last_input.W + w] += g;
                }
            }
        }
    }

    void save(std::ofstream& ofs) const {

        ofs.write((char*)&in_channels, sizeof(int));
        ofs.write((char*)&out_channels, sizeof(int));
        ofs.write((char*)&kernel, sizeof(int));

        ofs.write((char*)weights.data(), weights.size() * sizeof(float));
        ofs.write((char*)bias.data(), bias.size() * sizeof(float));
    }

    void load(std::ifstream& ifs) {
        int in_c, out_c, k;
        ifs.read((char*)&in_c, sizeof(int));
        ifs.read((char*)&out_c, sizeof(int));
        ifs.read((char*)&k, sizeof(int));

        if(in_c != in_channels || out_c != out_channels || k != kernel) {
            std::cerr << "Error: Conv2D dimension mismatch in binary file." << std::endl;
            exit(1);
        }

        ifs.read((char*)weights.data(), weights.size() * sizeof(float));
        ifs.read((char*)bias.data(), bias.size() * sizeof(float));
    }

    int num_params() const { return static_cast<int>(weights.size() + bias.size()); }
};

class ReLU
{
public:
    Tensor last_input;

    Tensor forward(Tensor &x)
    {
        last_input = x;
        Tensor out(x.C, x.H, x.W);
        #pragma omp parallel for collapse(3)
        for (int c = 0; c < x.C; ++c)
            for (int h = 0; h < x.H; ++h)
                for (int w = 0; w < x.W; ++w)
                    out(c, h, w) = std::max(0.0f, x(c, h, w));
        return out;
    }

    void backward(const Tensor &d_out) {
        if (last_input.grad.empty()) last_input.grad.assign(last_input.data.size(), 0.0f);
        #pragma omp parallel for collapse(3)
        for (int c = 0; c < last_input.C; ++c)
            for (int h = 0; h < last_input.H; ++h)
                for (int w = 0; w < last_input.W; ++w) {
                    float grad = (last_input.data[c * last_input.H * last_input.W + h * last_input.W + w] > 0.0f) ?
                        d_out.data[c * d_out.H * d_out.W + h * d_out.W + w] : 0.0f;
                    #pragma omp atomic
                    last_input.grad[c * last_input.H * last_input.W + h * last_input.W + w] += grad;
                }
    }
};

class MaxPool2D
{
public:
    std::vector<int> last_argmax; 
    int last_H_out, last_W_out, last_C;

    Tensor forward(Tensor &x)
    {
        int H_out = x.H / 2;
        int W_out = x.W / 2;
        last_H_out = H_out; last_W_out = W_out; last_C = x.C;
        Tensor out(x.C, H_out, W_out);
        last_argmax.assign(x.C * H_out * W_out, -1);

        #pragma omp parallel for collapse(3)
        for (int c = 0; c < x.C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float m = -1e9;
                    int arg = -1;
                    for (int dh = 0; dh < 2; ++dh) {
                        for (int dw = 0; dw < 2; ++dw) {
                            int ih = 2 * h + dh;
                            int iw = 2 * w + dw;
                            float val = x.data[c * x.H * x.W + ih * x.W + iw];
                            if (val > m) { m = val; arg = ih * x.W + iw; }
                        }
                    }
                    out(c, h, w) = m;
                    last_argmax[c * H_out * W_out + h * W_out + w] = arg;
                }
            }
        }
        return out;
    }

    void backward(const Tensor &d_out, Tensor &input) {
        if (input.grad.empty()) input.grad.assign(input.data.size(), 0.0f);
        int H_out = last_H_out, W_out = last_W_out;
        #pragma omp parallel for collapse(3)
        for (int c = 0; c < input.C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    int arg = last_argmax[c * H_out * W_out + h * W_out + w];
                    #pragma omp atomic
                    input.grad[c * input.H * input.W + arg] += d_out.data[c * H_out * W_out + h * W_out + w];
                }
            }
        }
    }
};

class Linear
{
public:
    int in_features, out_features;
    std::vector<float> W; 
    std::vector<float> b;

    std::vector<float> grad_W;
    std::vector<float> grad_b;
    Tensor1D last_input;

    Linear() : in_features(0), out_features(0) {}
    Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f),
        W(out_f * in_f), b(out_f, 0.0f),
        grad_W(out_f * in_f, 0.0f), grad_b(out_f, 0.0f)
    {
        int fan_in = in_features;
        kaiming_normal(W, fan_in, 1234u);
    }

    Tensor1D forward(const Tensor1D &x)
    {
        last_input = x;
        Tensor1D out(out_features);
        for (int o = 0; o < out_features; ++o) {
            float sum = b[o];
            for (int i = 0; i < in_features; ++i) {
                sum += W[o * in_features + i] * x.data[i];
            }
            out(o) = sum;
        }
        return out;
    }

    void zero_grad() {
        std::fill(grad_W.begin(), grad_W.end(), 0.0f);
        std::fill(grad_b.begin(), grad_b.end(), 0.0f);
    }

    Tensor1D backward(const Tensor1D &d_out) {
        Tensor1D d_input(in_features);
        for (int o = 0; o < out_features; ++o) {
            grad_b[o] += d_out.data[o];
            for (int i = 0; i < in_features; ++i) {
                grad_W[o * in_features + i] += d_out.data[o] * last_input.data[i];
                d_input.data[i] += W[o * in_features + i] * d_out.data[o];
            }
        }
        return d_input;
    }

    void save(std::ofstream& ofs) const {
        ofs.write((char*)&in_features, sizeof(int));
        ofs.write((char*)&out_features, sizeof(int));
        ofs.write((char*)W.data(), W.size() * sizeof(float));
        ofs.write((char*)b.data(), b.size() * sizeof(float));
    }

    void load(std::ifstream& ifs) {
        int in_f, out_f;
        ifs.read((char*)&in_f, sizeof(int));
        ifs.read((char*)&out_f, sizeof(int));

        if (in_f != in_features || out_f != out_features) {
            in_features = in_f;
            out_features = out_f;
            W.resize(out_f * in_f);
            b.resize(out_f);
            grad_W.resize(out_f * in_f, 0.0f);
            grad_b.resize(out_f, 0.0f);
        }

        ifs.read((char*)W.data(), W.size() * sizeof(float));
        ifs.read((char*)b.data(), b.size() * sizeof(float));
    }

    int num_params() const { return static_cast<int>(W.size() + b.size()); }
};

float cross_entropy_loss_and_grad(Tensor1D &logits, int label, Tensor1D &grad_logits)
{
    float maxv = logits.data[0];
    for (int i = 1; i < logits.size; ++i) if (logits.data[i] > maxv) maxv = logits.data[i];
    float sum = 0.0f;
    for (int i = 0; i < logits.size; ++i) {
        sum += std::exp(logits.data[i] - maxv);
    }
    float loss = - (logits.data[label] - maxv) + std::log(sum);

    grad_logits.resize(logits.size);
    for (int i = 0; i < logits.size; ++i) {
        float p = std::exp(logits.data[i] - maxv) / sum;
        grad_logits.data[i] = p - (i == label ? 1.0f : 0.0f);
    }
    return loss;
}

class SimpleCNN
{
public:
    Conv2D conv1;
    ReLU relu1;
    Conv2D conv2;
    ReLU relu2;
    MaxPool2D pool;
    Linear *fc1;     
    ReLU relu_fc;
    Linear fc2;

    SimpleCNN(int num_classes)
        : conv1(3, 8, 3),
          conv2(8, 8, 3),
          fc1(nullptr),
          fc2(64, num_classes)
    {}

    ~SimpleCNN() {
        if (fc1) delete fc1;
    }

    Tensor1D forward(Tensor &x)
    {
        Tensor y = conv1.forward(x);
        y = relu1.forward(y);
        y = conv2.forward(y);
        y = relu2.forward(y);
        y = pool.forward(y);

        Tensor1D flat(y.C * y.H * y.W);
        int idx = 0;
        for (int c = 0; c < y.C; ++c)
            for (int h = 0; h < y.H; ++h)
                for (int w = 0; w < y.W; ++w)
                    flat.data[idx++] = y.data[c * y.H * y.W + h * y.W + w];

        if (!fc1) {
            fc1 = new Linear(flat.size, 64);
        }

        Tensor1D out1 = fc1->forward(flat);
        Tensor1D out_relu(64);
        for (int i = 0; i < 64; ++i) out_relu.data[i] = std::max(0.0f, out1.data[i]);
        return fc2.forward(out_relu);
    }

    void backward(Tensor1D &logits_grad) {

        Tensor1D d_fc2_in = fc2.backward(logits_grad); 

        Tensor1D preact_fc1(fc1->out_features);
        for (int o = 0; o < fc1->out_features; ++o) {
            float s = fc1->b[o];
            for (int i = 0; i < fc1->in_features; ++i)
                s += fc1->W[o * fc1->in_features + i] * fc1->last_input.data[i];
            preact_fc1.data[o] = s;
        }

        Tensor1D d_after_relu(fc1->out_features);
        for (int i = 0; i < fc1->out_features; ++i) {
            d_after_relu.data[i] = (preact_fc1.data[i] > 0.0f) ? d_fc2_in.data[i] : 0.0f;
        }

        Tensor1D d_flat = fc1->backward(d_after_relu); 

        int C = conv2.out_channels;
        int H_after_conv2 = conv2.last_input.H - conv2.kernel + 1; 
        int W_after_conv2 = conv2.last_input.W - conv2.kernel + 1;
        int H_pool = H_after_conv2 / 2;
        int W_pool = W_after_conv2 / 2;

        Tensor d_pool(C, H_pool, W_pool);
        int idx = 0;
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H_pool; ++h)
                for (int w = 0; w < W_pool; ++w)
                    d_pool.data[c * H_pool * W_pool + h * W_pool + w] = d_flat.data[idx++];

        Tensor conv2_out(C, H_after_conv2, W_after_conv2);
        conv2_out.grad.assign(conv2_out.data.size(), 0.0f);
        pool.backward(d_pool, conv2_out); 

        relu2.last_input = conv2_out; 
        relu2.backward(conv2_out); 

        Tensor d_conv2_out(C, H_after_conv2, W_after_conv2);
        d_conv2_out.data = conv2.last_input.grad; 
        conv2.backward(d_conv2_out);

        Tensor relu1_out = conv2.last_input; 
        Tensor d_relu1_out(relu1_out.C, relu1_out.H, relu1_out.W);
        d_relu1_out.data = conv2.last_input.grad;
        relu1.last_input = relu1_out;
        relu1.backward(d_relu1_out);

        Tensor d_conv1_out(relu1.last_input.C, relu1.last_input.H, relu1.last_input.W);
        d_conv1_out.data = relu1.last_input.grad;
        conv1.backward(d_conv1_out);
    }

    void zero_grad() {
        conv1.zero_grad();
        conv2.zero_grad();
        if (fc1) fc1->zero_grad();
        fc2.zero_grad();
    }

    std::vector<std::pair<std::vector<float>*, std::vector<float>*>> get_param_grad_pairs() {
        std::vector<std::pair<std::vector<float>*, std::vector<float>*>> out;
        out.push_back({ &conv1.weights, &conv1.grad_w });
        out.push_back({ &conv1.bias, &conv1.grad_b });
        out.push_back({ &conv2.weights, &conv2.grad_w });
        out.push_back({ &conv2.bias, &conv2.grad_b });
        if (fc1) {
            out.push_back({ &fc1->W, &fc1->grad_W });
            out.push_back({ &fc1->b, &fc1->grad_b });
        }
        out.push_back({ &fc2.W, &fc2.grad_W });
        out.push_back({ &fc2.b, &fc2.grad_b });
        return out;
    }

    void save_parameters(const std::string& filename) {
        if (!fc1) {
            std::cerr << "Error: Cannot save model before initialization (run at least one forward pass to shape layers)." << std::endl;
            return;
        }
        std::ofstream ofs(filename, std::ios::binary);
        if(!ofs) {
            std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
            return;
        }

        conv1.save(ofs);
        conv2.save(ofs);
        fc1->save(ofs);
        fc2.save(ofs);

        ofs.close();
        std::cout << "Model parameters saved to " << filename << std::endl;
    }

    void save_model(const std::string& filename) { save_parameters(filename); }
    void load_model(const std::string& filename) { load_parameters(filename); }

    void load_parameters(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if(!ifs) {
            std::cerr << "Error: Could not open " << filename << " for reading." << std::endl;
            return;
        }

        conv1.load(ifs);
        conv2.load(ifs);

        int fc1_in, fc1_out;
        auto pos = ifs.tellg(); 
        ifs.read((char*)&fc1_in, sizeof(int));
        ifs.read((char*)&fc1_out, sizeof(int));
        ifs.seekg(pos); 

        if (fc1) delete fc1;
        fc1 = new Linear(fc1_in, fc1_out);
        fc1->load(ifs); 

        fc2.load(ifs);

        ifs.close();
        std::cout << "Model parameters loaded from " << filename << std::endl;
    }
};

#endif