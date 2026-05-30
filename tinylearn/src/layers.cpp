#include "../include/layers.hpp"
#include <cmath>
#include <iostream>
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <vector>


static void write_tensor(std::ofstream& out, const Tensor& t) {
    out.write(reinterpret_cast<const char*>(&t.b), sizeof(int));
    out.write(reinterpret_cast<const char*>(&t.c), sizeof(int));
    out.write(reinterpret_cast<const char*>(&t.h), sizeof(int));
    out.write(reinterpret_cast<const char*>(&t.w), sizeof(int));

    if (!t.data.empty()) {
        out.write(reinterpret_cast<const char*>(t.data.data()), t.data.size() * sizeof(float));
    }
}

static void read_tensor(std::ifstream& in, Tensor& t) {
    int b, c, h, w;
    in.read(reinterpret_cast<char*>(&b), sizeof(int));
    in.read(reinterpret_cast<char*>(&c), sizeof(int));
    in.read(reinterpret_cast<char*>(&h), sizeof(int));
    in.read(reinterpret_cast<char*>(&w), sizeof(int));


    t = Tensor(b, c, h, w);

    if (!t.data.empty()) {
        in.read(reinterpret_cast<char*>(t.data.data()), t.data.size() * sizeof(float));
    }
}


Conv2D::Conv2D(int in_c, int out_c, int k_size, int stride, int padding)
: in_c(in_c), out_c(out_c), k_size(k_size), stride(stride), padding(padding) {

    weights = Tensor(out_c, in_c, k_size, k_size);
    bias = Tensor(1, out_c, 1, 1);

    float scale = std::sqrt(2.0f / (in_c * k_size * k_size));
    weights.randomize(scale);

    params = (long long)(k_size * k_size * in_c + 1) * out_c;
}

Tensor Conv2D::forward(const Tensor& input) {
    input_cache = input;
    int batch = input.b;
    int h_out = (input.h + 2 * padding - k_size) / stride + 1;
    int w_out = (input.w + 2 * padding - k_size) / stride + 1;

    Tensor output(batch, out_c, h_out, w_out);

    macs = (long long)batch * out_c * h_out * w_out * (k_size * k_size * in_c);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_c; ++oc) {
            for (int oh = 0; oh < h_out; ++oh) {
                for (int ow = 0; ow < w_out; ++ow) {

                    float sum = bias.data[oc];
                    int h_start = oh * stride - padding;
                    int w_start = ow * stride - padding;

                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int kh = 0; kh < k_size; ++kh) {
                            for (int kw = 0; kw < k_size; ++kw) {
                                sum += input.get(b, ic, h_start + kh, w_start + kw) *
                                weights(oc, ic, kh, kw);
                            }
                        }
                    }
                    output(b, oc, oh, ow) = sum;
                }
            }
        }
    }
    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    Tensor grad_input(input_cache.b, input_cache.c, input_cache.h, input_cache.w);

    weights.zero_grad();
    bias.zero_grad();

    int h_out = grad_output.h;
    int w_out = grad_output.w;

    #pragma omp parallel for
    for (int b = 0; b < input_cache.b; ++b) {
        for (int oc = 0; oc < out_c; ++oc) {
            for (int oh = 0; oh < h_out; ++oh) {
                for (int ow = 0; ow < w_out; ++ow) {
                    float g = grad_output.get(b, oc, oh, ow);

                    #pragma omp atomic
                    bias.grad[oc] += g;

                    int h_start = oh * stride - padding;
                    int w_start = ow * stride - padding;

                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int kh = 0; kh < k_size; ++kh) {
                            for (int kw = 0; kw < k_size; ++kw) {
                                int ih = h_start + kh;
                                int iw = w_start + kw;

                                if (ih >= 0 && ih < input_cache.h && iw >= 0 && iw < input_cache.w) {
                                    int w_idx = oc * (in_c * k_size * k_size) +
                                    ic * (k_size * k_size) +
                                    kh * k_size + kw;
                                    float val = input_cache(b, ic, ih, iw);

                                    #pragma omp atomic
                                    weights.grad[w_idx] += val * g;

                                    int in_idx = b * (in_c * input_cache.h * input_cache.w) +
                                    ic * (input_cache.h * input_cache.w) +
                                    ih * input_cache.w + iw;
                                    float w_val = weights(oc, ic, kh, kw);

                                    #pragma omp atomic
                                    grad_input.grad[in_idx] += w_val * g;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    grad_input.data = grad_input.grad;
    return grad_input;
}

void Conv2D::update(float lr) {
    for (size_t i = 0; i < weights.data.size(); ++i) {
        weights.data[i] -= lr * weights.grad[i];
    }
    for (size_t i = 0; i < bias.data.size(); ++i) {
        bias.data[i] -= lr * bias.grad[i];
    }
}

void Conv2D::save(std::ofstream& out) const {
    write_tensor(out, weights);
    write_tensor(out, bias);
}

void Conv2D::load(std::ifstream& in) {
    read_tensor(in, weights);
    read_tensor(in, bias);
}


MaxPool2D::MaxPool2D(int size, int stride)
: pool_size(size), stride(stride) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    input_cache = input;
    int h_out = (input.h - pool_size) / stride + 1;
    int w_out = (input.w - pool_size) / stride + 1;

    Tensor output(input.b, input.c, h_out, w_out);
    max_indices.assign(output.size(), -1);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < input.b; ++b) {
        for (int c = 0; c < input.c; ++c) {
            for (int oh = 0; oh < h_out; ++oh) {
                for (int ow = 0; ow < w_out; ++ow) {

                    int h_start = oh * stride;
                    int w_start = ow * stride;
                    float max_val = -1e9;
                    int max_idx = -1; 

                    for (int kh = 0; kh < pool_size; ++kh) {
                        for (int kw = 0; kw < pool_size; ++kw) {
                            int ih = h_start + kh;
                            int iw = w_start + kw;
                            float val = input.get(b, c, ih, iw);
                            if (val > max_val) {
                                max_val = val;
                                max_idx = ih * input.w + iw;
                            }
                        }
                    }
                    output(b, c, oh, ow) = max_val;

                    int flat_out_idx = b*(input.c*h_out*w_out) + c*(h_out*w_out) + oh*w_out + ow;
                    max_indices[flat_out_idx] = max_idx;
                }
            }
        }
    }
    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    Tensor grad_input(input_cache.b, input_cache.c, input_cache.h, input_cache.w);
    int total_elements = grad_output.size();

    #pragma omp parallel for
    for (int i = 0; i < total_elements; ++i) {
        int idx_in_hw = max_indices[i];
        if (idx_in_hw == -1) continue;

        int hw_out = grad_output.h * grad_output.w;
        int chw_out = grad_output.c * hw_out;

        int b = i / chw_out;
        int rem = i % chw_out;
        int c = rem / hw_out;

        int input_offset = b * (input_cache.c * input_cache.h * input_cache.w) +
        c * (input_cache.h * input_cache.w);

        grad_input.data[input_offset + idx_in_hw] += grad_output.data[i];
    }
    return grad_input;
}


Tensor ReLU::forward(const Tensor& input) {
    input_cache = input;
    Tensor output = input;
    for (size_t i = 0; i < output.data.size(); ++i) {
        if (output.data[i] < 0) output.data[i] = 0;
    }
    return output;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    Tensor grad_input = grad_output;
    for (size_t i = 0; i < grad_input.data.size(); ++i) {
        if (input_cache.data[i] <= 0) {
            grad_input.data[i] = 0;
        }
    }
    return grad_input;
}


Linear::Linear(int in_feat, int out_feat)
: in_features(in_feat), out_features(out_feat) {

   
    weights = Tensor(1, 1, in_features, out_features);
    bias = Tensor(1, 1, 1, out_features);

    weights.randomize(std::sqrt(2.0f / in_features));

    params = (long long)(in_features + 1) * out_features;
}

Tensor Linear::forward(const Tensor& input) {
    input_cache = input; 
    int batch = input.b;

    Tensor output(batch, 1, 1, out_features);


    macs = (long long)batch * in_features * out_features;

    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        for (int o = 0; o < out_features; ++o) {
            float sum = bias.data[o];
            for (int i = 0; i < in_features; ++i) {
                sum += input.data[b * in_features + i] * weights.data[i * out_features + o];
            }
            output.data[b * out_features + o] = sum;
        }
    }
    return output;
}

Tensor Linear::backward(const Tensor& grad_output) {
    int batch = input_cache.b;
    Tensor grad_input(batch, 1, 1, in_features);

    weights.zero_grad();
    bias.zero_grad();

    for (int b = 0; b < batch; ++b) {
        for (int o = 0; o < out_features; ++o) {
            bias.grad[o] += grad_output.data[b * out_features + o];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < in_features; ++i) {
        for (int o = 0; o < out_features; ++o) {
            float grad_w = 0;
            for (int b = 0; b < batch; ++b) {
                grad_w += input_cache.data[b * in_features + i] *
                grad_output.data[b * out_features + o];
            }
            #pragma omp atomic
            weights.grad[i * out_features + o] += grad_w;
        }
    }

    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < in_features; ++i) {
            float sum = 0;
            for (int o = 0; o < out_features; ++o) {
                sum += grad_output.data[b * out_features + o] *
                weights.data[i * out_features + o];
            }
            grad_input.data[b * in_features + i] = sum;
        }
    }
    return grad_input;
}

void Linear::update(float lr) {
    for (size_t i = 0; i < weights.data.size(); ++i) {
        weights.data[i] -= lr * weights.grad[i];
    }
    for (size_t i = 0; i < bias.data.size(); ++i) {
        bias.data[i] -= lr * bias.grad[i];
    }
}

void Linear::save(std::ofstream& out) const {
    write_tensor(out, weights);
    write_tensor(out, bias);
}

void Linear::load(std::ifstream& in) {
    read_tensor(in, weights);
    read_tensor(in, bias);
}


float CrossEntropyLoss::forward(const Tensor& logits, const Tensor& targets) {
    preds = logits;
    int batch = logits.b;
    int num_classes = logits.w; 

    float total_loss = 0;

    for (int b = 0; b < batch; ++b) {

        float max_val = -1e9;
        for(int i = 0; i < num_classes; ++i) {
            if(logits.data[b*num_classes + i] > max_val)
                max_val = logits.data[b*num_classes + i];
        }


        float sum_exp = 0;
        for(int i = 0; i < num_classes; ++i) {
            preds.data[b*num_classes + i] = std::exp(logits.data[b*num_classes + i] - max_val);
            sum_exp += preds.data[b*num_classes + i];
        }

        int label = (int)targets.data[b];

        for(int i = 0; i < num_classes; ++i) {
            preds.data[b*num_classes + i] /= sum_exp;
            if (i == label) {
                total_loss -= std::log(preds.data[b*num_classes + i] + 1e-7f);
            }
        }
    }
    return total_loss / batch;
}

Tensor CrossEntropyLoss::backward(const Tensor& targets) {
    Tensor grad = preds; 
    int batch = preds.b;
    int num_classes = preds.w;

    for (int b = 0; b < batch; ++b) {
        int label = (int)targets.data[b];
        grad.data[b * num_classes + label] -= 1.0f;
    }

    for(auto& v : grad.data) v /= batch;

    return grad;
}
