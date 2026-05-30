#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <iostream>

struct Tensor {
    std::vector<float> data;
    std::vector<float> grad;
    int b, c, h, w; 

    Tensor();
    Tensor(int b, int c, int h, int w, float init_val = 0.0f);

    float get(int ib, int ic, int ih, int iw) const;

    float& operator()(int ib, int ic, int ih, int iw);

    void randomize(float scale);
    void zero_grad();
    int size() const;
    void flatten(); 
};

#endif
