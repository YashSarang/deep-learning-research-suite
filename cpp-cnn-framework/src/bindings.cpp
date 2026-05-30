#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/layers.hpp"
#include "../include/dataset.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <utility> // for std::pair

namespace py = pybind11;

// Wrapper for Dataset
class PyDataset {
    Dataset ds;
public:
    void load(const std::string& path) {
        ds.load_structure(path);
    }

    int size() {
        return (int)ds.labels.size();
    }

    // Returns a tuple: (flat_input_vector, flat_label_vector)
    std::pair<std::vector<float>, std::vector<float>> get_batch_data(const std::vector<int>& indices) {
        Tensor X, Y;
        ds.get_batch(indices, X, Y);
        return {X.data, Y.data};
    }
};

// Wrapper for Model
class CppModel {
    std::vector<Layer*> layers;
    CrossEntropyLoss criterion;

public:
    CppModel() {}

    ~CppModel() {
        for (auto l : layers) delete l;
    }

    void add_conv2d(int in_c, int out_c, int k_size, int stride, int padding) {
        layers.push_back(new Conv2D(in_c, out_c, k_size, stride, padding));
    }

    void add_maxpool2d(int size, int stride) {
        layers.push_back(new MaxPool2D(size, stride));
    }

    void add_relu() {
        layers.push_back(new ReLU());
    }

    void add_linear(int in_feat, int out_feat) {
        layers.push_back(new Linear(in_feat, out_feat));
    }

    std::vector<float> forward(const std::vector<float>& input_data, int b, int c, int h, int w) {
        Tensor x(b, c, h, w);
        x.data = input_data;

        for (auto l : layers) {
            x = l->forward(x);
            if (dynamic_cast<Linear*>(l) && x.c > 1) {
                x.flatten();
            }
        }
        return x.data;
    }

    float train_step(const std::vector<float>& input_data, const std::vector<float>& target_data,
                     int b, int c, int h, int w, float lr) {
        Tensor x(b, c, h, w);
        x.data = input_data;
        Tensor y(b, 1, 1, 1);
        y.data = target_data;

        for (auto l : layers) {
            x = l->forward(x);
            if (dynamic_cast<Linear*>(l) && x.c > 1) x.flatten();
        }

        float loss = criterion.forward(x, y);

        Tensor grad = criterion.backward(y);
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = (*it)->backward(grad);
        }

        for (auto l : layers) {
            l->update(lr);
        }
        return loss;
                     }

                     // NEW: Calculates Params and MACs (per single input)
                     std::pair<long long, long long> get_complexity_info(int c, int h, int w) {
                         // Run a dummy forward pass with Batch Size = 1
                         Tensor x(1, c, h, w);
                         // Fill with zeros just to be safe
                         std::fill(x.data.begin(), x.data.end(), 0.0f);

                         long long total_params = 0;
                         long long total_macs = 0;

                         for (auto l : layers) {
                             x = l->forward(x); // This calculates MACs for batch=1 inside the layer
                             if (dynamic_cast<Linear*>(l) && x.c > 1) {
                                 x.flatten();
                             }
                             total_params += l->get_params();
                             total_macs += l->get_macs();
                         }

                         return {total_params, total_macs};
                     }

                     void save_weights(const std::string& filename) {
                         std::ofstream out(filename, std::ios::binary);
                         if (!out.is_open()) throw std::runtime_error("Cannot open file for saving");
                         int size = (int)layers.size();
                         out.write((char*)&size, sizeof(int));
                         for (auto l : layers) l->save(out);
                         out.close();
                     }

                     void load_weights(const std::string& filename) {
                         std::ifstream in(filename, std::ios::binary);
                         if (!in.is_open()) throw std::runtime_error("Cannot open file for loading");
                         int size;
                         in.read((char*)&size, sizeof(int));
                         if ((size_t)size != layers.size()) throw std::runtime_error("Arch mismatch");
                         for (auto l : layers) l->load(in);
                         in.close();
                     }
};

PYBIND11_MODULE(deepcpp, m) {
    py::class_<PyDataset>(m, "PyDataset")
    .def(py::init<>())
    .def("load", &PyDataset::load)
    .def("size", &PyDataset::size)
    .def("get_batch", &PyDataset::get_batch_data);

    py::class_<CppModel>(m, "CppModel")
    .def(py::init<>())
    .def("add_conv2d", &CppModel::add_conv2d)
    .def("add_maxpool2d", &CppModel::add_maxpool2d)
    .def("add_relu", &CppModel::add_relu)
    .def("add_linear", &CppModel::add_linear)
    .def("forward", &CppModel::forward)
    .def("train_step", &CppModel::train_step)
    .def("get_complexity_info", &CppModel::get_complexity_info) // Added Binding
    .def("save_weights", &CppModel::save_weights)
    .def("load_weights", &CppModel::load_weights);
}
