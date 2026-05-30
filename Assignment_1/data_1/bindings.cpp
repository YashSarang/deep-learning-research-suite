#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor.h"
#include "model.h"
#include "data_loader.h"
#include "optimizer.h"
#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(assignment_1, m) {
    m.doc() = "Assignment 1 Deep Learning Framework Bindings";

    py::class_<Tensor1D>(m, "Tensor1D")
        .def(py::init<int>())
        .def_readwrite("data", &Tensor1D::data)
        .def_readwrite("size", &Tensor1D::size)
        .def("numpy", [](Tensor1D &t) {

            return py::array_t<float>(
                {t.size},
                {sizeof(float)},
                t.data.data(),
                py::cast(t)
            );
        });

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<int, int, int>())
        .def_readwrite("C", &Tensor::C)
        .def_readwrite("H", &Tensor::H)
        .def_readwrite("W", &Tensor::W)
        .def_readwrite("data", &Tensor::data);

    py::class_<Sample>(m, "Sample")
        .def_readwrite("image", &Sample::image)
        .def_readwrite("label", &Sample::label);

    py::class_<ImageFolderDataset>(m, "ImageFolderDataset")
        .def(py::init<std::string>())
        .def_readonly("samples", &ImageFolderDataset::samples);

    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<ImageFolderDataset&, size_t, bool>(), 
             py::keep_alive<1, 2>()) 
        .def("has_next", &DataLoader::has_next)
        .def("next_batch", &DataLoader::next_batch, py::return_value_policy::reference_internal)
        .def("reset", &DataLoader::reset);

    py::class_<SimpleCNN>(m, "SimpleCNN")
        .def(py::init<int>())
        .def("forward", &SimpleCNN::forward)
        .def("backward", &SimpleCNN::backward)
        .def("zero_grad", &SimpleCNN::zero_grad)
        .def("save_model", &SimpleCNN::save_model)
        .def("load_model", &SimpleCNN::load_model)

        .def("get_param_grad_pairs", &SimpleCNN::get_param_grad_pairs);

    py::class_<SGD>(m, "SGD")
        .def(py::init<float, float, float>(), 
             py::arg("lr")=1e-2f, py::arg("momentum")=0.9f, py::arg("weight_decay")=5e-4f)
        .def("step_model", [](SGD &opt, SimpleCNN &model) {
            auto params = model.get_param_grad_pairs();
            if (opt.velocity.size() != params.size()) opt.init(params);
            opt.step(params);
        });

    m.def("compute_loss_and_grad", [](Tensor1D &logits, int label) {
        Tensor1D grad(logits.size);
        float loss = cross_entropy_loss_and_grad(logits, label, grad);
        return std::make_pair(loss, grad);
    });

    m.def("argmax", &print_argmax);

    m.def("num_params", &num_params);
    m.def("macs_conv", &macs);
    m.def("macs_linear", &macs_linear);
}