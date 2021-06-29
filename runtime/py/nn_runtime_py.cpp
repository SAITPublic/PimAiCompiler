// #include<pybind11/pybind11.h>
#include<torch/extension.h>
#include "nn_runtime.h"

namespace py = pybind11;
using namespace nnrt;

PYBIND11_MODULE(Nnrt, m) {

    py::class_<NNRuntime>(m, "NNRuntime")
    .def(py::init<const std::string>())
    .def("test", &NNRuntime::test)
    .def("inferenceModel", &NNRuntime::inferenceModel, py::arg("input_tensors"));

    // Add bindings here
    // m.def("test", &NnrtTest, "A test funtion");
    // m.def("inference", &NnrtInference, "inference once");
}
