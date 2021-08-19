#include <torch/extension.h>
#include "nn_runtime.h"

namespace py = pybind11;
using namespace nnrt;

PYBIND11_MODULE(Nnrt, m)
{
    py::class_<NNRuntime>(m, "NNRuntime")
        // ref: https://pybind11.readthedocs.io/en/latest/advanced/functions.html#default-arguments-revisited
        .def(py::init<const std::string, const int>(), py::arg("input_file"), py::arg("compile_level") = 1)
        .def("test", &NNRuntime::test)
        .def("inferenceModel", &NNRuntime::inferenceModel, py::arg("input_tensors"));
}
