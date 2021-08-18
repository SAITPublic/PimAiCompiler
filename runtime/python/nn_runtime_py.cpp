#include <torch/extension.h>
#include "nn_runtime.h"

namespace py = pybind11;
using namespace nnrt;

PYBIND11_MODULE(Nnrt, m)
{
    py::class_<NNRuntime>(m, "NNRuntime")
        .def(py::init<const std::string, const int>())
        .def("test", &NNRuntime::test)
        .def("inferenceModel", &NNRuntime::inferenceModel, py::arg("input_tensors"));
}
