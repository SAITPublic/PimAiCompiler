#include<pybind11/pybind11.h>
#include "nn_runtime.h"

namespace py = pybind11;
using namespace nnr;

PYBIND11_MODULE(Nnr, m) {

    py::class_<NNRuntime>(m, "NNRuntime")
    .def(py::init<const std::string>())
    .def("test", &NNRuntime::test);

    // Add bindings here
    // m.def("test", &NnrTest, "A test funtion");
    // m.def("inference", &NnrInference, "inference once");

}
