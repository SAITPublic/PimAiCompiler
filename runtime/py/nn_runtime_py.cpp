#include "nn_runtime_api.h"

#include<pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(Nnr, m) {

    m.doc() = "NNRuntime module";

    // Add bindings here
    m.def("test", &NnrTest, "A test funtion");
    // m.def("inference", &NnrInference, "inference once");

}
