#include <torch/extension.h>
#include "pipeline_manager/pipeline_manager.h"

namespace py = pybind11;
using namespace NNRuntimeInterface;

PYBIND11_MODULE(NNCompiler, m)
{
    py::class_<PipelineManager>(m, "PipelineManager")
        // ref: https://pybind11.readthedocs.io/en/latest/advanced/functions.html#default-arguments-revisited
        .def(py::init<const std::string&, std::string>(), py::arg("input_file"), py::arg("model_type") = "")
        .def("inferenceModel", &PipelineManager::inferenceModel, py::arg("input_tensors"));
}
