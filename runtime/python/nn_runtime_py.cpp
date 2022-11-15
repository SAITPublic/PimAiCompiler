/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include <torch/extension.h>
#include "pipeline_manager/pipeline_manager.h"

namespace py = pybind11;
using namespace NNRuntimeInterface;

PYBIND11_MODULE(NNCompiler, m)
{
    py::class_<PipelineManager>(m, "PipelineManager")
        // ref: https://pybind11.readthedocs.io/en/latest/advanced/functions.html#default-arguments-revisited
        .def(py::init<const std::string&, std::string, int>(), py::arg("input_file"), py::arg("model_type") = "",
             py::arg("gpu_num") = 1)
        .def("inferenceModel", &PipelineManager::inferenceModel, py::arg("input_tensors"));
}
