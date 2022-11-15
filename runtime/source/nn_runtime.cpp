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

#include "c10/hip/HIPFunctions.h"
#include "executor/op_executor/aten_ops_executor.h"
#include "nn_runtime.h"
#include "pim_runtime_api.h"

namespace nn_compiler
{
namespace runtime
{
NNRuntime::NNRuntime(std::unique_ptr<nn_compiler::ir::NNModel>& model, std::string model_type)
{
    model_type_ = model_type;

    mbuilder_ = std::make_shared<ModelBuilder>();
    mbuilder_->preProcess(model);
    mbuilder_->preloadModel(model);
    executor_ = std::make_shared<StreamExecutor>(std::make_pair(model->getGraphs()[0], mbuilder_->getPreLoadedData()),
                                                 model_type_);
    executor_->preProcess();

    PimInitialize(RT_TYPE_HIP, PIM_FP16);
    rocblas_init();
}

void NNRuntime::inferenceModel(const std::vector<torch::Tensor>& input_tensors,
                               std::vector<torch::Tensor>& output_tensors, bool profiling)
{
    DLOG(INFO) << "numInputs:" << input_tensors.size();
    for (auto& item : input_tensors) {
        DLOG(INFO) << "shape:" << item.sizes() << " dtype:" << item.dtype() << " device:" << item.device();
    }

    auto status = RetVal::FAILURE;
    if (profiling) {
        // profiling result of the first running time is not accurate.
        for (int i = 0; i < 10; i++) {
            status = executor_->inferenceModel(input_tensors, output_tensors);
        }
        status = executor_->inferenceModelwithProfiling(input_tensors, output_tensors);
    } else {
        status = executor_->inferenceModel(input_tensors, output_tensors);
    }

    if (status != RetVal::SUCCESS) {
        DLOG(FATAL) << " inference model fail!";
    }

    DLOG(INFO) << "Num of Outputs: " << output_tensors.size();
}

int NNRuntime::rocblas_init(void)
{
    int M = 1;
    int N = 240;
    int K = 4096;
    auto l_gpu = at::randn({M, K}, at::kCUDA);
    auto r_gpu = at::randn({K, N}, at::kCUDA);
    auto result = op_executor::atenMatmul(l_gpu, r_gpu);
    at::hip::device_synchronize();

    return 0;
}

int NNRuntime::test(void)
{
    DLOG(INFO) << "hello NNRuntime::test!";

    return 0;
}

NNRuntime::~NNRuntime()
{
    DLOG(INFO) << "NNRuntime Destructor is called";
    PimDeinitialize();
}

}  // namespace runtime
}  // namespace nn_compiler
