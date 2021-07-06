#include "nn_runtime.h"
#include <glog/logging.h>
#include <cstdio>
#include <memory>
#include <tuple>

namespace nnrt
{
NNRuntime::NNRuntime(const std::string torch_model_path)
{
    DLOG(INFO) << "get torch model path:" << torch_model_path;

    // compile and preload model
    ModelBuilder builder(torch_model_path);
    builder.compileModel();
    builder.preloadModel();

    this->mbuilder_ = std::make_shared<ModelBuilder>(builder);

    this->executor_ = std::make_shared<StreamExecutor>(this->mbuilder_->get_runnable_ir());
}

std::vector<torch::Tensor> NNRuntime::inferenceModel(const std::vector<torch::Tensor>& input_tensors)
{
    DLOG(INFO) << "numInputs:" << input_tensors.size();
    for (auto& item : input_tensors) {
        DLOG(INFO) << "shape:" << item.sizes() << " dtype:" << item.dtype() << " device:" << item.device();
    }

    std::vector<torch::Tensor> output_tensors;
    auto status = executor_->inferenceModel(this->mbuilder_->get_runnable_ir(), input_tensors, output_tensors);
    if (status != RetVal::SUCCESS) {
        LOG(ERROR) << " inference model fail!";
    }

    // for simple test, fill zeros to outputs
    // output_tensors.push_back(torch::zeros({10, 10}, torch::kF16));

    DLOG(INFO) << "numOutps:" << output_tensors.size() << std::endl;

    return output_tensors;
}

int NNRuntime::test(void)
{
    LOG(INFO) << "hello NNRuntime::test!";
    return 0;
}

}  // namespace nnrt
