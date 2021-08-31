#include "nn_runtime.h"
#include <glog/logging.h>
#include <cstdio>
#include <memory>
#include <tuple>

namespace nnrt
{
NNRuntime::NNRuntime(const std::string torch_model_path, int compile_level, std::string model_type)
{
    DLOG(INFO) << "get torch model path:" << torch_model_path;

    // compile and preload model
    ModelBuilder builder(torch_model_path);
    builder.compileModel(compile_level, model_type);
    builder.preloadModel();

    this->mbuilder_ = std::make_shared<ModelBuilder>(builder);

    this->executor_ = std::make_shared<StreamExecutor>(this->mbuilder_->get_runnable_ir());

    rocblas_init();
}

std::vector<torch::Tensor> NNRuntime::inferenceModel(const std::vector<torch::Tensor>& input_tensors, bool profiling)
{
    DLOG(INFO) << "numInputs:" << input_tensors.size();
    for (auto& item : input_tensors) {
        DLOG(INFO) << "shape:" << item.sizes() << " dtype:" << item.dtype() << " device:" << item.device();
    }

    std::vector<torch::Tensor> output_tensors;
    auto status = RetVal::FAILURE;
    if (profiling) {
        status =
            executor_->inferenceModelwithProfiling(this->mbuilder_->get_runnable_ir(), input_tensors, output_tensors);
    } else {
        status = executor_->inferenceModel(this->mbuilder_->get_runnable_ir(), input_tensors, output_tensors);
    }

    if (status != RetVal::SUCCESS) {
        LOG(ERROR) << " inference model fail!";
    }

    // for simple test, fill zeros to outputs
    // output_tensors.push_back(torch::zeros({10, 10}, torch::kF16));

    DLOG(INFO) << "numOutps:" << output_tensors.size() << std::endl;

    return output_tensors;
}

int NNRuntime::rocblas_init(void)
{
    int M = 1;
    int N = 240;
    int K = 4096;
    auto l_gpu = at::randn({M, K}, at::kCUDA);
    auto r_gpu = at::randn({K, N}, at::kCUDA);
    auto result = nnrt::atenMatmul(l_gpu, r_gpu);
    at::hip::device_synchronize();

    return 0;
}

int NNRuntime::test(void)
{
    DLOG(INFO) << "hello NNRuntime::test!";

    return 0;
}

}  // namespace nnrt
