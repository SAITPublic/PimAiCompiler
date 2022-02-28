#include <glog/logging.h>
#include <cstdio>
#include <memory>
#include <tuple>

#include "c10/hip/HIPFunctions.h"
#include "nn_runtime.h"
#include "pim_runtime_api.h"

namespace nn_compiler {
namespace runtime {

NNRuntime::NNRuntime(std::unique_ptr<nn_compiler::ir::NNModel> &model, std::string model_type)
{
    model_type_ = model_type;

    mbuilder_ = std::make_shared<ModelBuilder>();
    mbuilder_->preProcess(model);
    mbuilder_->preloadModel(model);

    this->executor_ = std::make_shared<StreamExecutor>(mbuilder_->getPreLoadedData(), model_type_);

    rocblas_init();
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
}

std::vector<torch::Tensor> NNRuntime::inferenceModel(std::unique_ptr<nn_compiler::ir::NNModel> &model,
                                                     const std::vector<torch::Tensor>& input_tensors,
                                                     bool profiling)
{
    DLOG(INFO) << "numInputs:" << input_tensors.size();
    for (auto& item : input_tensors) {
        DLOG(INFO) << "shape:" << item.sizes() << " dtype:" << item.dtype() << " device:" << item.device();
    }

    std::vector<torch::Tensor> output_tensors;
    auto status = RetVal::FAILURE;
    if (profiling) {
        // profiling result of the first running time is not accurate.
        for (int i = 0; i < 10; i ++) {
            status = executor_->inferenceModel(model, input_tensors, output_tensors);
        }
        status = executor_->inferenceModelwithProfiling(model, input_tensors, output_tensors);
    } else {
        status = executor_->inferenceModel(model, input_tensors, output_tensors);
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

NNRuntime::~NNRuntime() {
    DLOG(INFO) << "NNRuntime Destructor is called";
    PimDeinitialize();
}

}  // namespace nnrt