#include <cstdio>
#include <memory>
#include <tuple>

#include "c10/hip/HIPFunctions.h"
#include "new_runtime/include/nn_runtime.h"
#include "pim_runtime_api.h"

namespace nn_compiler {
namespace runtime {

NNRuntime::NNRuntime(std::unique_ptr<nn_compiler::ir::NNModel> &model, std::string model_type)
{
    model_type_ = model_type;

    mbuilder_ = std::make_shared<ModelBuilder>();
    mbuilder_->preProcess(model);
    mbuilder_->preloadModel(model);

    executor_ = std::make_shared<StreamExecutor>(mbuilder_->getPreLoadedData(), model_type_);
    executor_->preProcess(model);

    rocblas_init();
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
}

std::vector<torch::Tensor> NNRuntime::inferenceModel(std::unique_ptr<nn_compiler::ir::NNModel> &model,
                                                     const std::vector<torch::Tensor>& input_tensors,
                                                     bool profiling)
{
    Log::RT::D() << "numInputs:" << input_tensors.size();
    for (auto& item : input_tensors) {
        Log::RT::D() << "shape:" << item.sizes() << " dtype:" << item.dtype() << " device:" << item.device();
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
        Log::RT::E() << " inference model fail!";
    }

    // for simple test, fill zeros to outputs
    // output_tensors.push_back(torch::zeros({10, 10}, torch::kF16));
    Log::RT::D() << "numOutps:" << output_tensors.size();

    return output_tensors;
}

int NNRuntime::rocblas_init(void)
{
    int M = 1;
    int N = 240;
    int K = 4096;
    auto l_gpu = at::randn({M, K}, at::kCUDA);
    auto r_gpu = at::randn({K, N}, at::kCUDA);
    // TODO(SRCX): release API when it gets ready
    // auto result = atenMatmul(l_gpu, r_gpu);
    at::hip::device_synchronize();

    return 0;
}

int NNRuntime::test(void)
{
    Log::RT::D() << "hello NNRuntime::test!";

    return 0;
}

NNRuntime::~NNRuntime() {
    Log::RT::D() << "NNRuntime Destructor is called";
    PimDeinitialize();
}

}  // namespace runtime
}  // namespace nn_compiler
