#include <glog/logging.h>
#include <cstdio>
#include <memory>
#include "nn_runtime.h"
#include <tuple>

namespace nnrt
{
NNRuntime::NNRuntime(const std::string torch_model_path)
{
    LOG(INFO) << "get torch model path:" << torch_model_path;
    ModelBuilder builder;

    // compile and preload model
    // ...

    this->mbuilder = std::make_shared<ModelBuilder>(builder);

    this->executor = std::make_shared<StreamExecutor>();
}

// int NNRuntime::inferenceModel(NnrtBuffer *inputBuffer, NnrtBuffer *outputBuffer) {
//     LOG(INFO)<< "inferenceModel with inputBuffer and output in outputBuffer!";
//     int ret = executor->inferenceModel(/*mbuilder->runnableIR,*/ inputBuffer, outputBuffer);
//     return ret;
// }

std::vector<torch::Tensor> NNRuntime::inferenceModel(const std::vector<torch::Tensor>& input_tensors)
{
    std::cout << "numInputs:" << input_tensors.size() << std::endl;
    LOG(INFO) << "inferenceModel with inputTensors and output in outputTensors!";
    LOG(INFO) << "numInputs:" << input_tensors.size();
    for (auto& item : input_tensors) {
        LOG(INFO) << "shape:" << item.sizes() << " dtype:" << item.dtype() << " device:" << item.device();
        std::cout << "shape:" << item.sizes() << " dtype:" << item.dtype() << " device:" << item.device() <<std::endl;
    }

    std::vector<torch::Tensor> output_tensors;
    int status = executor->inferenceModel(input_tensors, output_tensors);

    // for simple test, fill zeros to outputs
    output_tensors.push_back(torch::zeros({10,10}, torch::kF16));
    output_tensors.push_back(torch::zeros({5,5}, torch::kF16));

    std::cout << "numOutps:" << output_tensors.size() << std::endl;
    
    return output_tensors;
}

int NNRuntime::test(void)
{
    LOG(INFO) << "hello NNRuntime::test!";
    return 0;
}

}  // namespace nnrt
