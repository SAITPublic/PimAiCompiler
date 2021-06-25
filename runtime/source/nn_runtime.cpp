#include <glog/logging.h>
#include <cstdio>
#include <memory>

#include "nn_runtime.h"
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

int NNRuntime::inferenceModel(NnrtBuffer *inputBuffer, NnrtBuffer *outputBuffer)
{
    LOG(INFO) << "inferenceModel with inputBuffer and output in outputBuffer!";
    int ret = executor->inferenceModel(/*mbuilder->runnableIR,*/ inputBuffer, outputBuffer);
    return ret;
}

int NNRuntime::test(void)
{
    LOG(INFO) << "hello NNRuntime::test!";
    return 0;
}

}  // namespace nnrt
