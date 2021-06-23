#include <cstdio>
#include <glog/logging.h>
#include <memory>

#include "nn_runtime.h"
namespace nnr {


NNRuntime::NNRuntime(const std::string torch_model_path) {
    LOG(INFO)<< "get torch model path:" << torch_model_path;  
    ModelBuilder builder;
    
    // compile and preload model
    // ...

    this->mbuilder = std::make_shared<ModelBuilder>(builder);

    this->executor = std::make_shared<StreamExecutor>();
}

int NNRuntime::inferenceModel(NnrBuffer *inputBuffer, NnrBuffer *outputBuffer) {
    LOG(INFO)<< "inferenceModel with inputBuffer and output in outputBuffer!";
    int ret = executor->inferenceModel(/*mbuilder->runnableIR,*/ inputBuffer, outputBuffer);
    return ret;
}

int NNRuntime::test(void) {
    LOG(INFO)<< "hello NNRuntime::test!";
    return 0;
}

} // namespace nnr
