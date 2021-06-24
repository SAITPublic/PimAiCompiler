#pragma once

#include <memory>

#include "nnrt_types.h"
#include "model_builder.h"
#include "stream_executor.h"

namespace nnrt {

class NNRuntime {
  public:
    NNRuntime(){}

    NNRuntime(const std::string torch_model_path);

    int inferenceModel(NnrtBuffer* inputBuffer, NnrtBuffer* outputBuffer);
  
    int test(void);
  private:

    // Runnable NNIR in ModelBuilder
    std::shared_ptr<ModelBuilder> mbuilder;

    std::shared_ptr<StreamExecutor> executor;
};

} // namespace nnrt
