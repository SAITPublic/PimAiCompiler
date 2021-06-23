#pragma once

#include <memory>

#include "nnr_types.h"
#include "model_builder.h"
#include "stream_executor.h"

namespace nnr {

class NNRuntime {
  public:
    NNRuntime(){}

    NNRuntime(const std::string torch_model_path);

    int inferenceModel(NnrBuffer* inputBuffer, NnrBuffer* outputBuffer);
  
    int test(void);
  private:

    // Runnable NNIR in ModelBuilder
    std::shared_ptr<ModelBuilder> mbuilder;

    std::shared_ptr<StreamExecutor> executor;
};

} // namespace nnr
