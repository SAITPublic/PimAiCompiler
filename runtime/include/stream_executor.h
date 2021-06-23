#pragma once

#include "model_builder.h"
#include "nnr_types.h"

namespace nnr {

class StreamExecutor {

  public:
    StreamExecutor(){}

    int inferenceModel(/* RunnableNNIR IR,*/  NnrBuffer* inputBuffer, NnrBuffer* outputBuffer);
  private:


};

} // namespace nnr
