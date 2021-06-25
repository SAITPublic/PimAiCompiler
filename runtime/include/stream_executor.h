#pragma once

#include "model_builder.h"
#include "nnrt_types.h"

namespace nnrt
{
class StreamExecutor
{
   public:
    StreamExecutor() {}

    int inferenceModel(/* RunnableNNIR IR,*/ NnrtBuffer* inputBuffer, NnrtBuffer* outputBuffer);

   private:
};

}  // namespace nnrt
