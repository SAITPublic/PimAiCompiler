#pragma once

#include <string>
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

// execute current op in runtime
void executeOp(OpNodeDescription* cur_op);

/**
 * @brief Get the Next Execution Node object
 *
 * @param cur_op currently Op
 * @return OpNodeDescription* the next
 */
OpNodeDescription* getNextExecutionOp(OpNodeDescription* cur_op);

} // namespace nnrt

