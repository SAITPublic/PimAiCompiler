

#include "stream_executor.h"
#include "nnrt_types.h"

namespace nnrt
{
int StreamExecutor::inferenceModel(/* RunnableNNIR IR,*/ NnrtBuffer* inputBuffer, NnrtBuffer* outputBuffer)
{
    return 0;
}

// execute current op in runtime
void executeOp(OpNodeDescription* cur_op)
{
    // TODO
}

/**
 * @brief Get the Next Execution Node object
 *
 * @param cur_op currently Op
 * @return OpNodeDescription* the next
 */
OpNodeDescription* getNextExecutionOp(OpNodeDescription* cur_op)
{
    // TODO
}

} // namespace nnrt

