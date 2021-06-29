#include "stream_executor.h"
#include <torch/script.h>
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

int StreamExecutor::inferenceModel(const std::vector<torch::Tensor>& input_tensors, std::vector<torch::Tensor>& output_tensors)
{
    return 0;
}

}  // namespace nnrt

