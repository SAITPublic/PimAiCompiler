#pragma once

#include <torch/script.h>
#include <string>
#include "model_builder.h"
#include "nnrt_types.h"

namespace nnrt
{
class StreamExecutor
{
   public:
    StreamExecutor() {}

    RetVal inferenceModel(const std::vector<torch::Tensor>& input_tensors, std::vector<torch::Tensor>& output_tensors);

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

}  // namespace nnrt
