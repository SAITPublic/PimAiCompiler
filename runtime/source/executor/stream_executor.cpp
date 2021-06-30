#include "executor/stream_executor.h"
#include <torch/script.h>
#include "nnrt_types.h"
#include "ir/include/nn_ir.hpp"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{

RetVal StreamExecutor::inferenceModel(const std::shared_ptr<nncir::NNIR> graph, const std::vector<torch::Tensor>& input_tensors,
                                   std::vector<torch::Tensor>& output_tensors)
{
    for (auto&& node : graph->getNodes()) {
        LOG(INFO) <<"Node id:" <<node.getId() <<" name:" <<node.getName() << " type:" << node.getNodeType();
    }
    return RetVal::SUCCESS;
}

void executeOp(OpNodeDescription* cur_op)
{

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
    return nullptr;
}


}  // namespace nnrt

