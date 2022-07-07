#include "middlend/optimizer/pass_manager.h"

#include "middlend/optimizer/memory_allocation/cat_labeling.h"
#include "middlend/optimizer/memory_allocation/lstm_labeling.h"
#include "middlend/optimizer/stream_execution/control_layer_execution.h"
#include "middlend/optimizer/stream_execution/multi_stream_execution.h"

namespace nn_compiler
{
namespace middlend
{
PassManager::PassManager() {}

void PassManager::runPasses(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "PassManager::runPasses is called.";
    auto base_pass = std::make_shared<Pass>();

    auto control_layer_execution = std::make_shared<ControlLayerExecution>();
    auto muti_stream_execution = std::make_shared<MutiStreamExecution>();
    auto lstm_labeling = std::make_shared<LstmLabeling>();
    auto cat_labeling = std::make_shared<CatLabeling>();

    base_pass->add(control_layer_execution);
    control_layer_execution->add(muti_stream_execution);
    muti_stream_execution->add(lstm_labeling);
    lstm_labeling->add(cat_labeling);

    while (base_pass->getSuccessor() != nullptr) {
        base_pass = base_pass->getSuccessor();
        if (base_pass->fitCondition(model)) {
            base_pass->run(model);
        }
    }
}

}  // namespace middlend
}  // namespace nn_compiler
