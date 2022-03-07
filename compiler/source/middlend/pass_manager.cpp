#include "compiler/include/middlend/pass_manager.h"
#include "compiler/include/common/log.hpp"

#include "compiler/include/middlend/lstm_labeling.h"
#include "compiler/include/middlend/update_layer_id.h"
#include "compiler/include/middlend/control_layer_execution.h"
#include "compiler/include/middlend/cat_labeling.h"

namespace nn_compiler
{
namespace middlend
{
PassManager::PassManager() { }
void PassManager::runPasses(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    Log::FE::I() << "PassManager::runPasses is called.";
    auto base_pass = std::make_shared<Pass>();

    /*TODO: add middlend passes*/
    auto update_layer_id = std::make_shared<UpdateLayerId>();
    auto control_layer_execution = std::make_shared<ControlLayerExecution>();
    auto lstm_labeling = std::make_shared<LstmLabeling>();
    auto cat_labeling = std::make_shared<CatLabeling>();

    base_pass->add(update_layer_id);
    update_layer_id->add(control_layer_execution);
    control_layer_execution->add(lstm_labeling);
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
