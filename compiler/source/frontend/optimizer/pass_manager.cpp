#include "compiler/include/frontend/optimizer/pass_manager.h"
#include "compiler/include/common/log.hpp"
#include "compiler/include/frontend/optimizer/construct_list.h"
#include "compiler/include/frontend/optimizer/remake_dtensor_of_prim_variable.h"
#include "compiler/include/frontend/optimizer/remove_get_attr_layers.h"
#include "compiler/include/frontend/optimizer/remove_if_with_addmm.h"

#include "compiler/include/frontend/optimizer/take_in_body_net.h"

namespace nn_compiler {
namespace frontend {

PassManager::PassManager(const std::string& model_name) { model_name_ = model_name; }

void PassManager::runPasses(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
    Log::FE::I() << "PassManager::runPasses is called.";
    auto base_pass = std::make_shared<Pass>();

    // 1. TODO(SRCX): declare optimization passes, like: auto fuse_act = std::make_shared<FuseActivation>();
    auto take_in_body_net = std::make_shared<TakeInBodyNet>();
    auto construct_list = std::make_shared<ConstructList>();
    auto remake_dtensor_of_prim_variable = std::make_shared<RemakeDTensorOfPrimVariable>();

    auto remove_get_attr_layers          = std::make_shared<RemoveGetAttrLayers>();
    auto remove_if_with_addmm            = std::make_shared<RemoveIfWithAddmm>();
    // 2. TODO(SRCX): add optimization passes, like: base_pass->add(fuse_act);
    base_pass->add(take_in_body_net);
    take_in_body_net->add(construct_list);
    construct_list->add(remake_dtensor_of_prim_variable);

    remove_get_attr_layers->add(remove_if_with_addmm);
    remove_if_with_addmm->add(remove_cat_for_addmm);

    while (base_pass->getSuccessor() != nullptr) {
        base_pass = base_pass->getSuccessor();
        if (base_pass->fitCondition(model)) {
            base_pass->run(model);
        }
    }
}

}  // namespace frontend
}  // namespace nn_compiler
