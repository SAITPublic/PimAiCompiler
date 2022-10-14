#include "frontend/optimizer/pass_manager.h"

#include "frontend/optimizer/construct_list.h"
#include "frontend/optimizer/convert_linear_to_addmm.h"
#include "frontend/optimizer/fuse_activation.h"
#include "frontend/optimizer/remake_dtensor_of_prim_variable.h"
#include "frontend/optimizer/remove_cat_for_addmm.h"
#include "frontend/optimizer/remove_constant_layers.h"
#include "frontend/optimizer/remove_dropout_layers.h"
#include "frontend/optimizer/remove_get_attr_layers.h"
#include "frontend/optimizer/remove_if_with_addmm.h"
#include "frontend/optimizer/remove_set_attr_layers.h"
#include "frontend/optimizer/set_attribute.h"
#include "frontend/optimizer/set_weights_for_embedding.h"
#include "frontend/optimizer/swap_addmm_inputs.h"
#include "frontend/optimizer/swap_matmul_inputs.h"
#include "frontend/optimizer/take_in_body_net.h"

namespace nn_compiler
{
namespace frontend
{
PassManager::PassManager(const std::string& model_name) { model_name_ = model_name; }

void PassManager::runPasses(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "frontend PassManager::runPasses is called.";
    auto base_pass = std::make_shared<Pass>();

    auto take_in_body_net = std::make_shared<TakeInBodyNet>();
    auto construct_list = std::make_shared<ConstructList>();
    auto remake_dtensor_of_prim_variable = std::make_shared<RemakeDTensorOfPrimVariable>();
    auto set_attribute = std::make_shared<SetAttribute>();
    auto remove_constant_layers = std::make_shared<RemoveConstantLayers>();
    auto remove_dropout_layers = std::make_shared<RemoveDropoutLayers>();
    auto remove_set_attr_layers = std::make_shared<RemoveSetAttrLayers>();
    auto remove_get_attr_layers = std::make_shared<RemoveGetAttrLayers>();
    auto convert_linear_to_addmm = std::make_shared<ConvertLinearToAddmm>();
    auto remove_if_with_addmm = std::make_shared<RemoveIfWithAddmm>();
    auto remove_cat_for_addmm = std::make_shared<RemoveCatForAddmm>();
    auto swap_addmm_inputs = std::make_shared<SwapAddmmInputs>();
    auto swap_matmul_inputs = std::make_shared<SwapMatmulInputs>();
    auto fuse_act = std::make_shared<FuseActivation>();
    auto set_weights_for_embedding = std::make_shared<SetWeightsForEmbedding>();

    base_pass->add(take_in_body_net);
    take_in_body_net->add(construct_list);
    construct_list->add(remake_dtensor_of_prim_variable);
    remake_dtensor_of_prim_variable->add(set_attribute);
    set_attribute->add(remove_constant_layers);
    remove_constant_layers->add(remove_dropout_layers);
    remove_dropout_layers->add(remove_set_attr_layers);
    remove_set_attr_layers->add(remove_get_attr_layers);
    remove_get_attr_layers->add(convert_linear_to_addmm);
    convert_linear_to_addmm->add(remove_if_with_addmm);
    remove_if_with_addmm->add(swap_addmm_inputs);
    // TODO(SRCX): Fix remove_cat_for_addmm for torch-1.10 models.
    // remove_cat_for_addmm->add(swap_addmm_inputs);
    swap_addmm_inputs->add(swap_matmul_inputs);
    swap_matmul_inputs->add(fuse_act);

    if (model_name_ == "RNNT") {
        fuse_act->add(set_weights_for_embedding);
    }

    while (base_pass->getSuccessor() != nullptr) {
        base_pass = base_pass->getSuccessor();
        if (base_pass->fitCondition(model)) {
            base_pass->run(model);
        }
    }
}

}  // namespace frontend
}  // namespace nn_compiler
