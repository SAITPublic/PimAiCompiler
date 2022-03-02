#include "importer/layer_builder/layer_builder.h"
#include "new_ir/include/common/log.hpp"
#include "new_ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace frontend
{
std::shared_ptr<ir::NNLayer> PrimCallMethodBuilder::buildLayer(const torch::jit::Node* node_ref)
{
    Log::NIR::I() << "NOT uesed PrimCallMethodBuilder::buildLayer";
    return nullptr;
}
std::shared_ptr<ir::NNLayer> PrimCallMethodBuilder::buildLayerCustom(const std::string target_network_name)
{
    Log::NIR::I() << "build prim::callmethod";
    nn_compiler::ir::LayerType type = nn_compiler::ir::LayerType::PRIMCALLMETHOD;
    std::string name = "";

    prim_callmethod_layer_ = std::make_shared<ir::PrimCallMethodLayer>(name, type);
    prim_callmethod_layer_->setAttr(target_network_name);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(prim_callmethod_layer_);
    return layer;
}

}  // namespace frontend
}  // namespace nn_compiler
