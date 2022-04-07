
#include "frontend/importer/layer_builder/layer_builder.h"

namespace nn_compiler
{
namespace frontend
{
std::shared_ptr<ir::NNLayer> PrimConstantBuilder::buildLayer(const torch::jit::Node *node_ref)
{
    DLOG(INFO) << "build prim::Constant";
    nn_compiler::ir::LayerType type = nn_compiler::ir::LayerType::PRIMCONSTANT;
    std::string name = "";
    prim_constant_layer_ = std::make_shared<ir::PrimConstantLayer>(name, type);
    std::string ntype = node_ref->output()->type()->str();
    prim_constant_layer_->setNType(ntype);
    auto data = getDTensorData(node_ref);
    prim_constant_layer_->setAttr(data);
    return prim_constant_layer_;
}

}  // namespace frontend
}  // namespace nn_compiler
