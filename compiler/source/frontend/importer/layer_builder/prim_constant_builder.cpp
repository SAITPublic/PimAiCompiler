
#include "importer/layer_builder/layer_builder.h"
#include "new_ir/include/common/log.hpp"
#include "new_ir/include/tensors/data_tensor.h"

using nn_compiler::ir::DTensor;
using nn_compiler::ir::STensor;

namespace nn_compiler
{
namespace frontend
{
std::shared_ptr<ir::NNLayer> PrimConstantBuilder::buildLayer(const torch::jit::Node *node_ref)
{
    Log::NIR::I() << "build prim::Constant";
    std::string type = "prim::Constant";
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
