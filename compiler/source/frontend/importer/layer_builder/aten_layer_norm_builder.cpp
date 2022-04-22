#include "frontend/importer/layer_builder/layer_builder.h"
#include "ir/include/tensors/data_tensor.h"

using nn_compiler::ir::DTensor;
using nn_compiler::ir::STensor;

namespace nn_compiler
{
namespace frontend
{
std::shared_ptr<ir::NNLayer> AtenLayerNormBuilder::buildLayer(const torch::jit::Node* node_ref)
{
    DLOG(INFO) << "build aten::layer_norm";

    nn_compiler::ir::LayerType type = nn_compiler::ir::LayerType::ATENLAYERNORM;
    std::string name = "";

    aten_layer_norm_layer_ = std::make_shared<ir::AtenLayerNormLayer>(name, type);

    auto weight_bias = parser()->getGeneralWeightAndBias(node_ref, 2, 3);
    aten_layer_norm_layer_->setWeights(weight_bias.first);
    aten_layer_norm_layer_->setBiases(weight_bias.second);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(aten_layer_norm_layer_);

    return layer;
}

}  // namespace frontend
}  // namespace nn_compiler
