#include "frontend/importer/layer_builder/layer_builder.h"

namespace nn_compiler
{
namespace frontend
{
std::shared_ptr<ir::NNLayer> AtenConv2dBuilder::buildLayer(const torch::jit::Node* node_ref)
{
    DLOG(INFO) << "build aten::conv2d";

    nn_compiler::ir::LayerType type = nn_compiler::ir::LayerType::ATENCONV2D;
    std::string name = "";

    aten_conv2d_layer_ = std::make_shared<ir::AtenConv2dLayer>(name, type);

    auto weight_bias = parser()->getGeneralWeightAndBias(node_ref);
    aten_conv2d_layer_->setWeights(weight_bias.first);
    aten_conv2d_layer_->setBiases(weight_bias.second);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(aten_conv2d_layer_);
    return layer;
}

}  // namespace frontend
}  // namespace nn_compiler
