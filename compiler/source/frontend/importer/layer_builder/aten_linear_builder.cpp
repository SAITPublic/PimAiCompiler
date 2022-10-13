#include "frontend/importer/layer_builder/layer_builder.h"
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace frontend
{
/*
 * Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias = {})
 *
 */

std::shared_ptr<ir::NNLayer> AtenLinearBuilder::buildLayer(const torch::jit::Node* node_ref)
{
    DLOG(INFO) << "build aten::linear";

    nn_compiler::ir::LayerType type = nn_compiler::ir::LayerType::ATENLINEAR;
    std::string name = "";

    aten_linear_layer_ = std::make_shared<ir::AtenLinearLayer>(name, type);

    auto weight_bias = parser()->getGeneralWeightAndBias(node_ref, 1, 2);
    aten_linear_layer_->setWeights(weight_bias.first);
    aten_linear_layer_->setBiases(weight_bias.second);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(aten_linear_layer_);

    return layer;
}

}  // namespace frontend
}  // namespace nn_compiler
