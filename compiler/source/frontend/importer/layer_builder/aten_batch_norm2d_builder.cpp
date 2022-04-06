#include "importer/layer_builder/layer_builder.h"
#include "importer/utils/attr_parser.h"
#include "ir/include/tensors/data_tensor.h"

using nn_compiler::ir::DTensor;
using nn_compiler::ir::STensor;

namespace nn_compiler
{
namespace frontend
{
std::shared_ptr<ir::NNLayer> AtenBatchNorm2dBuilder::buildLayer(const torch::jit::Node* node_ref)
{
    DLOG(INFO) << "build aten::batch_norm";

    nn_compiler::ir::LayerType type = nn_compiler::ir::LayerType::ATENBATCHNORM2D;
    std::string name = "";

    aten_batch_norm2d_layer_ = std::make_shared<ir::AtenBatchNorm2dLayer>(name, type);

    auto weight_bias = parser()->getGeneralWeightAndBias(node_ref);
    aten_batch_norm2d_layer_->setWeights(weight_bias.first);
    aten_batch_norm2d_layer_->setBiases(weight_bias.second);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(aten_batch_norm2d_layer_);
    return layer;
}

}  // namespace frontend
}  // namespace nn_compiler
