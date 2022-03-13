#include "importer/layer_builder/layer_builder.h"
#include "ir/include/common/log.hpp"
#include "ir/include/tensors/data_tensor.h"

using nn_compiler::ir::DTensor;
using nn_compiler::ir::STensor;

namespace nn_compiler
{
namespace frontend
{
std::shared_ptr<ir::NNLayer> AtenConv2dBuilder::buildLayer(const torch::jit::Node* node_ref)
{
    Log::IR::I() << "build aten::conv2d";

    nn_compiler::ir::LayerType type = nn_compiler::ir::LayerType::ATENCONV2D;
    std::string name = "";

    aten_conv2d_layer_ = std::make_shared<ir::AtenConv2dLayer>(name, type);

    // get weights
    auto weight_node = node_ref->inputs()[1]->node();
    assert(weight_node->kind() == c10::prim::Constant);
    assert(weight_node->hasAttribute(c10::attr::value));
    auto weight_tensor_cpu = weight_node->t(c10::attr::value);
    auto weight_tensor = std::move(weight_tensor_cpu.cuda());
    std::vector<at::Tensor> weight_vec;
    weight_vec.push_back(weight_tensor);
    // get bias
    auto bias_node = node_ref->inputs()[2]->node();
    assert(bias_node->kind() == c10::prim::Constant);
    assert(bias_node->hasAttribute(c10::attr::value));
    auto bias_tensor_cpu = bias_node->t(c10::attr::value);
    auto bias_tensor = std::move(bias_tensor_cpu.cuda());
    std::vector<at::Tensor> bias_vec;
    bias_vec.push_back(bias_tensor);

    aten_conv2d_layer_->setWeights(weight_vec);
    aten_conv2d_layer_->setBiases(bias_vec);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(aten_conv2d_layer_);
    return layer;
}

}  // namespace frontend
}  // namespace nn_compiler
