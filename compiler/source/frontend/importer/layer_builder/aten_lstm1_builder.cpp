#include "importer/layer_builder/layer_builder.h"
#include "new_ir/include/common/log.hpp"
#include "new_ir/include/tensors/data_tensor.h"

using nn_compiler::ir::DTensor;
using nn_compiler::ir::STensor;

namespace nn_compiler
{
namespace frontend
{
std::shared_ptr<ir::NNLayer> AtenLSTM1Builder::buildLayer(const torch::jit::Node* node_ref)
{
    Log::NIR::I() << "build aten::lstm1";

    nn_compiler::ir::LayerType type = nn_compiler::ir::LayerType::ATENLSTM1;
    std::string name = "";

    aten_lstm1_layer_ = std::make_shared<ir::AtenLSTM1Layer>(name, type);

    // the learnable parameter of aten::lstm contains 8 or 12 tensors in RNNT
    // model, they are all prim::Constant these tensors are input to
    // prim::ListConstrcut and return a tensor[], instead of to unpack these
    // tensor from tensor[], we firstly get the previous prim::ListConstruct Node
    // and then get all the inputs of prim::ListConstruct
    std::vector<at::Tensor> weight_vec;
    std::vector<at::Tensor> bias_vec;
    // get the prim::ListConstruct
    auto list_construct = node_ref->inputs()[3]->node();
    if (list_construct->kind() != c10::prim::ListConstruct) {
        list_construct = node_ref->inputs()[2]->node();
    }
    // get each input of prim::ListConstruct
    for (auto item : list_construct->inputs()) {
        auto constant_node = item->node();
        assert(constant_node->kind() == c10::prim::Constant);
        assert(constant_node->hasAttribute(c10::attr::value));
        // get the at::Tensor from prim::Constant
        auto torch_tensor = constant_node->t(c10::attr::value);
        if (torch_tensor.dim() == 1) {
            bias_vec.push_back(torch_tensor);
        } else {
            weight_vec.push_back(torch_tensor);
        }
    }

    aten_lstm1_layer_->setWeights(weight_vec);
    aten_lstm1_layer_->setBiases(bias_vec);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(aten_lstm1_layer_);
    return layer;
}

}  // namespace frontend
}  // namespace nn_compiler
