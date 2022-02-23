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

    std::string type = "aten::lstm1";
    std::string name = "";

    aten_lstm1_layer_ = std::make_shared<ir::AtenLSTM1Layer>(name, type);
    const torch::jit::Node* node_lstm1 = node_ref;

    // get the weights/bias from jit::Node and add to layer
    std::vector<DTensor> weights;
    std::vector<DTensor> biases;
    getLearnableParameters(node_lstm1, weights, biases);
    aten_lstm1_layer_->setWeights(weights);
    aten_lstm1_layer_->setBiases(biases);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(aten_lstm1_layer_);
    return layer;
}

/**
 * @brief Convert the FP32/FP16(HALF) type torch at::Tensor to
 * nn_compiler::ir::DTensor
 *
 * @param torch_tensor the input at::Tensor
 * @param d_tensor the convereted DTensor
 */
void AtenLSTM1Builder::ptTensor2nncompilerDTensor(at::Tensor torch_tensor, DTensor& d_tensor)
{
    c10::ScalarType dtype = torch_tensor.scalar_type();
    // dtype of the weights of LSTM are float32 or float16/half
    assert(dtype == c10::ScalarType::Float || dtype == c10::ScalarType::Half);

    // get the data pointer of tensor
    int num_elements = torch_tensor.numel();
    auto tensor_data = torch_tensor.data_ptr();

    // set the data pointer to DTensor
    if (dtype == c10::ScalarType::Float) {
        d_tensor.setData(tensor_data, num_elements * sizeof(float));
        d_tensor.setDataType(nn_compiler::ir::DataType::FLOAT32);
        // for float32, bit_width = 32
        d_tensor.setBitWidth(32);
    } else if (dtype == c10::ScalarType::Half) {
        d_tensor.setData(tensor_data, num_elements * sizeof(float) / 2);  // fp16
        d_tensor.setDataType(nn_compiler::ir::DataType::FLOAT16);
        // for float32, bit_width = 16
        d_tensor.setBitWidth(16);
    }

    if (torch_tensor.dim() == 1) {
        Log::NIR::I() << "LSTM bias:" << torch_tensor.sizes();
    } else {
        Log::NIR::I() << "LSTM weight:" << torch_tensor.sizes();
    }
    // set the shape for DTensor
    // the dims of weights of lstm is <=2, but the dims of DTensor is 4
    // (BxCxHxW), so set the B=C=1

    // LSTM weight:[4096, 240]
    // LSTM weight:[4096]
    // some dims of LSTM's weight is 1, set B=C=H=1
    if (torch_tensor.dim() == 1) {
        d_tensor.setTensorShape(STensor(0, 0, 0, torch_tensor.size(0)));
    } else {
        // if the dims of LSTM's weight is 2, only set B=C=1
        d_tensor.setTensorShape(STensor(0, 0, torch_tensor.size(0), torch_tensor.size(1)));
    }
}

/**
 * @brief Get the weights&bias of LSTM op from torch::jit::Node( type is aten::lstm)
 *
 * @param node_lstm aten::lstm node
 */
void AtenLSTM1Builder::getLearnableParameters(const torch::jit::Node* node_lstm, std::vector<DTensor>& weight_tensors,
                                              std::vector<DTensor>& bias_tensors)
{
    // the learnable parameter of aten::lstm contains 8 or 12 tensors in RNNT
    // model, they are all prim::Constant these tensors are input to
    // prim::ListConstrcut and return a tensor[], instead of to unpack these
    // tensor from tensor[], we firstly get the previous prim::ListConstruct Node
    // and then get all the inputs of prim::ListConstruct

    // get the prim::ListConstruct
    auto list_construct = node_lstm->inputs()[3]->node();
    if (list_construct->kind() != c10::prim::ListConstruct) {
        list_construct = node_lstm->inputs()[2]->node();
    }
    // get each input of prim::ListConstruct
    for (auto item : list_construct->inputs()) {
        auto constant_node = item->node();
        assert(constant_node->kind() == c10::prim::Constant);
        assert(constant_node->hasAttribute(c10::attr::value));
        // get the at::Tensor from prim::Constant
        auto torch_tensor = constant_node->t(c10::attr::value);
        // convert at::Tensor to nn_compiler::ir::DTensor
        DTensor d_tensor;
        ptTensor2nncompilerDTensor(torch_tensor, d_tensor);
        if (torch_tensor.dim() == 1) {
            bias_tensors.push_back(d_tensor);
        } else {
            weight_tensors.push_back(d_tensor);
        }
    }
}

}  // namespace frontend
}  // namespace nn_compiler
