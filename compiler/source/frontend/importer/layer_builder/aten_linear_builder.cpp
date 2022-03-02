#include "importer/layer_builder/layer_builder.h"
#include "new_ir/include/common/log.hpp"
#include "new_ir/include/tensors/data_tensor.h"

using nn_compiler::ir::DTensor;
using nn_compiler::ir::STensor;

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
    Log::NIR::I() << "build aten::linear";

    nn_compiler::ir::LayerType type = nn_compiler::ir::LayerType::ATENLINEAR;
    std::string name = "";

    aten_linear_layer_ = std::make_shared<ir::AtenLinearLayer>(name, type);

    // get the weights/bias from jit::Node and add to layer
    std::vector<DTensor> weights;
    std::vector<DTensor> biases;

    getLearnableParameters(node_ref, weights, biases);
    aten_linear_layer_->setWeights(weights);
    aten_linear_layer_->setBiases(biases);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(aten_linear_layer_);

    return layer;
}

/**
 * @brief Convert the FP32/FP16(HALF) type torch at::Tensor to
 * nn_compiler::ir::DTensor
 *
 * @param torch_tensor the input at::Tensor
 * @param d_tensor the convereted DTensor
 */
void AtenLinearBuilder::ptTensor2nncompilerDTensor(at::Tensor torch_tensor, DTensor& d_tensor)
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
        d_tensor.setTensorShape(STensor(0, 0, 0, torch_tensor.size(0)));
    } else {
        d_tensor.setTensorShape(STensor(0, 0, torch_tensor.size(0), torch_tensor.size(1)));
    }
}

void AtenLinearBuilder::getLearnableParameters(const torch::jit::Node* node_ref, std::vector<DTensor>& weight_tensors,
                                               std::vector<DTensor>& bias_tensors)
{
    //  weight
    auto weight_node = node_ref->inputs()[1]->node();
    assert(weight_node->kind() == c10::prim::Constant);
    assert(weight_node->hasAttribute(c10::attr::value));

    auto weight_torch_tensor = weight_node->t(c10::attr::value);
    DTensor weight_d_tensor;
    ptTensor2nncompilerDTensor(weight_torch_tensor, weight_d_tensor);
    weight_tensors.push_back(weight_d_tensor);

    //  bias
    auto bais_node = node_ref->inputs()[2]->node();
    if (bais_node->kind() == c10::prim::Constant && bais_node->hasAttribute(c10::attr::value)) {
        auto bais_torch_tensor = bais_node->t(c10::attr::value);
        DTensor bais_d_tensor;
        ptTensor2nncompilerDTensor(bais_torch_tensor, bais_d_tensor);
        bias_tensors.push_back(bais_d_tensor);
    }
}

}  // namespace frontend
}  // namespace nn_compiler
