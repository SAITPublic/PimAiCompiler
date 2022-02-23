#include "importer/layer_builder/layer_builder.h"
#include "new_ir/include/common/log.hpp"
#include "new_ir/include/tensors/data_tensor.h"

using nn_compiler::ir::DTensor;
using nn_compiler::ir::STensor;

namespace nn_compiler
{
namespace frontend
{
std::shared_ptr<ir::NNLayer> AtenConv2dBuilder::buildLayer(const torch::jit::Node* node_ref)
{
    Log::NIR::I() << "build aten::conv2d";

    std::string type = "aten::conv2d";
    std::string name = "";

    aten_conv2d_layer_ = std::make_shared<ir::AtenConv2dLayer>(name, type);
    const torch::jit::Node* node_conv2d = node_ref;

    // get the weights/bias from jit::Node and add to layer
    std::vector<DTensor> weights;
    std::vector<DTensor> bias;

    getLearnableParameters(node_conv2d, weights, bias);
    aten_conv2d_layer_->setWeights(weights);
    aten_conv2d_layer_->setBiases(bias);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(aten_conv2d_layer_);
    return layer;
}

/**
 * @brief Convert the FP32/FP16(HALF) type torch at::Tensor to
 * nn_compiler::ir::DTensor
 *
 * @param torch_tensor the input at::Tensor
 * @param d_tensor the convereted DTensor
 */
void AtenConv2dBuilder::ptTensor2nncompilerDTensor(at::Tensor torch_tensor, DTensor& d_tensor)
{
    c10::ScalarType dtype = torch_tensor.scalar_type();
    // dtype of the weights of conv2d are float32 or float16/half
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

    // set the shape for DTenso
    /**
     * For nn.Conv2d, weight: dim=4, bias; dim=1
     * example:  (conv_l1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
     *
     * conv_l1.weight  torch.Size([16, 1, 3, 3])
     * conv_l1.bias  torch.Size([16])
     *
     */
    assert(torch_tensor.dim() == 1 || torch_tensor.dim() == 4);
    if (torch_tensor.dim() == 1) {
        // bias
        d_tensor.setTensorShape(STensor(0, 0, 0, torch_tensor.size(0)));
        DLOG(INFO) << "conv2d: bias: " << torch_tensor.sizes();
    } else if (torch_tensor.dim() == 4) {
        // weight
        d_tensor.setTensorShape(
            STensor(torch_tensor.size(0), torch_tensor.size(1), torch_tensor.size(2), torch_tensor.size(3)));
        DLOG(INFO) << "conv2d: weight: " << torch_tensor.sizes();
    }
}

/**
 * @brief Get the weights&bias of conv2d op from torch::jit::Node( type is aten::conv2d)
 *
 * @param node_conv2d
 */
void AtenConv2dBuilder::getLearnableParameters(const torch::jit::Node* node_conv2d,
                                               std::vector<DTensor>& weight_tensors, std::vector<DTensor>& bias_tensors)
{
    // get weights
    auto weight_node = node_conv2d->inputs()[1]->node();
    assert(weight_node->kind() == c10::prim::Constant);
    assert(weight_node->hasAttribute(c10::attr::value));
    auto weight_tensor = weight_node->t(c10::attr::value);
    // convert at::Tensor to nn_compiler::ir::DTensor
    DTensor weight_d_tensor;
    ptTensor2nncompilerDTensor(weight_tensor, weight_d_tensor);
    weight_tensors.push_back(weight_d_tensor);

    // get bias
    auto bias_node = node_conv2d->inputs()[2]->node();
    assert(bias_node->kind() == c10::prim::Constant);
    assert(bias_node->hasAttribute(c10::attr::value));
    auto bias_tensor = bias_node->t(c10::attr::value);
    // convert at::Tensor to nn_compiler::ir::DTensor
    DTensor bias_d_tensor;
    ptTensor2nncompilerDTensor(bias_tensor, bias_d_tensor);
    bias_tensors.push_back(bias_d_tensor);
}

}  // namespace frontend
}  // namespace nn_compiler
