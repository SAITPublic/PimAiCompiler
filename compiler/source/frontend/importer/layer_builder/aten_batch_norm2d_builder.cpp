#include "importer/layer_builder/layer_builder.h"
#include "importer/utils/attr_parser.h"
#include "new_ir/include/common/log.hpp"
#include "new_ir/include/tensors/data_tensor.h"

using nn_compiler::ir::DTensor;
using nn_compiler::ir::STensor;

namespace nn_compiler
{
namespace frontend
{
std::shared_ptr<ir::NNLayer> AtenBatchNorm2dBuilder::buildLayer(const torch::jit::Node* node_ref)
{
    Log::NIR::I() << "build aten::batch_norm";

    nn_compiler::ir::LayerType type = nn_compiler::ir::LayerType::ATENBATCHNORM2D;
    std::string name = "";

    aten_batch_norm2d_layer_ = std::make_shared<ir::AtenBatchNorm2dLayer>(name, type);

    // get the weights/bias from jit::Node and add to layer
    std::vector<DTensor> weights;
    std::vector<DTensor> bias;

    getLearnableParameters(node_ref, weights, bias);
    aten_batch_norm2d_layer_->setWeights(weights);
    aten_batch_norm2d_layer_->setBiases(bias);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(aten_batch_norm2d_layer_);
    return layer;
}

/**
 * @brief Convert the FP32/FP16(HALF) type torch at::Tensor to
 * nn_compiler::ir::DTensor
 *
 * @param torch_tensor the input at::Tensor
 * @param d_tensor the convereted DTensor
 */
void AtenBatchNorm2dBuilder::ptTensor2nncompilerDTensor(at::Tensor torch_tensor, DTensor& d_tensor)
{
    c10::ScalarType dtype = torch_tensor.scalar_type();
    // dtype of the weights of bn_2d are float32 or float16/half
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

    // set the shape for DTensor
    // the dims of weights of bn_2d is <=2, but the dims of DTensor is 4 (BxCxHxW)
    // dim == 2, B=C=1
    // bn_2d.weight (Tensor) the learnable weights of the module of shape (B, C,kernel_size[0], kernel_size[1])
    // dim == 1,B=C=H=1

    if (torch_tensor.dim() == 1) {
        d_tensor.setTensorShape(STensor(0, 0, 0, torch_tensor.size(0)));
    } else {
        d_tensor.setTensorShape(STensor(0, 0, torch_tensor.size(0), torch_tensor.size(1)));
    }
}

/**
 * @brief Get the weights&bias of bn_2d op from torch::jit::Node( type is aten::bn_2d)
 *
 * @param node_bn_2d
 */
void AtenBatchNorm2dBuilder::getLearnableParameters(const torch::jit::Node* node_bn_2d,
                                                    std::vector<DTensor>& weight_tensors,
                                                    std::vector<DTensor>& bias_tensors)
{
    // get weights
    auto weight_node = node_bn_2d->inputs()[1]->node();
    assert(weight_node->kind() == c10::prim::Constant);
    assert(weight_node->hasAttribute(c10::attr::value));
    auto weight_tensor = weight_node->t(c10::attr::value);
    // convert at::Tensor to nn_compiler::ir::DTensor
    DTensor weight_d_tensor;
    ptTensor2nncompilerDTensor(weight_tensor, weight_d_tensor);
    weight_tensors.push_back(weight_d_tensor);

    // get bias
    auto bias_node = node_bn_2d->inputs()[2]->node();
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
