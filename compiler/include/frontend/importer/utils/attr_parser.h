#pragma once

#include <assert.h>
#include <torch/script.h>
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace frontend
{
class AttrParser
{
   public:
    AttrParser() = default;

    int64_t getIntAttr(const torch::jit::Node* node, c10::Symbol symbol);

    std::vector<int64_t> getIntArrayAttr(const torch::jit::Node* node, c10::Symbol symbol);

    double getFP64Attr(const torch::jit::Node* node, c10::Symbol symbol);

    std::string getStrAttr(const torch::jit::Node* node, c10::Symbol symbol);

    at::Tensor getTensorAttr(const torch::jit::Node* node, c10::Symbol symbol);

    ~AttrParser() = default;
};

std::shared_ptr<nn_compiler::ir::DTensor> getDTensorData(const torch::jit::Node* node_constant);

void ptTensor2DTensor(at::Tensor torch_tensor, std::shared_ptr<nn_compiler::ir::DTensor> d_tensor);

}  // namespace frontend
}  // namespace nn_compiler
