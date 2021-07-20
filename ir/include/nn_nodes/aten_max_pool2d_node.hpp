#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenMaxPool2dNode : public NodeMixin<AtenMaxPool2dNode, NNNode>
{
   public:
    explicit AtenMaxPool2dNode(const NodeInfo& node_info, Shape2D kernel_size, Pad4 pad, Shape2D stride,
                               Shape2D dilation, int return_indices)
        : NodeMixin(node_info, NodeType::ATENMAXPOOL2D),
          kernel_size_(kernel_size),
          pad_(pad),
          stride_(stride),
          dilation_(dilation),
          return_indices_(return_indices)
    {
    }

    std::string getNodeTypeAsString(void) const override { return "AtenMaxPool2d"; }
    Shape2D getKernelSize() const { return kernel_size_; }
    Pad4 getPad() const { return pad_; }
    Shape2D getStride() const { return stride_; }
    Shape2D getDilation() const { return dilation_; }
    int getReturnIndices() const { return return_indices_; }

   private:
    Shape2D kernel_size_;
    Pad4 pad_;
    Shape2D stride_;
    Shape2D dilation_;
    int return_indices_;
};  // class AtenMaxPool2dNode

}  // namespace nn_ir
}  // namespace nn_compiler
