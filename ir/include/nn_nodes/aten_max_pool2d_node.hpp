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
    explicit AtenMaxPool2dNode(const NodeInfo& node_info, Shape2D kernel_size, Shape2D stride, Pad4 pad,
                               Shape2D dilation, int ceil_mode)
        : NodeMixin(node_info, NodeType::ATENMAXPOOL2D),
          kernel_size_(kernel_size),
          stride_(stride),
          pad_(pad),
          dilation_(dilation),
          ceil_mode_(ceil_mode)
    {
    }

    std::string getNodeTypeAsString(void) const override { return "AtenMaxPool2d"; }
    Shape2D getKernelSize() const { return kernel_size_; }
    Pad4 getPad() const { return pad_; }
    Shape2D getStride() const { return stride_; }
    Shape2D getDilation() const { return dilation_; }
    int getCeilMode() const { return ceil_mode_; }

   private:
    Shape2D kernel_size_;
    Shape2D stride_;
    Pad4 pad_;
    Shape2D dilation_;
    int ceil_mode_;
};  // class AtenMaxPool2dNode

}  // namespace nn_ir
}  // namespace nn_compiler
