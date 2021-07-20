#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenOnesNode : public NodeMixin<AtenOnesNode, NNNode>
{
   public:
    explicit AtenOnesNode(const NodeInfo& node_info, int64_t size, int64_t dtype, int64_t layout, int64_t device)
        : NodeMixin(node_info, NodeType::ATENONES), size_(size), dtype_(dtype), layout_(layout), device_(device)
    {
    }

    std::string getNodeTypeAsString() const override { return "AtenOnes"; }

    int64_t getSize() const { return size_; }
    int64_t getDtype() const { return dtype_; }
    int64_t getLayout() const { return layout_; }
    int64_t getDevice() const { return device_; }

   private:
    int64_t size_;
    int64_t dtype_;
    int64_t layout_;
    int64_t device_;
};  // class AtenOnesNode

}  // namespace nn_ir
}  // namespace nn_compiler
