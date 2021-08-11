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
    explicit AtenOnesNode(const NodeInfo& node_info, int64_t dtype, int64_t layout, std::string device, int pin_memory)
        : NodeMixin(node_info, NodeType::ATENONES), dtype_(dtype), layout_(layout), device_(device), pin_memory_(pin_memory)
    {
    }

    std::string getNodeTypeAsString() const override { return "AtenOnes"; }

    void setDtype(int64_t dtype) { dtype_ = dtype; }

    int64_t getDtype() const { return dtype_; }

    void setLayout(int layout) { layout_ = layout; }

    int getLayout() const { return layout_; }

    void setDevice(std::string device) { device_ = device; }

    std::string getDevice() const { return device_; }

    void setPinMemory(int pin_memory) { pin_memory_ = pin_memory; }

    int getPinMemory() const { return pin_memory_; }

   private:
    int64_t dtype_      = INT64_MIN;
    int64_t layout_     = INT64_MIN;
    std::string device_ = "";
    int pin_memory_     = INT32_MIN;
};  // class AtenOnesNode

}  // namespace nn_ir
}  // namespace nn_compiler
