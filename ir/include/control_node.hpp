#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/node.hpp"

namespace nn_compiler {
namespace nn_ir {

class CONTROLNode : public AbstractNodeMixin<CONTROLNode, Node> {
 public:
    explicit CONTROLNode(const NodeInfo& node_info) : CONTROLNode(node_info, NodeType::CONTROLNode) {}

    void setTargetDevice(const std::string& target_device) { target_device_ = target_device; }

    const std::string& getTargetDevice() { return target_device_; }

 protected:
    CONTROLNode(const CONTROLNode&) = default;
    CONTROLNode(CONTROLNode&&)      = default;

    // constructor for inherited classes
    explicit CONTROLNode(const NodeInfo& node_info, NodeType node_type) : AbstractNodeMixin(node_info, node_type) {}

 private:
    std::string target_device_ = "CPU";
}; // class CONTROLNode

} // namespace nn_ir
} // namespace nn_compiler
