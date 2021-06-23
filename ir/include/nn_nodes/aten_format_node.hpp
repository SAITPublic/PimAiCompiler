#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenFormatNode : public NodeMixin<AtenFormatNode, NNNode> {
 public:
    explicit AtenFormatNode(const NodeInfo& node_info,  std::string assembly_format)
        : NodeMixin(node_info, NodeType::ATENFORMAT), assembly_format_(assembly_format) {}

    std::string getNodeTypeAsString() const override { return "AtenFormat"; }
    std::string getAssemblyFormat() const { return assembly_format_; }

 private:
    std::string assembly_format_;
}; // class AtenFormatNode

} // namespace nn_ir
} // namespace nn_compiler
