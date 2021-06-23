#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

#include "ir/include/ir_includes.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimTupleConstructNode : public NodeMixin<PrimTupleConstructNode, CONTROLNode> {
 public:
    explicit PrimTupleConstructNode(const NodeInfo& node_info)
        : NodeMixin(node_info, NodeType::PRIMTUPLECONSTRUCT) {}

    std::string getNodeTypeAsString() const override { return "PrimTupleConstruct"; }

}; // class PrimTupleConstructNode

} // namespace nn_ir
} // namespace nn_compiler
