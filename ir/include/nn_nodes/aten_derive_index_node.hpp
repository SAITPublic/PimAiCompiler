
#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenDeriveIndexNode : public NodeMixin<AtenDeriveIndexNode, NNNode> {
 public:
    explicit AtenDeriveIndexNode(const NodeInfo& node_info, int64_t start, int64_t step)
        : NodeMixin(node_info, NodeType::ATENZEROSLIKE), start_(start), step_(step) {}

    std::string getNodeTypeAsString() const override { return "AtenDeriveIndex"; }

    const int64_t getStatr() const {return start_; }
    const int64_t getStep() const {return step_; }

 private:
    int64_t start_;
    int64_t step_;
}; // class AtenDeriveIndexNode

} // namespace nn_ir
} // namespace nn_compiler
