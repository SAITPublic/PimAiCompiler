#pragma once

#include "ir/include/all_nodes.hpp"

namespace nn_compiler {

// 1. Determine whether an Op is an aten or prim Op.
// 2. get aten/prim Op type name.
class OpBasicUtil {
 public:
    bool isAtenOp(const nn_ir::Node& node) const;

    bool isPrimOp(const nn_ir::Node& node) const;

    std::string getAtenOpName(const nn_ir::Node& node) const;

    std::string getPrimOpName(const nn_ir::Node& node) const;

    void getOffspring(std::vector<int64_t>& res, nn_ir::NNIR& graph, nn_ir::Node& node, nn_ir::NodeType targetNodeType, int level) const;
};

} // namespace nn_compiler
