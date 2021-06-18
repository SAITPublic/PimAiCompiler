/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file.    matmul_node.hpp
 * @brief.   This is matmul node class
 * @details. This header defines matmul node class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"

#include "ir/include/nn_ir.hpp"
#include "ir/include/nn_node.hpp"

#include "ir/include/ir_includes.hpp"

namespace nn_compiler {
namespace nn_ir {

enum MatrixType { LEFT_MATRIX, RIGHT_MATRIX };

class MatMulNode : public NodeMixin<MatMulNode, NNNode> {
 public:
    explicit MatMulNode(const NodeInfo& node_info, std::unique_ptr<ShiftNode> shift_node)
        : NodeMixin(node_info, NodeType::MATMUL), shift_node_(std::move(shift_node)) {}

    std::string getNodeTypeAsString() const override { return "MatMul"; }

    void             setShiftNode(std::unique_ptr<ShiftNode> shift_node) { shift_node_ = std::move(shift_node); }
    const ShiftNode* getShiftNode() const { return shift_node_.get(); }
    ShiftNode*       getShiftNode() { return shift_node_.get(); }
    nn_ir::Edge&     getLeftMatrixEdge() const;
    nn_ir::Edge&     getRightMatrixEdge() const;

    MatMulNode(const MatMulNode& node);
    MatMulNode(MatMulNode&&) = default;

 private:
    nn_ir::Edge& getMatrixEdge(MatrixType type) const;

    std::unique_ptr<ShiftNode> shift_node_;
};
} // namespace nn_ir
} // namespace nn_compiler
