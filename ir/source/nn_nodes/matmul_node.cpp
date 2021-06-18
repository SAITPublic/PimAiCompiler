/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/include/nn_nodes/matmul_node.hpp"

namespace nn_compiler {
namespace nn_ir {

MatMulNode::MatMulNode(const MatMulNode& node)
    : NodeMixin(node), shift_node_(node.shift_node_ ? node.shift_node_->clone() : nullptr) {}

nn_ir::Edge& MatMulNode::getMatrixEdge(MatrixType type) const {
    Shape4D inputs_dim[2];
    int     index              = 0;
    int     left_matrix_index  = 0;
    int     right_matrix_index = 0;

    for (auto& in_edge : getInEdges()) {
        auto in_blob      = cast<nn_ir::DataEdge>(in_edge).getBlob();
        inputs_dim[index] = in_blob->getShape();
        index++;
    }

    bool has_split = false;
    if (inputs_dim[1].h == 1) {
        has_split = true;
    }

    auto     output_dim = cast<nn_ir::DataEdge>(getFirstOutEdge()).getBlob()->getShape();
    uint32_t output_row = has_split ? output_dim.c : output_dim.h;
    uint32_t second_row = has_split ? inputs_dim[1].c : inputs_dim[1].h;
    uint32_t first_col  = inputs_dim[0].w;
    uint32_t first_row  = has_split ? inputs_dim[0].c : inputs_dim[0].h;

    if (first_col == second_row && first_row == output_row) {
        left_matrix_index  = 0;
        right_matrix_index = 1;
    } else {
        left_matrix_index  = 1;
        right_matrix_index = 0;
    }

    if (type == LEFT_MATRIX) {
        return this->getInEdge(left_matrix_index);
    }

    return this->getInEdge(right_matrix_index);
}

nn_ir::Edge& MatMulNode::getLeftMatrixEdge() const { return getMatrixEdge(LEFT_MATRIX); }

nn_ir::Edge& MatMulNode::getRightMatrixEdge() const { return getMatrixEdge(RIGHT_MATRIX); }

} // namespace nn_ir
} // namespace nn_compiler
