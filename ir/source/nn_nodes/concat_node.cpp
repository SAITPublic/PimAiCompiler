/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file    concat_node.cpp
 * @brief   This is ConcatNode class
 * @details This source defines ConcatNode class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */

#include "ir/nn_nodes/concat_node.hpp"

#include "common/algorithm_ext.hpp"
#include "ir/common/log.hpp"
#include "ir/nn_node_type_traits.hpp"

namespace nn_compiler {
namespace nn_ir {
/** @brief Returns "true" if a Reshape node was merged into this Concat node
 * @example Concat node may implicitly contain Reshape semantics:
 * [n:1 c:7 h:1 w:1] \
 * [n:1 c:7 h:1 w:1] —> (Concat) —> [n:1 c:1 h:7 w:3]
 * [n:1 c:7 h:1 w:1] /
 */
bool ConcatNode::isReshaping() const {
    const auto concatenated_shape = nn_ir::getFirstOfmShape(*this);
    return estd::any_of(getInEdges<nn_ir::DataEdge>(), [&concatenated_shape, this](const auto& in_edge) {
        const auto in_shape = in_edge.getBlob()->getShape();
        return estd::any_of(nn_ir::AllAxes4D, [&concatenated_shape, &in_shape, this](const auto axis) {
            return axis != this->axis_ && in_shape[axis] != concatenated_shape[axis];
        });
    });
}

// TODO(a.puschin): modify this, add a map of 4D offsets for inputs into ConcatNode (see CRANC-680 for detail)
nn_ir::Coord4D ConcatNode::getConcatenationOffsetFor(const nn_ir::Node& predecessor_node) const {
    nn_ir::Coord4D concatenation_offset = {{.n = 0, .c = 0, .h = 0, .w = 0}};
    for (const auto& incoming_edge : getInEdges<nn_ir::DataEdge>()) {
        if (incoming_edge.getInNode() == &predecessor_node) {
            return concatenation_offset;
        }
        const auto* input_blob = incoming_edge.getBlob();
        // All IFMs are getting aligned, except the last one
        const auto alignment_unit_along_concatenation_axis =
            std::max(input_blob->getSizeAlignment()[axis_], input_blob->getPositionAlignment()[axis_]);
        concatenation_offset[axis_] += alignUp(input_blob->getShape()[axis_], alignment_unit_along_concatenation_axis);
    }
    Log::IR::E() << predecessor_node << " isn't a predecessor of " << *this;
}
} // namespace nn_ir
} // namespace nn_compiler
