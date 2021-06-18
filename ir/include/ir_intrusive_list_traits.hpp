/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/include/ir_types.hpp"
#include "ir/include/node.hpp"

#pragma once

namespace nn_compiler::nn_ir {

template <typename GraphT> // template is necessary to use methods of graph without previous declaration of graph
struct NNIRIntrusiveListTraits {
    static void onAddNode(Node& node) { GraphT::updateNodeDataOnAdd(node.getGraph(), node); }
    static void onDeleteNode(Node& node) { GraphT::updateNodeDataOnDelete(node.getGraph(), node); }
    static void onMoveNodes(Node& first_moved, Node& last_moved) {
        GraphT::updateNodeDataOnMove(first_moved.getGraph(), first_moved, last_moved);
    }

    constexpr static bool isListOwning() { return true; }

    static std::string getPrintableNode(const Node& node) {
        return "Node #" + std::to_string(node.getId()) + " \"" + node.getName() +
               "\" Type: " + node.getNodeTypeAsString();
    }
};

} // namespace nn_compiler::nn_ir
