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
 * @file    nn_ir.cpp
 * @brief   This is NNIR class
 * @details This source defines NNIR class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */

#include "ir/include/nn_ir.hpp"
#include "ir/include/common/log.hpp"
#include "ir/include/nn_node_type_traits.hpp"
#include "ir/include/nn_nodes/concat_node.hpp"
#include "ir/include/v_nodes/vconcat_node.hpp"
#include "ir/include/v_nodes/vsplit_node.hpp"

#include "ir/include/global_nodes/global_concat_node.hpp"
#include "ir/include/global_nodes/global_split_node.hpp"

#include "common/include/algorithm_ext.hpp"

namespace nn_compiler {
namespace nn_ir {

std::vector<const nn_ir::Node*> NNIR::findMultipleInputEdgeNodes(iterator_range<nn_ir::NNIR::NodeIt> nodes) {
    std::vector<const nn_ir::Node*> ret;
    estd::transform_if(
        nodes.begin(),
        nodes.end(),
        std::back_inserter(ret),
        [](const nn_ir::Node& node) { return &node; },
        [](const nn_ir::Node& node) { return node.getPredecessorsNum() > 1; });
    return ret;
}

std::vector<const nn_ir::Node*> NNIR::findMultipleOutputEdgeNodes(iterator_range<nn_ir::NNIR::NodeIt> nodes) {
    std::vector<const nn_ir::Node*> ret;
    estd::transform_if(
        nodes.begin(),
        nodes.end(),
        std::back_inserter(ret),
        [](const nn_ir::Node& node) { return &node; },
        [](const nn_ir::Node& node) { return node.getOutEdgeIds().size() > 1; });
    return ret;
}

std::vector<nn_ir::Edge*> NNIR::findInEdges(iterator_range<nn_ir::NNIR::EdgeIt> edges) {
    std::vector<nn_ir::Edge*> ret;
    estd::transform_if(
        edges.begin(),
        edges.end(),
        std::back_inserter(ret),
        [](nn_ir::Edge& edge) { return &edge; },
        [](nn_ir::Edge& edge) { return edge.isGraphInput(); });
    return ret;
}

std::vector<nn_ir::Edge*> NNIR::findOutEdges(iterator_range<nn_ir::NNIR::EdgeIt> edges) {
    std::vector<nn_ir::Edge*> ret;
    estd::transform_if(
        edges.begin(),
        edges.end(),
        std::back_inserter(ret),
        [](nn_ir::Edge& edge) { return &edge; },
        [](nn_ir::Edge& edge) { return edge.isGraphOutput(); });
    return ret;
}

void NNIR::addBlob(std::unique_ptr<Blob> blob) {
    BLOB_ID_T id = blob->getId();
    Log::IR::E_IF(estd::contains(blobs_, id)) << "Already existing blob #" << id;
    blobs_[id] = std::move(blob);
}

NodesList::iterator NNIR::deleteNode(const Node& target) {
    auto pos = nodes_.getNodeIterator(target);
    return nodes_.erase(pos);
}

const nn_ir::Node* NNIR::getIfmNodeForPropagation(const NODE_ID_T node_id) const {
    if (node_id == INVALID_ID)
        return nullptr;
    auto node = getNode(node_id);
    return (isa<nn_ir::ConcatNode>(node) || isa<nn_ir::VSplitNode>(node) || isa<nn_ir::GlobalSplitNode>(node))
               ? node
               : nullptr;
}

static bool propagatesOfmMemType(const Node& node) {
    return isa<nn_ir::ConcatNode>(node) || isa<nn_ir::VConcatNode>(node) || isa<nn_ir::GlobalConcatNode>(node);
}

const nn_ir::Node* NNIR::getOfmNodeForPropagation(const NODE_ID_T node_id) const {
    if (node_id == INVALID_ID) {
        return nullptr;
    }
    auto node = getNode(node_id);
    return nn_ir::propagatesOfmMemType(*node) ? node : nullptr;
}
} // namespace nn_ir
} // namespace nn_compiler
