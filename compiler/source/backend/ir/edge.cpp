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
 * @file    edge.cpp
 * @brief   This is Edge class
 * @details This source defines Edge class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */
#include "ir/edge.hpp"
#include "ir/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {
Edge::Edge(const EdgeInfo& edge_info, EdgeType type)
    : id_(edge_info.id), name_(edge_info.name), graph_(edge_info.graph), type_(type), in_node_id_(edge_info.in_node_id),
      out_node_id_(edge_info.out_node_id) {
    for (unsigned type = 0; type < unsigned(nn_ir::EdgeExecutionStepType::COUNT); ++type) {
        steps_[type] = std::make_unique<EdgeExecutionStep>(
            graph_, graph_.getNextStepId(), ExecutionStepType::EDGE, id_, (nn_ir::EdgeExecutionStepType)type);
    }
}

Edge::Edge(const Edge& other)
    : id_(other.graph_.getNextEdgeId()), name_(other.name_), graph_(other.graph_), type_(other.type_),
      in_node_id_(other.in_node_id_), out_node_id_(other.out_node_id_) {
    for (unsigned type = 0; type < (unsigned)nn_ir::EdgeExecutionStepType::COUNT; ++type) {
        steps_[type] = std::make_unique<EdgeExecutionStep>(
            graph_, graph_.getNextStepId(), ExecutionStepType::EDGE, id_, (nn_ir::EdgeExecutionStepType)type);
    }
}

Edge::~Edge() = default;

void Edge::setId(EDGE_ID_T id) {
    id_ = id;
    for (unsigned type = 0; type < unsigned(nn_ir::NodeExecutionStepType::COUNT); ++type) {
        steps_[type]->setEdgeId(id);
    }
}

Node* Edge::getInNode() const {
    auto id = getInNodeId();
    return id == INVALID_ID ? nullptr : graph_.getNode(id);
}

Node* Edge::getOutNode() const {
    auto id = getOutNodeId();
    return id == INVALID_ID ? nullptr : graph_.getNode(id);
}

} // namespace nn_ir
} // namespace nn_compiler
