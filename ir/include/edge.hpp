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
 * @file.    edge.hpp
 * @brief.   This is Edge class
 * @details. This header defines Edge class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"
#include "ir/edge_execution_step.hpp"
#include "ir/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class Edge {
 public:
    virtual ~Edge() = 0;

    Edge& operator=(const Edge&) = delete;
    Edge& operator=(Edge&&) = delete;

    std::unique_ptr<Edge> clone() const& { return std::unique_ptr<Edge>(this->cloneImpl()); }
    std::unique_ptr<Edge> clone() && { return std::unique_ptr<Edge>(std::move(*this).cloneImpl()); }

    EDGE_ID_T getId() const { return id_; }

    std::string getName() const { return name_; }
    EdgeType    getEdgeType() const { return type_; }
    NODE_ID_T   getInNodeId() const { return in_node_id_; }
    Node*       getInNode() const;
    NODE_ID_T   getOutNodeId() const { return out_node_id_; }
    Node*       getOutNode() const;

    bool isGraphInput() const { return in_node_id_ == INVALID_ID; }
    bool isGraphOutput() const { return out_node_id_ == INVALID_ID; }

    const NNIR& getGraph() const { return graph_; }

    const EdgeExecutionStep& getStep(nn_ir::EdgeExecutionStepType type) const { return *steps_[unsigned(type)]; }

    const std::vector<std::unique_ptr<Instruction>>& getInstructions(nn_ir::EdgeExecutionStepType type) const {
        return steps_[unsigned(type)]->getInstructions();
    }

    void setId(EDGE_ID_T id);

    void setName(std::string name) { name_ = name; }
    void setEdgeType(EdgeType type) { type_ = type; }
    void setInNodeId(NODE_ID_T id) { in_node_id_ = id; }
    void setOutNodeId(NODE_ID_T id) { out_node_id_ = id; }

    void setStep(nn_ir::EdgeExecutionStepType type, std::unique_ptr<EdgeExecutionStep>& pstep) {
        steps_[unsigned(type)] = std::move(pstep);
    }

    void setInstructions(nn_ir::EdgeExecutionStepType type, std::vector<std::unique_ptr<Instruction>> instr) {
        steps_[unsigned(type)]->setInstructions(std::move(instr));
    }

 protected:
    explicit Edge(const EdgeInfo& edge_info, EdgeType type);
    Edge(const Edge&);
    Edge(Edge&&) = default;

 private:
    virtual Edge* cloneImpl() const& = 0;
    virtual Edge* cloneImpl() &&     = 0;

    EDGE_ID_T   id_;
    std::string name_;
    const NNIR& graph_;
    EdgeType    type_;
    NODE_ID_T   in_node_id_;
    NODE_ID_T   out_node_id_;

    std::unique_ptr<EdgeExecutionStep> steps_[unsigned(nn_ir::EdgeExecutionStepType::COUNT)];
}; // class Edge

inline std::ostream& operator<<(std::ostream& os, const nn_ir::Edge& edge) { return os << "edge #" << edge.getId(); }

} // namespace nn_ir
} // namespace nn_compiler
