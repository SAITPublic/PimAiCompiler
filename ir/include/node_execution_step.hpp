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
 * @file.    node_execution_step.hpp
 * @brief.   This is NodeExecutionStep class
 * @details. This header defines NodeExecutionStep class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/common/log.hpp"
#include "ir/include/execution_step.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class NodeExecutionStep : public ExecutionStep {
 public:
    /**
     * @brief.      Constructor of NodeExecutionStep.
     * @details.    This function constructs NodeExecutionStep
     * @param[in].
     * @param[out].
     * @returns.
     */
    NodeExecutionStep(
        const NNIR& graph, INSTR_ID_T id, ExecutionStepType type, NODE_ID_T node_id, NodeExecutionStepType exec_type)
        : ExecutionStep(graph, id, type), node_id_(node_id), exec_type_(exec_type) {}

    NODE_ID_T             getNodeId() const { return node_id_; }
    Node*                 getNode() const;
    NodeExecutionStepType getNodeStepType() const { return exec_type_; }
    void                  setNodeId(NODE_ID_T id) { node_id_ = id; }

    std::unique_ptr<NodeExecutionStep> clone() const& { return std::unique_ptr<NodeExecutionStep>(this->cloneImpl()); }
    std::unique_ptr<NodeExecutionStep> clone() && {
        return std::unique_ptr<NodeExecutionStep>(std::move(*this).cloneImpl());
    }

    NodeExecutionStep(const NodeExecutionStep& other)
        : ExecutionStep(other), node_id_(INVALID_ID), exec_type_(other.exec_type_) {}
    NodeExecutionStep(NodeExecutionStep&&) = default;

    template <typename T>
    static bool classof(const ExecutionStep* step) {
        static_assert(std::is_same<T, NodeExecutionStep>::value, "incorrect type");
        return step->getType() == ExecutionStepType::NODE;
    }

    static NodeExecutionStepType getSyncType(NodeExecutionStepType type) {
        switch (type) {
            case nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_START:
                return nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_SYNC;
            case nn_ir::NodeExecutionStepType::EXEC_START:
                return nn_ir::NodeExecutionStepType::EXEC_SYNC;
            default:
                Log::IR::E() << "unknown node step type";
                return type;
        }
    }

    bool isSync() const override {
        switch (exec_type_) {
            case nn_ir::NodeExecutionStepType::EXEC_SYNC:
            case nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_SYNC:
                return true;
            default:
                return false;
        }
    }

    bool isLoadStore() const override { return !isExec(); }

    bool isLoad() const override { return !isExec(); }

    bool isExec() const override {
        switch (exec_type_) {
            case nn_ir::NodeExecutionStepType::EXEC_START:
            case nn_ir::NodeExecutionStepType::EXEC_SYNC:
                return true;
            default:
                return false;
        }
    }

    const nn_ir::Node* getAttachedNode() const override { return getNode(); }

    NO_DISCARD const nn_ir::ExecutionStep& getSyncStep() const override;
    void                                   printName(std::ostream& s) const override;

 protected:
    NODE_ID_T             node_id_;
    NodeExecutionStepType exec_type_;

 private:
    NodeExecutionStep* cloneImpl() const& override { return new NodeExecutionStep(*this); }
    NodeExecutionStep* cloneImpl() && override { return new NodeExecutionStep(std::move(*this)); }
}; // class ExecutionStep

} // namespace nn_ir
} // namespace nn_compiler
