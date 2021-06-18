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
 * @file.    edge_execution_step.hpp
 * @brief.   This is EdgeExecutionStep class
 * @details. This header defines EdgeExecutionStep class.
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

class EdgeExecutionStep : public ExecutionStep {
 public:
    /**
     * @brief.      Constructor of EdgeExecutionStep.
     * @details.    This function constructs EdgeExecutionStep
     * @param[in].
     * @param[out].
     * @returns.
     */
    EdgeExecutionStep(
        const NNIR& graph, STEP_ID_T id, ExecutionStepType type, EDGE_ID_T edge_id, EdgeExecutionStepType exec_type)
        : ExecutionStep(graph, id, type), edge_id_(edge_id), exec_type_(exec_type) {}

    EDGE_ID_T             getEdgeId() const { return edge_id_; }
    Edge*                 getEdge() const;
    EdgeExecutionStepType getEdgeStepType() const { return exec_type_; }
    void                  setEdgeId(EDGE_ID_T id) { edge_id_ = id; }

    std::unique_ptr<EdgeExecutionStep> clone() const& { return std::unique_ptr<EdgeExecutionStep>(this->cloneImpl()); }
    std::unique_ptr<EdgeExecutionStep> clone() && {
        return std::unique_ptr<EdgeExecutionStep>(std::move(*this).cloneImpl());
    }

    EdgeExecutionStep(const EdgeExecutionStep& other)
        : ExecutionStep(other), edge_id_(INVALID_ID), exec_type_(other.exec_type_) {}
    EdgeExecutionStep(EdgeExecutionStep&&) = default;

    template <typename T>
    static bool classof(const ExecutionStep* step) {
        static_assert(std::is_same<T, EdgeExecutionStep>::value, "incorrect type");
        return step->getType() == ExecutionStepType::EDGE;
    }

    static EdgeExecutionStepType getSyncType(EdgeExecutionStepType type) {
        switch (type) {
            case nn_ir::EdgeExecutionStepType::LOAD_START:
                return nn_ir::EdgeExecutionStepType::LOAD_SYNC;
            case nn_ir::EdgeExecutionStepType::STORE_START:
                return nn_ir::EdgeExecutionStepType::STORE_SYNC;
            default:
                Log::IR::E() << "unknown edge step type";
                return type;
        }
    }

    bool isSync() const override {
        switch (exec_type_) {
            case EdgeExecutionStepType::LOAD_SYNC:
            case EdgeExecutionStepType::STORE_SYNC:
                return true;
            default:
                return false;
        }
    }

    bool isLoadStore() const override { return true; }

    bool isLoad() const override {
        switch (exec_type_) {
            case EdgeExecutionStepType::LOAD_START:
            case EdgeExecutionStepType::LOAD_SYNC:
                return true;
            default:
                return false;
        }
    }

    bool isStore() const {
        switch (exec_type_) {
            case EdgeExecutionStepType::STORE_START:
            case EdgeExecutionStepType::STORE_SYNC:
                return true;
            default:
                return false;
        }
    }

    bool isExec() const override { return false; }

    const nn_ir::Node* getAttachedNode() const override;

    NO_DISCARD const nn_ir::ExecutionStep& getSyncStep() const override;
    void                                   printName(std::ostream& s) const override;

 protected:
    EDGE_ID_T             edge_id_;
    EdgeExecutionStepType exec_type_;

 private:
    EdgeExecutionStep* cloneImpl() const& override { return new EdgeExecutionStep(*this); }
    EdgeExecutionStep* cloneImpl() && override { return new EdgeExecutionStep(std::move(*this)); }
}; // class ExecutionStep

} // namespace nn_ir
} // namespace nn_compiler
