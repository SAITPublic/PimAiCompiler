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
 * @file.    execution_step.hpp
 * @brief.   This is ExecutionStep class
 * @details. This header defines ExecutionStep class.
 * @version. 0.1.
 */

#pragma once

#include "ir/instruction.hpp"
#include "ir/ir_types.hpp"

#include "common/attributes.h"
#include "common/common.hpp"
#include "common/types.hpp"

#include <vector>

namespace nn_compiler {
namespace nn_ir {

class ExecutionStep {
 public:
    /**
     * @brief.      Constructor of ExecutionStep.
     * @details.    This function constructs ExecutionStep
     * @param[in].
     * @param[out].
     * @returns.
     */
    virtual ~ExecutionStep() = 0;
    ExecutionStep& operator=(const ExecutionStep&) = delete;
    ExecutionStep& operator=(ExecutionStep&&) = delete;

    std::unique_ptr<ExecutionStep> clone() const& { return std::unique_ptr<ExecutionStep>(this->cloneImpl()); }
    std::unique_ptr<ExecutionStep> clone() && { return std::unique_ptr<ExecutionStep>(std::move(*this).cloneImpl()); }

    void setInstructions(std::vector<std::unique_ptr<Instruction>> instr) { instrs_ = std::move(instr); }
    const std::vector<std::unique_ptr<Instruction>>& getInstructions() const { return instrs_; }

    STEP_ID_T         getId() const { return id_; }
    ExecutionStepType getType() const { return type_; }

    virtual bool isSync() const = 0;

    virtual bool isLoadStore() const = 0;

    virtual bool isLoad() const = 0;

    virtual const nn_ir::Node* getAttachedNode() const = 0;

    bool isStore() const { return isLoadStore() && !isLoad(); }

    virtual bool isExec() const = 0;

    NO_DISCARD virtual const nn_ir::ExecutionStep& getSyncStep() const = 0;

    virtual void printName(std::ostream& s) const = 0;

    friend bool operator==(const ExecutionStep& lhs, const ExecutionStep& rhs) { return lhs.id_ == rhs.id_; }

 protected:
    ExecutionStep(const NNIR& graph, INSTR_ID_T id, ExecutionStepType type) : graph_(graph), id_(id), type_(type) {}
    ExecutionStep(const ExecutionStep&);
    ExecutionStep(ExecutionStep&&) = default;

    const NNIR&                               graph_;
    STEP_ID_T                                 id_;
    std::vector<std::unique_ptr<Instruction>> instrs_;
    ExecutionStepType                         type_;

 private:
    virtual ExecutionStep* cloneImpl() const& = 0;
    virtual ExecutionStep* cloneImpl() &&     = 0;
}; // class ExecutionStep

inline std::ostream& operator<<(std::ostream& os, const nn_ir::ExecutionStep& step) {
    step.printName(os);
    return os;
}

} // namespace nn_ir
} // namespace nn_compiler
