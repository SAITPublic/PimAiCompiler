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
 * @file.    execute_sync_instr.hpp
 * @brief.   This is ExecuteSyncInstruction class
 * @details. This header defines ExecuteSyncInstruction class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/compute_instr.hpp"
#include "ir/ir_types.hpp"
#include "ir/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {

class ExecuteSyncInstruction : public ComputeInstruction {
 public:
    explicit ExecuteSyncInstruction(INSTR_ID_T id, InstructionType type, INSTR_ID_T start_id)
        : ComputeInstruction(id, type), start_id_(start_id) {}

    INSTR_ID_T getStartId() const { return start_id_; }

    std::unique_ptr<ExecuteSyncInstruction> clone() const& {
        return std::unique_ptr<ExecuteSyncInstruction>(this->cloneImpl());
    }
    std::unique_ptr<ExecuteSyncInstruction> clone() && {
        return std::unique_ptr<ExecuteSyncInstruction>(std::move(*this).cloneImpl());
    }

    template <typename T>
    static bool classof(const Instruction* instr) {
        static_assert(std::is_same<std::decay_t<T>, ExecuteSyncInstruction>::value, "incorrect type");
        return instr->getType() == InstructionType::COMPUTE_SYNC;
    }

    void dump(std::ostream& s) const override;

 private:
    ExecuteSyncInstruction(const ExecuteSyncInstruction&) = default;
    ExecuteSyncInstruction(ExecuteSyncInstruction&&)      = default;

    ExecuteSyncInstruction* cloneImpl() const& override { return new ExecuteSyncInstruction(*this); }
    ExecuteSyncInstruction* cloneImpl() && override { return new ExecuteSyncInstruction(std::move(*this)); }

    INSTR_ID_T start_id_;
}; // class ExecuteSyncInstruction

} // namespace nn_ir
} // namespace nn_compiler
