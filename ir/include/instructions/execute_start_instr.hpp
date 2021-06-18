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
 * @file.    execute_start_instr.hpp
 * @brief.   This is ExecuteStartInstruction class
 * @details. This header defines ExecuteStartInstruction class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/compute_instr.hpp"
#include "ir/include/ir_types.hpp"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {

class ExecuteStartInstruction : public ComputeInstruction {
 public:
    explicit ExecuteStartInstruction(INSTR_ID_T id, InstructionType type) : ComputeInstruction(id, type) {}

    std::unique_ptr<ExecuteStartInstruction> clone() const& {
        return std::unique_ptr<ExecuteStartInstruction>(this->cloneImpl());
    }
    std::unique_ptr<ExecuteStartInstruction> clone() && {
        return std::unique_ptr<ExecuteStartInstruction>(std::move(*this).cloneImpl());
    }

    template <typename T>
    static bool classof(const Instruction* instr) {
        static_assert(std::is_same<std::decay_t<T>, ExecuteStartInstruction>::value, "incorrect type");
        return instr->getType() == InstructionType::COMPUTE_START;
    }

 private:
    ExecuteStartInstruction(const ExecuteStartInstruction&) = default;
    ExecuteStartInstruction(ExecuteStartInstruction&&)      = default;

    ExecuteStartInstruction* cloneImpl() const& override { return new ExecuteStartInstruction(*this); }
    ExecuteStartInstruction* cloneImpl() && override { return new ExecuteStartInstruction(std::move(*this)); }
}; // class ExecuteStartInstruction

} // namespace nn_ir
} // namespace nn_compiler
