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
 * @file.    compute_instr.hpp
 * @brief.   This is ComputeInstruction class
 * @details. This header defines ComputeInstruction class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/instruction.hpp"
#include "ir/include/ir_types.hpp"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {

class ComputeInstruction : public Instruction {
 public:
    explicit ComputeInstruction(INSTR_ID_T id, InstructionType type) : Instruction(id, type) {}

    std::unique_ptr<ComputeInstruction> clone() const& {
        return std::unique_ptr<ComputeInstruction>(this->cloneImpl());
    }
    std::unique_ptr<ComputeInstruction> clone() && {
        return std::unique_ptr<ComputeInstruction>(std::move(*this).cloneImpl());
    }

    template <typename T>
    static bool classof(const Instruction* instr) {
        static_assert(std::is_same<T, ComputeInstruction>::value, "incorrect type");
        return instr->getType() >= InstructionType::COMP_INSTR && instr->getType() <= InstructionType::Last_COMP_INSTR;
    }

    void dump(std::ostream& s) const;

 protected:
    ComputeInstruction(const ComputeInstruction&) = default;
    ComputeInstruction(ComputeInstruction&&)      = default;

 private:
    virtual ComputeInstruction* cloneImpl() const& = 0;
    virtual ComputeInstruction* cloneImpl() &&     = 0;
}; // class ComputeInstruction

} // namespace nn_ir
} // namespace nn_compiler
