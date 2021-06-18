/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/instruction.hpp"
#include "ir/ir_types.hpp"
#include "ir/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {

class MemoryInstruction : public Instruction {
 public:
    explicit MemoryInstruction(INSTR_ID_T id, InstructionType type) : Instruction(id, type) {}

    std::unique_ptr<MemoryInstruction> clone() const& { return std::unique_ptr<MemoryInstruction>(this->cloneImpl()); }
    std::unique_ptr<MemoryInstruction> clone() && {
        return std::unique_ptr<MemoryInstruction>(std::move(*this).cloneImpl());
    }

    template <typename T>
    static bool classof(const Instruction* instr) {
        static_assert(std::is_same<T, MemoryInstruction>::value, "incorrect type");
        return instr->getType() >= InstructionType::MEM_INSTR && instr->getType() <= InstructionType::Last_MEM_INSTR;
    }

 protected:
    MemoryInstruction(const MemoryInstruction&) = default;
    MemoryInstruction(MemoryInstruction&&)      = default;

 private:
    virtual MemoryInstruction* cloneImpl() const& = 0;
    virtual MemoryInstruction* cloneImpl() &&     = 0;
}; // class MemoryInstruction

} // namespace nn_ir
} // namespace nn_compiler
