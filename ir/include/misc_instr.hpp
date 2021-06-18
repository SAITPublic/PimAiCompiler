/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or miscr language in any form by any means,
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

class MiscInstruction : public Instruction {
 public:
    explicit MiscInstruction(INSTR_ID_T id, InstructionType type) : Instruction(id, type) {}

    std::unique_ptr<MiscInstruction> clone() const& { return std::unique_ptr<MiscInstruction>(this->cloneImpl()); }
    std::unique_ptr<MiscInstruction> clone() && {
        return std::unique_ptr<MiscInstruction>(std::move(*this).cloneImpl());
    }

    template <typename T>
    static bool classof(const Instruction* instr) {
        static_assert(std::is_same<T, MiscInstruction>::value, "incorrect type");
        return instr->getType() >= InstructionType::MISC_INSTR && instr->getType() <= InstructionType::Last_MISC_INSTR;
    }

    void dump(std::ostream& s) const;

 protected:
    MiscInstruction(const MiscInstruction&) = default;
    MiscInstruction(MiscInstruction&&)      = default;

 private:
    virtual MiscInstruction* cloneImpl() const& = 0;
    virtual MiscInstruction* cloneImpl() &&     = 0;
}; // class Node

} // namespace nn_ir
} // namespace nn_compiler
