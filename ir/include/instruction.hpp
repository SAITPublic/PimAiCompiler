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

#include "ir/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class Instruction {
 public:
    Instruction(INSTR_ID_T id, InstructionType type) : id_(id), type_(type) {}

    virtual ~Instruction() = 0;

    Instruction& operator=(const Instruction&) = delete;
    Instruction& operator=(Instruction&&) = delete;

    std::unique_ptr<Instruction> clone() const& { return std::unique_ptr<Instruction>(this->cloneImpl()); }
    std::unique_ptr<Instruction> clone() && { return std::unique_ptr<Instruction>(std::move(*this).cloneImpl()); }

    INSTR_ID_T      getId() const { return id_; }
    InstructionType getType() const { return type_; }

    virtual void dump(std::ostream& s) const;

 protected:
    Instruction(const Instruction&) = default;
    Instruction(Instruction&&)      = default;

 private:
    virtual Instruction* cloneImpl() const& = 0;
    virtual Instruction* cloneImpl() &&     = 0;

    INSTR_ID_T      id_;
    InstructionType type_;
}; // class Instruction

inline std::ostream& operator<<(std::ostream& s, const Instruction& insn) {
    insn.dump(s);
    return s;
}

} // namespace nn_ir
} // namespace nn_compiler
