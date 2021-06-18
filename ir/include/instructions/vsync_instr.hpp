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
 * @file.    signal_wait_instr.hpp
 * @brief.   This is VsyncInstruction class
 * @details. This header defines VsyncInstruction class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/misc_instr.hpp"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {

class VsyncInstruction : public MiscInstruction {
 public:
    explicit VsyncInstruction(INSTR_ID_T id, InstructionType type) : MiscInstruction(id, type) {}

    std::unique_ptr<VsyncInstruction> clone() const& { return std::unique_ptr<VsyncInstruction>(this->cloneImpl()); }
    std::unique_ptr<VsyncInstruction> clone() && {
        return std::unique_ptr<VsyncInstruction>(std::move(*this).cloneImpl());
    }

    template <typename T>
    static bool classof(const Instruction* instr) {
        static_assert(std::is_same<std::decay_t<T>, VsyncInstruction>::value, "incorrect type");
        return instr->getType() == InstructionType::VSYNC;
    }

 private:
    VsyncInstruction(const VsyncInstruction&) = default;
    VsyncInstruction(VsyncInstruction&&)      = default;

    VsyncInstruction* cloneImpl() const& override { return new VsyncInstruction(*this); }
    VsyncInstruction* cloneImpl() && override { return new VsyncInstruction(std::move(*this)); }
}; // class VsyncInstruction

} // namespace nn_ir
} // namespace nn_compiler
