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
 * @brief.   This is SignalWaitInstruction class
 * @details. This header defines SignalWaitInstruction class.
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

class SignalWaitInstruction : public MiscInstruction {
 public:
    explicit SignalWaitInstruction(INSTR_ID_T id, InstructionType type, std::vector<INSTR_ID_T> send_ids)
        : MiscInstruction(id, type), send_ids_(send_ids) {}

    std::vector<INSTR_ID_T> getSendIds() const { return send_ids_; }

    std::unique_ptr<SignalWaitInstruction> clone() const& {
        return std::unique_ptr<SignalWaitInstruction>(this->cloneImpl());
    }
    std::unique_ptr<SignalWaitInstruction> clone() && {
        return std::unique_ptr<SignalWaitInstruction>(std::move(*this).cloneImpl());
    }

    template <typename T>
    static bool classof(const Instruction* instr) {
        static_assert(std::is_same<std::decay_t<T>, SignalWaitInstruction>::value, "incorrect type");
        return instr->getType() == InstructionType::SIG_WAIT;
    }

 private:
    SignalWaitInstruction(const SignalWaitInstruction&) = default;
    SignalWaitInstruction(SignalWaitInstruction&&)      = default;

    SignalWaitInstruction* cloneImpl() const& override { return new SignalWaitInstruction(*this); }
    SignalWaitInstruction* cloneImpl() && override { return new SignalWaitInstruction(std::move(*this)); }

    std::vector<INSTR_ID_T> send_ids_;
}; // class SignalWaitInstruction

} // namespace nn_ir
} // namespace nn_compiler
