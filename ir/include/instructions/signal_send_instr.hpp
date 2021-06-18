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
 * @file.    signal_send_instr.hpp
 * @brief.   This is SignalSendInstruction class
 * @details. This header defines SignalSendInstruction class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"
#include "ir/misc_instr.hpp"
#include "ir/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {

class SignalSendInstruction : public MiscInstruction {
 public:
    explicit SignalSendInstruction(INSTR_ID_T id, uint32_t dma_ch)
        : MiscInstruction(id, nn_ir::InstructionType::SIG_SEND), dma_ch_id_(dma_ch) {}

    std::unique_ptr<SignalSendInstruction> clone() const& {
        return std::unique_ptr<SignalSendInstruction>(this->cloneImpl());
    }
    std::unique_ptr<SignalSendInstruction> clone() && {
        return std::unique_ptr<SignalSendInstruction>(std::move(*this).cloneImpl());
    }

    uint32_t getDMAChannelId() const { return dma_ch_id_; }

    template <typename T>
    static bool classof(const Instruction* instr) {
        static_assert(std::is_same<std::decay_t<T>, SignalSendInstruction>::value, "incorrect type");
        return instr->getType() == InstructionType::SIG_SEND;
    }

 private:
    SignalSendInstruction(const SignalSendInstruction&) = default;
    SignalSendInstruction(SignalSendInstruction&&)      = default;

    SignalSendInstruction* cloneImpl() const& override { return new SignalSendInstruction(*this); }
    SignalSendInstruction* cloneImpl() && override { return new SignalSendInstruction(std::move(*this)); }

    uint32_t dma_ch_id_;
}; // class SignalSendInstruction

} // namespace nn_ir
} // namespace nn_compiler
