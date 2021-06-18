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
 * @file.    dma_sync_instr.hpp
 * @brief.   This is DMASyncInstruction class
 * @details. This header defines DMASyncInstruction class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/memory_instr.hpp"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {

class DMASyncInstruction : public MemoryInstruction {
 public:
    explicit DMASyncInstruction(INSTR_ID_T id, InstructionType type, INSTR_ID_T start_id)
        : MemoryInstruction(id, type), start_id_(start_id) {}

    INSTR_ID_T getStartId() const { return start_id_; }

    std::unique_ptr<DMASyncInstruction> clone() const& {
        return std::unique_ptr<DMASyncInstruction>(this->cloneImpl());
    }
    std::unique_ptr<DMASyncInstruction> clone() && {
        return std::unique_ptr<DMASyncInstruction>(std::move(*this).cloneImpl());
    }

    void dump(std::ostream& s) const override;

    template <typename T>
    static bool classof(const Instruction* instr) {
        static_assert(std::is_same<std::decay_t<T>, DMASyncInstruction>::value, "incorrect type");
        return instr->getType() == InstructionType::DMA_SYNC;
    }

    void setStartId(INSTR_ID_T new_id) { start_id_ = new_id; }

 private:
    DMASyncInstruction(const DMASyncInstruction&) = default;
    DMASyncInstruction(DMASyncInstruction&&)      = default;

    DMASyncInstruction* cloneImpl() const& override { return new DMASyncInstruction(*this); }
    DMASyncInstruction* cloneImpl() && override { return new DMASyncInstruction(std::move(*this)); }

    INSTR_ID_T start_id_;
}; // class DMASyncInstruction

} // namespace nn_ir
} // namespace nn_compiler
