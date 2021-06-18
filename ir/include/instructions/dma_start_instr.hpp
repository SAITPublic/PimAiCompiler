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
 * @file.    dma_start_instr.hpp
 * @brief.   This is DMAStartInstruction class
 * @details. This header defines DMAStartInstruction class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/memory_instr.hpp"
#include "ir/include/nn_ir.hpp"

// DMA channel numbers. In fact there's nothing type-specific about channels themselves,
// but we currently nail them down to a particular data type in order to allow some parallelization
// in an easy way. We could do more advanced scheduling in future.
#define DMAC_OFM     0
#define DMAC_IFM     1
#define DMAC_WEIGHT  2
#define DMAC_CFIFO   4
#define DMAC_SDMA_OP 8

namespace nn_compiler {
namespace nn_ir {

class DMAStartInstruction : public MemoryInstruction {
 public:
    /**
     * @brief.      Constructor of DMAStartInstruction.
     * @details.    This function constructs DMAStartInstruction
     * @param[in].
     * @param[out].
     * @returns.
     */
    explicit DMAStartInstruction(INSTR_ID_T               id,
                                 InstructionType          type,
                                 DMADirection             direction,
                                 MemoryDataType           data_type,
                                 uint32_t                 dma_ch_id,
                                 uint32_t                 sram_id,
                                 uint32_t                 size,
                                 uint32_t                 src_addr,
                                 uint32_t                 dst_addr,
                                 const nn_ir::DataLayout& src_layout,
                                 const nn_ir::DataLayout& dst_layout)
        : MemoryInstruction(id, type), direction_(direction), data_type_(data_type), dma_ch_id_(dma_ch_id),
          sram_id_(sram_id), size_(size), src_addr_(src_addr), dst_addr_(dst_addr), src_layout_(src_layout),
          dst_layout_(dst_layout) {}

    std::unique_ptr<DMAStartInstruction> clone() const& {
        return std::unique_ptr<DMAStartInstruction>(this->cloneImpl());
    }
    std::unique_ptr<DMAStartInstruction> clone() && {
        return std::unique_ptr<DMAStartInstruction>(std::move(*this).cloneImpl());
    }

    DMADirection             getDirection() const { return direction_; }
    MemoryDataType           getDataType() const { return data_type_; }
    uint32_t                 getDMAChannelId() const { return dma_ch_id_; }
    uint32_t                 getSramId() const { return sram_id_; }
    uint32_t                 getSize() const { return size_; }
    uint32_t                 getSrcAddr() const { return src_addr_; }
    uint32_t                 getDstAddr() const { return dst_addr_; }
    const nn_ir::DataLayout& getSrcLayout() const { return src_layout_; }
    const nn_ir::DataLayout& getDstLayout() const { return dst_layout_; }
    void                     setSize(uint32_t new_size) { size_ = new_size; }
    void                     setDMAChannelId(uint32_t dma_ch_id) { dma_ch_id_ = dma_ch_id; }

    void dump(std::ostream& s) const override;

    template <typename T>
    static bool classof(const Instruction* instr) {
        static_assert(std::is_same<std::decay_t<T>, DMAStartInstruction>::value, "incorrect type");
        return instr->getType() == InstructionType::DMA_START;
    }

 private:
    DMAStartInstruction(const DMAStartInstruction&) = default;
    DMAStartInstruction(DMAStartInstruction&&)      = default;

    DMAStartInstruction* cloneImpl() const& override { return new DMAStartInstruction(*this); }
    DMAStartInstruction* cloneImpl() && override { return new DMAStartInstruction(std::move(*this)); }

    DMADirection      direction_;
    MemoryDataType    data_type_;
    uint32_t          dma_ch_id_;
    uint32_t          sram_id_;
    uint32_t          size_;
    uint32_t          src_addr_;
    uint32_t          dst_addr_;
    nn_ir::DataLayout src_layout_;
    nn_ir::DataLayout dst_layout_;
}; // class DMAStartInstruction

} // namespace nn_ir
} // namespace nn_compiler
