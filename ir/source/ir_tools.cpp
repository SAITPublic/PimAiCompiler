/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/include/ir_tools.hpp"
#include "ir/include/common/log.hpp"
#include "ir/include/ir_includes.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {

std::unique_ptr<nn_ir::ShiftNode> parseShiftNode(const nn_ir::NodeInfo&       shift_node_info,
                                                 const IR::OPNode::ShiftNode* ir_shift_node) {
    std::vector<int32_t>  quantization_shift(getShift<int32_t>(ir_shift_node->quantization_shift()));
    std::vector<int32_t>  multiplication_shift(getShift<int32_t>(ir_shift_node->multiplication_shift()));
    std::vector<int32_t>  activation_shift(getShift<int32_t>(ir_shift_node->activation_shift()));
    std::vector<uint16_t> lut_scale(getShift<uint16_t>(ir_shift_node->lut_scale()));
    std::vector<uint16_t> lut_bias(getShift<uint16_t>(ir_shift_node->lut_bias()));
    std::vector<uint32_t> grelu_info(getShift<uint32_t>(ir_shift_node->grelu_info()));

    return std::make_unique<nn_ir::ShiftNode>(
        shift_node_info, quantization_shift, multiplication_shift, activation_shift, lut_scale, lut_bias, grelu_info);
}

nn_ir::DataLayout parseDataLayout(const IR::TargetHardware::Type::DataLayout* ir_layout) {
    nn_ir::DataLayout layout;

    layout.total_dim  = parseParam<nn_ir::Shape4D>(ir_layout->total_dim());
    layout.padding    = parseParam<nn_ir::Pad4>(ir_layout->padding());
    layout.gap        = parseParam<nn_ir::Shape4D>(ir_layout->gap());
    layout.cell_unit  = parseParam<nn_ir::Shape4D>(ir_layout->cell_info()->cell_unit());
    layout.byte_order = parseEnum<nn_ir::PixelByteOrder>(ir_layout->cell_info()->cell_byte_order());
    layout.bpp        = ir_layout->cell_info()->cell_bpp();

    return layout;
}

nn_ir::MemoryInfo parseMemoryInfo(const IR::TargetHardware::Type::MemoryInfo* ir_mem_info) {
    nn_ir::IR_Node_Config_Type_ ir_memory_type   = ir_mem_info->type();
    nn_ir::IR_Node_Config_Type_ ir_memory_region = ir_mem_info->data_type();
    nn_ir::MemoryInfo           mem;

    mem.memory_type = parseEnum<nn_ir::MemoryType>(ir_memory_type);
    mem.data_type   = parseEnum<nn_ir::MemoryDataType>(ir_memory_region);
    mem.mem_id      = ir_mem_info->mem_id();
    mem.addr        = static_cast<MEMORY_OFFSET_T>(ir_mem_info->addr());
    mem.size        = static_cast<MEMORY_SIZE_T>(ir_mem_info->size());
    mem.layout      = parseDataLayout(ir_mem_info->layout());

    return mem;
}

} // namespace nn_compiler
