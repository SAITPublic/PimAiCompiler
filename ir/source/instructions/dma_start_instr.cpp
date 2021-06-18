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
 * @file    dma_start_instr.cpp
 * @brief   This is DMAStartInstruction class
 * @details This source defines DMAStartInstruction class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */

#include "ir/include/instructions/dma_start_instr.hpp"
#include "ir/include/common/log.hpp"

namespace nn_compiler {
namespace nn_ir {

void DMAStartInstruction::dump(std::ostream& s) const {
    s << "[" << data_type_ << "] ";
    Instruction::dump(s);
    s << ", " << direction_ << ", size: " << size_ << " (0x" << std::hex << size_ << std::dec << "), src: " << src_addr_
      << "(0x" << std::hex << src_addr_ << std::dec << "), dst: " << dst_addr_ << "(0x" << std::hex << dst_addr_
      << std::dec << ')';
}

} // namespace nn_ir
} // namespace nn_compiler
