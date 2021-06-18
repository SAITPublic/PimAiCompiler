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
 * @file    instruction.cpp
 * @brief   This is Instruction class
 * @details This source defines Instruction class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */

#include "ir/include/instruction.hpp"
#include "ir/include/common/log.hpp"

namespace nn_compiler {
namespace nn_ir {

Instruction::~Instruction() = default;

void Instruction::dump(std::ostream& s) const { s << type_ << ", id " << id_; }

} // namespace nn_ir
} // namespace nn_compiler
