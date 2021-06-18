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
 * @file    execution_step.cpp
 * @brief   This is ExecutionStep class
 * @details This source defines ExecutionStep class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */
#include "ir/include/execution_step.hpp"
#include "ir/include/common/log.hpp"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {

ExecutionStep::~ExecutionStep() = default;

ExecutionStep::ExecutionStep(const ExecutionStep& other)
    : graph_(other.graph_), id_(other.graph_.getNextStepId()), type_(other.type_) {
    instrs_.reserve(other.instrs_.size());
    for (const auto& instr : other.instrs_) {
        instrs_.push_back(instr->clone());
    }
}

} // namespace nn_ir
} // namespace nn_compiler
