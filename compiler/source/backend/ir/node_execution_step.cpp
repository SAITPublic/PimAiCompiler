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
 * @file    node_execution_step.cpp
 * @brief   This is NodeExecutionStep class
 * @details This source defines NodeExecutionStep class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */
#include "ir/common/log.hpp"

#include "ir/nn_ir.hpp"
#include "ir/node_execution_step.hpp"

namespace nn_compiler {
namespace nn_ir {

Node* NodeExecutionStep::getNode() const { return node_id_ == INVALID_ID ? nullptr : graph_.getNode(node_id_); }

NO_DISCARD const nn_ir::ExecutionStep& NodeExecutionStep::getSyncStep() const {
    auto sync_type = getSyncType(getNodeStepType());
    return getNode()->getStep(sync_type);
}

void NodeExecutionStep::printName(std::ostream& s) const { s << exec_type_ << ' ' << *getNode(); }

} // namespace nn_ir
} // namespace nn_compiler
