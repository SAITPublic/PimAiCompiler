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
 * @file    edge_execution_step.cpp
 * @brief   This is EdgeExecutionStep class
 * @details This source defines EdgeExecutionStep class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */

#include "ir/include/edge_execution_step.hpp"
#include "ir/include/common/log.hpp"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {

Edge* EdgeExecutionStep::getEdge() const { return edge_id_ == INVALID_ID ? nullptr : graph_.getEdge(edge_id_); }

const nn_ir::Node* EdgeExecutionStep::getAttachedNode() const {
    if (auto* edge = getEdge()) {
        switch (exec_type_) {
            case nn_ir::EdgeExecutionStepType::LOAD_START:
            case nn_ir::EdgeExecutionStepType::LOAD_SYNC:
                return edge->getOutNode();
            case nn_ir::EdgeExecutionStepType::STORE_START:
            case nn_ir::EdgeExecutionStepType::STORE_SYNC:
                return edge->getInNode();
            default:
                Log::IR::E() << "invalid edge execution step";
                break;
        }
    }
    return nullptr;
}

NO_DISCARD const nn_ir::ExecutionStep& EdgeExecutionStep::getSyncStep() const {
    auto sync_type = getSyncType(getEdgeStepType());
    return getEdge()->getStep(sync_type);
}

void EdgeExecutionStep::printName(std::ostream& s) const {
    s << exec_type_ << " ";
    if (isLoad()) {
        if (auto* out_node = getEdge()->getOutNode()) {
            s << *out_node;
        } else {
            s << "input";
        }
    } else {
        if (auto* in_node = getEdge()->getInNode()) {
            s << *in_node;
        } else {
            s << "output";
        }
    }
}

} // namespace nn_ir
} // namespace nn_compiler
