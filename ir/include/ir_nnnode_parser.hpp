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
 * @file.    ir_importer.hpp
 * @brief.   This is IRParser class
 * @details. This header defines IRParser class.
 * @version. 0.1.
 */

#pragma once

#include "ir/include/generated/ir_generated.h"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {

class IRNNNodeParser {
 public:
    /**
     * @brief.      Constructor of IRNNNodeParser.
     * @details.    This function constructs IRNNNodeParser
     * @param[in].
     * @param[out].
     * @returns.
     */
    IRNNNodeParser() = default;

    IRNNNodeParser(const IRNNNodeParser&) = delete;
    IRNNNodeParser(IRNNNodeParser&&)      = delete;
    IRNNNodeParser& operator=(const IRNNNodeParser&) = delete;
    IRNNNodeParser& operator=(IRNNNodeParser&&) = delete;

    template <IR::NNNode::AnyType>
    std::unique_ptr<nn_ir::NNNode> parseNNNode(const IR::NnNode* ir_node, const nn_ir::NodeInfo& node_info);

    std::unique_ptr<nn_ir::ActivationNode> getActNode(const nn_ir::NodeInfo&            node_info,
                                                      const IR::NNNode::ActivationNode* ir_act_node);
    std::unique_ptr<nn_ir::ShiftNode>      getShiftNode(const nn_ir::NodeInfo&       node_info,
                                                        const IR::OPNode::ShiftNode* ir_shift_node);
}; // class IRNNNodeParser
} // namespace nn_compiler
