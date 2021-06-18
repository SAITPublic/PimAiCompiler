/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "ir/include/generated/ir_generated.h"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {

class IRHWNodeParser {
 public:
    IRHWNodeParser() = default;

    IRHWNodeParser(const IRHWNodeParser&) = delete;
    IRHWNodeParser(IRHWNodeParser&&)      = delete;
    IRHWNodeParser& operator=(const IRHWNodeParser&) = delete;
    IRHWNodeParser& operator=(IRHWNodeParser&&) = delete;

    template <IR::HWNode::AnyType>
    std::unique_ptr<nn_ir::HWNode> parseHWNode(const IR::HwNode* ir_node, const nn_ir::NodeInfo& node_info);

    std::unique_ptr<nn_ir::ActivationNode> getActNode(const nn_ir::NodeInfo&            node_info,
                                                      const IR::NNNode::ActivationNode* ir_act_node);
    std::unique_ptr<nn_ir::ShiftNode>      getShiftNode(const nn_ir::NodeInfo&       node_info,
                                                        const IR::OPNode::ShiftNode* ir_shift_node);
}; // class IRNNNodeParser
} // namespace nn_compiler
