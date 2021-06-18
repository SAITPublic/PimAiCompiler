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
#include "ir/include/ir_hwnode_parser.hpp"
#include "ir/include/ir_nnnode_parser.hpp"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {

class IRParser {
 public:
    /**
     * @brief.      Constructor of IRParser.
     * @details.    This function constructs IRParser
     * @param[in].
     * @param[out].
     * @returns.
     */
    IRParser();

    IRParser(const IRParser&) = delete;
    IRParser(IRParser&&)      = delete;
    IRParser& operator=(const IRParser&) = delete;
    IRParser& operator=(IRParser&&) = delete;

    using nnParseFunction = std::unique_ptr<nn_ir::NNNode> (IRNNNodeParser::*)(const IR::NnNode*      ir_node,
                                                                               const nn_ir::NodeInfo& node_info);

    using opParseFunction = std::unique_ptr<nn_ir::OPNode> (IRParser::*)(const IR::OpNode*      ir_node,
                                                                         const nn_ir::NodeInfo& node_info);

    using globalParseFunction = std::unique_ptr<nn_ir::GlobalNode> (IRParser::*)(const IR::globalNode*  ir_node,
                                                                                 const nn_ir::NodeInfo& node_info);

    using vParseFunction = std::unique_ptr<nn_ir::VNode> (IRParser::*)(const IR::vNode*       ir_node,
                                                                       const nn_ir::NodeInfo& node_info);

    using qParseFunction = std::unique_ptr<nn_ir::QNode> (IRParser::*)(const IR::qNode*       ir_node,
                                                                       const nn_ir::NodeInfo& node_info);

    using hwParseFunction = std::unique_ptr<nn_ir::HWNode> (IRHWNodeParser::*)(const IR::HwNode*      ir_node,
                                                                               const nn_ir::NodeInfo& node_info);

    std::unique_ptr<nn_ir::Node> parseNode(const IR::Node* node, const nn_ir::NNIR& graph);

 private:
    template <IR::OPNode::AnyType>
    std::unique_ptr<nn_ir::OPNode> parseOPNode(const IR::OpNode* ir_node, const nn_ir::NodeInfo& node_info);

    template <IR::GlobalNode::AnyType>
    std::unique_ptr<nn_ir::GlobalNode> parseGlobalNode(const IR::globalNode* ir_node, const nn_ir::NodeInfo& node_info);

    template <IR::VNode::AnyType>
    std::unique_ptr<nn_ir::VNode> parseVNode(const IR::vNode* ir_node, const nn_ir::NodeInfo& node_info);

    template <IR::QNode::AnyType>
    std::unique_ptr<nn_ir::QNode> parseQNode(const IR::qNode* ir_node, const nn_ir::NodeInfo& node_info);

    std::map<IR::NNNode::AnyType, nnParseFunction>         nn_node_parse_func_map_;
    std::map<IR::HWNode::AnyType, hwParseFunction>         hw_node_parse_func_map_;
    std::map<IR::OPNode::AnyType, opParseFunction>         op_node_parse_func_map_;
    std::map<IR::QNode::AnyType, qParseFunction>           q_node_parse_func_map_;
    std::map<IR::VNode::AnyType, vParseFunction>           v_node_parse_func_map_;
    std::map<IR::GlobalNode::AnyType, globalParseFunction> global_node_parse_func_map_;
}; // class IRParser

} // namespace nn_compiler
