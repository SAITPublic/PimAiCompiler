#pragma once

#include "ir/include/generated/ir_generated.h"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {

class IRCONTROLNodeParser {
 public:
    /**
     * @brief.      Constructor of IRCONTROLNodeParser.
     * @details.    This function constructs IRCONTROLNodeParser
     * @param[in].
     * @param[out].
     * @returns.
     */
    IRCONTROLNodeParser() = default;

    IRCONTROLNodeParser(const IRCONTROLNodeParser&) = delete;
    IRCONTROLNodeParser(IRCONTROLNodeParser&&)      = delete;
    IRCONTROLNodeParser& operator=(const IRCONTROLNodeParser&) = delete;
    IRCONTROLNodeParser& operator=(IRCONTROLNodeParser&&) = delete;

    template <IR::CONTROLNode::AnyType>
    std::unique_ptr<nn_ir::CONTROLNode> parseControlNode(const IR::ControlNode* ir_node, const nn_ir::NodeInfo& node_info);
}; // class IRCONTROLNodeParser
} // namespace nn_compiler
