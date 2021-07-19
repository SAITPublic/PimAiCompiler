/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenWarnNode : public NodeMixin<AtenWarnNode, NNNode>
{
   public:
    explicit AtenWarnNode(const NodeInfo& node_info, int value)
        : NodeMixin(node_info, NodeType::ATENWARN), value_(value)
    {
    }

    std::string getNodeTypeAsString() const override { return "AtenWarn"; }

    void setValue(int value) { value_ = value; }

    int getValue() const { return value_; }

   private:
    int value_;
};  // class AtenWarnNode

}  // namespace nn_ir
}  // namespace nn_compiler
