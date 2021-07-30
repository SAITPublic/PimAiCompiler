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
class AtenSetItemNode : public NodeMixin<AtenSetItemNode, NNNode>
{
   public:
    explicit AtenSetItemNode(const NodeInfo& node_info, int64_t indices)
        : NodeMixin(node_info, NodeType::ATENSETITEM), indices_(indices)
    {
    }

    std::string getNodeTypeAsString() const override { return "AtenSetItem"; }

    void setIndices(int64_t indices) { indices_ = indices; }

    int64_t getIndices() const { return indices_; }

   private:
    int64_t indices_;
};  // class AtenSetItemNode

}  // namespace nn_ir
}  // namespace nn_compiler
