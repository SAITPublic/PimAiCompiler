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
class AtenPackPaddedSequenceNode : public NodeMixin<AtenPackPaddedSequenceNode, NNNode>
{
   public:
    explicit AtenPackPaddedSequenceNode(const NodeInfo& node_info, int batch_first)
        : NodeMixin(node_info, NodeType::ATENPACKPADDEDSEQUENCE), batch_first_(batch_first)
    {
    }

    std::string getNodeTypeAsString() const override { return "AtenPackPaddedSequence"; }

    void setBatchFirst(int batch_first) { batch_first_ = batch_first; }

    int getBatchFirst() const { return batch_first_; }

   private:
    int batch_first_;

};  // class AtenPackPaddedSequenceNode

}  // namespace nn_ir
}  // namespace nn_compiler
