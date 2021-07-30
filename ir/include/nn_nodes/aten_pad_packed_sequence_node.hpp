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
class AtenPadPackedSequenceNode : public NodeMixin<AtenPadPackedSequenceNode, NNNode>
{
   public:
    explicit AtenPadPackedSequenceNode(const NodeInfo& node_info, int batch_first, float padding_value, int64_t total_length)
        : NodeMixin(node_info, NodeType::ATENPADPACKEDSEQUENCE),
          batch_first_(batch_first),
          padding_value_(padding_value),
          total_length_(total_length)
    {
    }

    std::string getNodeTypeAsString() const override { return "AtenPadPackedSequence"; }

    void setBatchFirst(int batch_first) { batch_first_ = batch_first; }
    void setPaddingValue(float padding_value) { padding_value_ = padding_value; }

    int getBatchFirst() const { return batch_first_; }
    float getPaddingValue() const { return padding_value_; }
    int64_t getTotalLength() const { return total_length_; }

   private:
    int batch_first_;
    float padding_value_;
    int64_t total_length_;

};  // class AtenPadPackedSequenceNode

}  // namespace nn_ir
}  // namespace nn_compiler
