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
class AtenTopkNode : public NodeMixin<AtenTopkNode, NNNode>
{
   public:
    explicit AtenTopkNode(const NodeInfo& node_info, int64_t k, int64_t dim, int largest, int sorted)
        : NodeMixin(node_info, NodeType::ATENTOPK), k_(k), dim_(dim), largest_(largest), sorted_(sorted)
    {
    }

    std::string getNodeTypeAsString() const override { return "AtenTopk"; }

    void setK(int64_t k) { k_ = k; }
    void setLargest(int largest) { largest_ = largest; }
    void setDim(int64_t dim) { dim_ = dim; }
    void setSorted(int sorted) { sorted_ = sorted; }

    int64_t getK() { return k_; }
    int64_t getDim() { return dim_; }
    int getLargest() { return largest_; }
    int getSorted() { return sorted_; }

   private:
    int64_t k_;
    int64_t dim_;
    int largest_;
    int sorted_;

};  // class AtenTopkNode

}  // namespace nn_ir
}  // namespace nn_compiler
