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
class AtenSoftmaxNode : public NodeMixin<AtenSoftmaxNode, NNNode>
{
   public:
    explicit AtenSoftmaxNode(const NodeInfo& node_info, int64_t dim, int64_t dtype)
        : NodeMixin(node_info, NodeType::ATENSOFTMAX), dim_(dim), dtype_(dtype)
    {
    }

    std::string getNodeTypeAsString() const override { return "AtenSoftmaxNode"; }

    void setDim(int64_t dim) { dim_ = dim; }
    void setDtype(int64_t dtype) { dtype_ = dtype; }

    int64_t getDim() const { return dim_; }
    int64_t getDtype() const { return dtype_; }

   private:
    int64_t dim_;
    int64_t dtype_;
};  // class AtenSoftmaxNode

}  // namespace nn_ir
}  // namespace nn_compiler
