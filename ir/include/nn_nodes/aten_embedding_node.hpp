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

namespace nn_compiler {
namespace nn_ir {

class AtenEmbeddingNode : public NodeMixin<AtenEmbeddingNode, NNNode> {
 public:
    explicit AtenEmbeddingNode(const NodeInfo& node_info, int64_t padding_idx,
                                bool scale_grad_by_freq, bool sparse)
                                : NodeMixin(node_info, NodeType::ATENEMBEDDING), padding_idx_(padding_idx), 
                                scale_grad_by_freq_(scale_grad_by_freq), sparse_(sparse) {}

    std::string getNodeTypeAsString(void) const override { return "AtenEmbedding"; }

    void setPaddingIdx(int64_t padding_idx) { padding_idx_ = padding_idx;}
    void setScaleGradByFreq(bool scale_grad_by_freq) { scale_grad_by_freq_ = scale_grad_by_freq; }
    void setSparse(bool sparse) { sparse_ = sparse; }

    int64_t  getPaddingIdx() { return padding_idx_; }
    bool getScaleGradByFreq() { return scale_grad_by_freq_; }
    bool getSparse() { return sparse_; }

 private:
    int64_t padding_idx_;
    bool scale_grad_by_freq_;
    bool sparse_;
}; // class AtenEmbeddingNode

} // namespace nn_ir
} // namespace nn_compiler
