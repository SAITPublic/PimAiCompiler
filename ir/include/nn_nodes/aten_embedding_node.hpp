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
    explicit AtenEmbeddingNode(const NodeInfo& node_info, std::vector<std::vector<float16>> weights,
                                Shape2D weights_shape, int64_t padding_idx,
                                int scale_grad_by_freq, int sparse)
                                : NodeMixin(node_info, NodeType::ATENEMBEDDING), 
                                weights_(weights), weights_shape_(weights_shape),
                                padding_idx_(padding_idx), 
                                scale_grad_by_freq_(scale_grad_by_freq), sparse_(sparse) {}

    std::string getNodeTypeAsString(void) const override { return "AtenEmbedding"; }

    void setWeights(std::vector<std::vector<float16>> weights) { weights_ = weights; }
    void setWeightsShape(Shape2D weights_shape) { weights_shape_ = weights_shape; }
    void setPaddingIdx(int64_t padding_idx) { padding_idx_ = padding_idx;}
    void setScaleGradByFreq(int scale_grad_by_freq) { scale_grad_by_freq_ = scale_grad_by_freq; }
    void setSparse(int sparse) { sparse_ = sparse; }

    std::vector<std::vector<float16>> getWeights() { return weights_; }
    Shape2D getWeightsShape() { return weights_shape_; }
    int64_t  getPaddingIdx() { return padding_idx_; }
    int getScaleGradByFreq() { return scale_grad_by_freq_; }
    int getSparse() { return sparse_; }

 private:
    std::vector<std::vector<float16>> weights_;
    Shape2D weights_shape_;
    int64_t padding_idx_;
    int scale_grad_by_freq_;
    int sparse_;
}; // class AtenEmbeddingNode

} // namespace nn_ir
} // namespace nn_compiler
