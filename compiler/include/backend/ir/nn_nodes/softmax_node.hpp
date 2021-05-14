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
 * @file.    softmax_node.hpp
 * @brief.   This is SoftmaxNode class
 * @details. This header defines SoftmaxNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"
#include "ir/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class SoftmaxNode : public NodeMixin<SoftmaxNode, NNNode> {
 public:
    explicit SoftmaxNode(const NodeInfo& node_info,
                         nn_ir::Axis     axis,
                         int32_t         exp_lut_blob_id,
                         float           exp_scale,
                         float           exp_bias,
                         int32_t         softmax_lut_blob_id,
                         float           softmax_scale_ex,
                         int32_t         softmax_max_sum_ex,
                         int32_t         softmax_max_ex,
                         float           softmax_scale_sum_ex,
                         bool            has_mask)
        : NodeMixin(node_info, NodeType::SOFTMAX), axis_(axis), exp_lut_blob_id_(exp_lut_blob_id),
          exp_scale_(exp_scale), exp_bias_(exp_bias), softmax_lut_blob_id_(softmax_lut_blob_id),
          softmax_scale_ex_(softmax_scale_ex), softmax_max_sum_ex_(softmax_max_sum_ex), softmax_max_ex_(softmax_max_ex),
          softmax_scale_sum_ex_(softmax_scale_sum_ex), has_mask_(has_mask) {}

    std::string getNodeTypeAsString(void) const override { return "Softmax"; }

    nn_ir::Axis getAxis() const { return axis_; }
    int32_t     getExpLUTBlobId() const { return exp_lut_blob_id_; }
    float       getExpScale() const { return exp_scale_; }
    float       getExpBias() const { return exp_bias_; }
    int32_t     getSoftmaxLUTBlobId() const { return softmax_lut_blob_id_; }
    float       getSoftmaxScaleEx() const { return softmax_scale_ex_; }
    int32_t     getSoftmaxMaxSumEx() const { return softmax_max_sum_ex_; }
    int32_t     getSoftmaxMaxEx() const { return softmax_max_ex_; }
    float       getSoftmaxScaleSumEx() const { return softmax_scale_sum_ex_; }
    bool        hasMask() const { return has_mask_; }

 private:
    nn_ir::Axis axis_;
    int32_t     exp_lut_blob_id_;
    float       exp_scale_;
    float       exp_bias_;
    int32_t     softmax_lut_blob_id_;
    float       softmax_scale_ex_;
    int32_t     softmax_max_sum_ex_;
    int32_t     softmax_max_ex_;
    float       softmax_scale_sum_ex_;
    bool        has_mask_;
}; // class SoftmaxNode

} // namespace nn_ir
} // namespace nn_compiler
