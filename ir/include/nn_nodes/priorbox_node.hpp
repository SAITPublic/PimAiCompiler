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
 * @file.    priorbox_node.hpp
 * @brief.   This is PriorBoxNode class
 * @details. This header defines PriorBoxNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"

#include "ir/include/nn_ir.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class PriorBoxNode : public NodeMixin<PriorBoxNode, NNNode> {
 public:
    explicit PriorBoxNode(const NodeInfo&    node_info,
                          std::vector<float> min_sizes,
                          std::vector<float> max_sizes,
                          std::vector<float> aspect_ratios,
                          bool               flip,
                          bool               clip,
                          std::vector<float> variance,
                          float              step_h,
                          float              step_w,
                          float              offset,
                          PriorboxType       type,
                          BLOB_ID_T          blob_id)
        : NodeMixin(node_info, NodeType::PRIORBOX), min_sizes_(min_sizes), max_sizes_(max_sizes),
          aspect_ratios_(aspect_ratios), flip_(flip), clip_(clip), variance_(variance), step_h_(step_h),
          step_w_(step_w), offset_(offset), type_(type), blob_id_(blob_id) {}

    std::string getNodeTypeAsString() const override { return "PriorBox"; }
    void        setBlobId(BLOB_ID_T id) { blob_id_ = id; }
    BLOB_ID_T   getBlobId() const { return blob_id_; }
    Blob*       getBlob() const { return getGraph().getBlob(blob_id_); }

    const std::vector<float>& getMinSizes() const { return min_sizes_; }
    const std::vector<float>& getMaxSizes() const { return max_sizes_; }
    const std::vector<float>& getAspectRatios() const { return aspect_ratios_; }
    bool                      getFlip() const { return flip_; }
    bool                      getClip() const { return clip_; }
    const std::vector<float>& getVariance() const { return variance_; }
    float                     getStepH() const { return step_h_; }
    float                     getStepW() const { return step_w_; }
    float                     getOffset() const { return offset_; }
    PriorboxType              getPriorboxType() const { return type_; }

 private:
    std::vector<float> min_sizes_;
    std::vector<float> max_sizes_;
    std::vector<float> aspect_ratios_;
    bool               flip_;
    bool               clip_;
    std::vector<float> variance_;
    float              step_h_;
    float              step_w_;
    float              offset_;
    PriorboxType       type_;
    BLOB_ID_T          blob_id_;
}; // class PriorBoxNode

} // namespace nn_ir
} // namespace nn_compiler
