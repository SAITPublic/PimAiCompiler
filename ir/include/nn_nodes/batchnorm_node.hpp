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
 * @file.    batchnorm_node.hpp
 * @brief.   This is BatchNormNode class
 * @details. This header defines BatchNormNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class BatchNormNode : public NodeMixin<BatchNormNode, NNNode> {
 public:
    explicit BatchNormNode(const NodeInfo&    node_info,
                           nn_ir::Axis        axis,
                           bool               use_global_stats,
                           float              eps,
                           float              scale,
                           std::vector<float> std_buf,
                           std::vector<float> mean_buf)
        : NodeMixin(node_info, NodeType::BATCHNORM), axis_(axis), use_global_stats_(use_global_stats), eps_(eps),
          scale_(scale), std_buf_(std_buf), mean_buf_(mean_buf) {}

    std::string getNodeTypeAsString(void) const override { return "BatchNorm"; }

    void setEps(float eps) { eps_ = eps; }
    void setScale(float scale) { scale_ = scale; }
    void setStdBuf(std::vector<float> std_buf) { std_buf_ = std_buf; }
    void setMeanTerm(std::vector<float> mean_buf) { mean_buf_ = mean_buf; }
    void setUseGlobalStats(bool use_global_stats) { use_global_stats_ = use_global_stats; }

    float              getEps() const { return eps_; }
    nn_ir::Axis        getAxis() const { return axis_; }
    float              getScale() const { return scale_; }
    std::vector<float> getStdBuf() const { return std_buf_; }
    std::vector<float> getMeanBuf() const { return mean_buf_; }
    bool               getUseGlobalStats() const { return use_global_stats_; }

 private:
    nn_ir::Axis        axis_;
    bool               use_global_stats_;
    float              eps_;
    float              scale_;
    std::vector<float> std_buf_;
    std::vector<float> mean_buf_;
}; // class BatchNormNode

} // namespace nn_ir
} // namespace nn_compiler
