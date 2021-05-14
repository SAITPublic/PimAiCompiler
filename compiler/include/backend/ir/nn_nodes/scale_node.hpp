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
 * @file.    scale_node.hpp
 * @brief.   This is ScaleNode class
 * @details. This header defines ScaleNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"
#include "ir/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class ScaleNode : public NodeMixin<ScaleNode, NNNode> {
 public:
    ScaleNode(const NodeInfo& node_info, bool bias_term, std::vector<float> alpha_buf, std::vector<float> beta_buf)
        : NodeMixin(node_info, NodeType::SCALE), bias_term_(bias_term), alpha_buf_(alpha_buf), beta_buf_(beta_buf) {}

    std::string getNodeTypeAsString() const override { return "Scale"; }

    void setBiasTerm(bool bias_term) { bias_term_ = bias_term; }
    void setAlphaBuf(std::vector<float> alpha_buf) { alpha_buf_ = alpha_buf; }
    void setBiasTerm(std::vector<float> beta_buf) { beta_buf_ = beta_buf; }

    bool               getBiasTerm() const { return bias_term_; }
    std::vector<float> getAlphaBuf() const { return alpha_buf_; }
    std::vector<float> getBetaBuf() const { return beta_buf_; }

 private:
    bool               bias_term_;
    std::vector<float> alpha_buf_;
    std::vector<float> beta_buf_;
}; // class ScaleNode

} // namespace nn_ir
} // namespace nn_compiler
