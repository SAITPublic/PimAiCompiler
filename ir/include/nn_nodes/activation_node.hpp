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
 * @file.    activation_node.hpp
 * @brief.   This is ActivationNode class
 * @details. This header defines ActivationNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"
#include "ir/nn_node.hpp"

#include "ir/ir_includes.hpp"

namespace nn_compiler {
namespace nn_ir {

class ActivationNode : public NodeMixin<ActivationNode, NNNode> {
 public:
    explicit ActivationNode(const NodeInfo&            node_info,
                            std::unique_ptr<ShiftNode> shift_node,
                            ActivationType             act_type,
                            float                      slope,
                            float                      negative_slope,
                            float                      min,
                            float                      max)
        : NodeMixin(node_info, NodeType::ACTIVATION), shift_node_(std::move(shift_node)), activation_type_(act_type),
          slope_(slope), negative_slope_(negative_slope), min_(min), max_(max) {}

    std::string getNodeTypeAsString() const override { return "Activation"; }

    void setShiftNode(std::unique_ptr<ShiftNode> shift_node) { shift_node_ = std::move(shift_node); }

    float            getMin() const { return min_; }
    float            getMax() const { return max_; }
    float            getSlope() const { return slope_; }
    float            getNegativeSlope() const { return negative_slope_; }
    ActivationType   getActivationType() const { return activation_type_; }
    const ShiftNode* getShiftNode() const { return shift_node_.get(); }
    ShiftNode*       getShiftNode() { return shift_node_.get(); }

    ActivationNode(const ActivationNode& other);
    ActivationNode(ActivationNode&&) = default;

 private:
    std::unique_ptr<ShiftNode> shift_node_;
    ActivationType             activation_type_;
    float                      slope_;
    float                      negative_slope_;
    float                      min_;
    float                      max_;
}; // class ActivationNode

} // namespace nn_ir
} // namespace nn_compiler
