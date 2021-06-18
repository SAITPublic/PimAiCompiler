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
 * @file.    convolution_node.hpp
 * @brief.   This is ConvolutionNode class
 * @details. This header defines ConvolutionNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class InputNode : public NodeMixin<InputNode, NNNode> {
 public:
    explicit InputNode(
        const NodeInfo& node_info, InputType input_type, std::vector<float> mean, float scale, bool mirror)
        : NodeMixin(node_info, NodeType::INPUT), input_type_(input_type), mean_(mean), scale_(scale), mirror_(mirror) {}

    std::string getNodeTypeAsString() const override { return "Input"; }

    const bool&               getMirror() const { return mirror_; }
    const float&              getScale() const { return scale_; }
    const InputType&          getInputType() const { return input_type_; }
    const std::vector<float>& getMean() const { return mean_; }

 private:
    InputType          input_type_;
    std::vector<float> mean_;
    float              scale_;
    bool               mirror_;
}; // class InputNode

} // namespace nn_ir
} // namespace nn_compiler
