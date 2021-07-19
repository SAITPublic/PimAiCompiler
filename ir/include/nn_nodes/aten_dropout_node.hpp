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

#include "ir/include/ir_includes.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenDropoutNode : public NodeMixin<AtenDropoutNode, NNNode> {
 public:
    explicit AtenDropoutNode(const NodeInfo& node_info, float proportion, int train)
    : NodeMixin(node_info, NodeType::ATENDROPOUT), proportion_(proportion), train_(train) {}

    std::string getNodeTypeAsString() const override { return "AtenDropout"; }

    void setProportion(float proportion) { proportion_ = proportion; }
    void setTrain(int train) { train_ = train; }

    float getProportion() { return proportion_; }
    int getTrain() { return train_; }

 private:
    float proportion_;
    int train_;
}; // class AtenDropoutNode

} // namespace nn_ir
} // namespace nn_compiler
