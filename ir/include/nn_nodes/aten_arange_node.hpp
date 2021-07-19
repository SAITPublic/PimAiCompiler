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

class AtenArangeNode : public NodeMixin<AtenArangeNode, NNNode> {
 public:
    explicit AtenArangeNode(const NodeInfo& node_info, int start, int step, int dtype, int layout, int pin_memory) :
            NodeMixin(node_info, NodeType::ATENARANGE), start_(start),
            step_(step), dtype_(dtype), layout_(layout), pin_memory_(pin_memory) {}

    std::string getNodeTypeAsString(void) const override { return "AtenArange"; }

    void setStart(int start) { start_ = start; }

    int getStart() const { return start_; }

    void setStep(int step) { step_ = step; }

    int getStep() const { return step_; }

    void setDtype(int dtype) { dtype_ = dtype; }

    int getDtype() const { return dtype_; }

    void setLayout(int layout) { layout_ = layout; }

    int getLayout() const { return layout_; }

    void setPinMemory(int pin_memory) { pin_memory_ = pin_memory; }

    int getPinMemory() const { return pin_memory_; }
    
 private:
    int  start_     = INT32_MAX;
    int  step_      = INT32_MAX;
    int  dtype_     = INT32_MAX;
    int  layout_    = INT32_MAX;
    int pin_memory_ = INT32_MAX;

}; // class AtenArangeNode

} // namespace nn_ir
} // namespace nn_compiler
