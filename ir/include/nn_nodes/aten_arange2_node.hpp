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

class AtenArange2Node : public NodeMixin<AtenArange2Node, NNNode> {
 public:
    explicit AtenArange2Node(const NodeInfo& node_info, int64_t start, int64_t end,
	                             int64_t dtype, int layout, int device, int pin_memory) :
            NodeMixin(node_info, NodeType::ATENARANGE1), start_(start), end_(end),
            dtype_(dtype), layout_(layout), device_(device), pin_memory_(pin_memory) {}

    std::string getNodeTypeAsString(void) const override { return "AtenArange2"; }

    void setStart(int64_t start) { start_ = start; }

    int64_t getStart() const { return start_; }

    void setEnd(int64_t end) { end_ = end; }

    int64_t getEnd() const { return end_; }

    void setDtype(int64_t dtype) { dtype_ = dtype; }

    int64_t getDtype() const { return dtype_; }

    void setLayout(int layout) { layout_ = layout; }

    int getLayout() const { return layout_; }

    void setDevice(int device) { device_ = device; }

    int getDevice() const { return device_; }

    void setPinMemory(int pin_memory) { pin_memory_ = pin_memory; }

    int getPinMemory() const { return pin_memory_; }
    
 private:
    int64_t start_  = INT64_MIN;
    int64_t end_    = INT64_MIN;
    int64_t dtype_  = INT32_MIN;
    int layout_     = INT32_MIN;
    int device_     = INT32_MIN;
    int pin_memory_ = INT32_MIN;

}; // class AtenArange2Node

} // namespace nn_ir
} // namespace nn_compiler
