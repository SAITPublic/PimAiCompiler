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

class AtenAsTensorNode : public NodeMixin<AtenAsTensorNode, NNNode> {
 public:
    explicit AtenAsTensorNode(const NodeInfo& node_info, int dtype, int device) :
            NodeMixin(node_info, NodeType::ATENASTENSOR), dtype_(dtype), device_(device) {}

    std::string getNodeTypeAsString(void) const override { return "AtenAsTensor"; }
    
    void setDtype(int dtype) { dtype_ = dtype; }

    int getDtype() { return dtype_; }

    void setDevice(int device) { device_ = device; }

    int getDevice() { return device_; }
    
 private:
    int dtype_  = INT32_MAX;
    int device_ = INT32_MAX;

}; // class AtenAsTensorNode

} // namespace nn_ir
} // namespace nn_compiler
