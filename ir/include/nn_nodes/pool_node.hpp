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
 * @file.    pool_node.hpp
 * @brief.   This is PoolNode class
 * @details. This header defines PoolNode class.
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

class PoolNode : public NodeMixin<PoolNode, NNNode> {
 public:
    explicit PoolNode(const NodeInfo& node_info,
                      PoolType        pool_type,
                      Shape2D         kernel_size,
                      Shape2D         stride_size,
                      Shape2D         dilation_size,
                      Pad4            padding_size,
                      PadCalcType     pad_calc_type);

    std::string getNodeTypeAsString() const override { return "Pool"; }

    void setKernelNodeParameters(const KernelNodeParameters& kernel_node_parameters) {
        kernel_node_parameters_ = kernel_node_parameters;
    }

    KernelNodeParameters&       getKernelNodeParameters() { return kernel_node_parameters_; }
    const KernelNodeParameters& getKernelNodeParameters() const { return kernel_node_parameters_; }

    const PoolType&    getPoolType() const { return pool_type_; }
    const PadCalcType& getPadCalcType() const { return pad_calc_type_; }

 private:
    PoolType             pool_type_;
    KernelNodeParameters kernel_node_parameters_;
    PadCalcType          pad_calc_type_;
}; // class PoolNode

} // namespace nn_ir
} // namespace nn_compiler
