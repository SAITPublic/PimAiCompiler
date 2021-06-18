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
 * @file.    shift_node.hpp
 * @brief.   This is ShiftNode class
 * @details. This header defines ShiftNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"

#include "ir/nn_ir.hpp"
#include "ir/op_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class ShiftNode : public NodeMixin<ShiftNode, OPNode> {
 public:
    explicit ShiftNode(const NodeInfo&       node_info,
                       std::vector<int32_t>  quantization_shift,
                       std::vector<int32_t>  multiplication_shift,
                       std::vector<int32_t>  activation_shift,
                       std::vector<uint16_t> lut_scale,
                       std::vector<uint16_t> lut_bias,
                       std::vector<uint32_t> grelu_info);

    std::string getNodeTypeAsString() const override { return "Shift"; }

    std::vector<int32_t>  getQuantizationShift() const { return quantization_shift_; }
    std::vector<int32_t>  getMultiplicationShift() const { return multiplication_shift_; }
    std::vector<int32_t>  getActivationShift() const { return activation_shift_; }
    std::vector<uint16_t> getLutScale() const { return lut_scale_; }
    std::vector<uint16_t> getLutBias() const { return lut_bias_; }
    std::vector<uint32_t> getGreluInfo() const { return grelu_info_; }

    void setLutBias(const std::vector<uint16_t>& lut_bias) { lut_bias_ = lut_bias; }
    void setLutScale(const std::vector<uint16_t>& lut_scale) { lut_scale_ = lut_scale; }
    void setGreluInfo(const std::vector<uint32_t>& grelu_info) { grelu_info_ = grelu_info; }
    void setQuantizationShift(const std::vector<int32_t>& quantization_shift) {
        quantization_shift_ = quantization_shift;
    }
    void setMultiplicationShift(const std::vector<int32_t>& multiplication_shift) {
        multiplication_shift_ = multiplication_shift;
    }
    void setActivationShift(const std::vector<int32_t>& activation_shift) { activation_shift_ = activation_shift; }

 private:
    std::vector<int32_t>  quantization_shift_;
    std::vector<int32_t>  multiplication_shift_;
    std::vector<int32_t>  activation_shift_;
    std::vector<uint16_t> lut_scale_;
    std::vector<uint16_t> lut_bias_;
    std::vector<uint32_t> grelu_info_;
}; // class ShiftNode

} // namespace nn_ir
} // namespace nn_compiler
