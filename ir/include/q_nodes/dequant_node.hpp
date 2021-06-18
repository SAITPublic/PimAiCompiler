/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file.    dequant_node.hpp
 * @brief.   This is DequantNode class
 * @details. This header defines DequantNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"

#include "ir/include/nn_ir.hpp"
#include "ir/include/q_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class DequantNode : public NodeMixin<DequantNode, QNode> {
 public:
    explicit DequantNode(const NodeInfo&      node_info,
                         QuantType            quant_type,
                         std::vector<float>   scale,
                         std::vector<int32_t> zero_point,
                         std::vector<int8_t>  frac_len)
        : NodeMixin(node_info, NodeType::DEQUANT), quant_type_(quant_type), scale_(std::move(scale)),
          zero_point_(std::move(zero_point)), frac_len_(std::move(frac_len)) {}

    std::string getNodeTypeAsString() const override { return "Dequant"; }

    const QuantType&            getQuantType() const { return quant_type_; }
    const std::vector<float>&   getScale() const { return scale_; }
    const std::vector<int32_t>& getZeroPoint() const { return zero_point_; }
    const std::vector<int8_t>&  getFracLen() const { return frac_len_; }

 private:
    QuantType            quant_type_;
    std::vector<float>   scale_;
    std::vector<int32_t> zero_point_;
    std::vector<int8_t>  frac_len_;
}; // class DequantNode

} // namespace nn_ir
} // namespace nn_compiler
