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
 * @file.    data_format_node.hpp
 * @brief.   This is DataFormatNode class
 * @details. This header defines DataFormatNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"

#include "ir/include/nn_ir.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class DataFormatNode : public NodeMixin<DataFormatNode, NNNode> {
 public:
    explicit DataFormatNode(const NodeInfo& node_info, DataFormatConversion format_direction, Shape4D shape);

    std::string getNodeTypeAsString() const override { return "DataFormat"; }

    const DataFormatConversion& getFormatDirection() const { return format_direction_; }
    const Shape4D&              getShape() const { return shape_; }

 private:
    DataFormatConversion format_direction_;
    Shape4D              shape_;
}; // class DataFormatNode

} // namespace nn_ir
} // namespace nn_compiler
