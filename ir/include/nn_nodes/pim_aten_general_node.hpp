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

#define DECLARE_NNIR_OP_NODE(op_name, node_type)                                                         \
    namespace nn_compiler                                                                                \
    {                                                                                                    \
    namespace nn_ir                                                                                      \
    {                                                                                                    \
    class op_name##Node : public NodeMixin<op_name##Node, NNNode>                                        \
    {                                                                                                    \
       public:                                                                                           \
        explicit op_name##Node(const NodeInfo& node_info) : NodeMixin(node_info, NodeType::node_type) {} \
        std::string getNodeTypeAsString(void) const override { return #op_name; }                        \
    };                                                                                                   \
    }                                                                                                    \
    }

DECLARE_NNIR_OP_NODE(AtenAnd, ATENAND)
DECLARE_NNIR_OP_NODE(AtenAny, ATENANY)
DECLARE_NNIR_OP_NODE(AtenAppend, ATENAPPEND)
DECLARE_NNIR_OP_NODE(AtenBitwiseNot, ATENBITWISENOT)
DECLARE_NNIR_OP_NODE(AtenBmm, ATENBMM)
DECLARE_NNIR_OP_NODE(AtenBool, ATENBOOL)
DECLARE_NNIR_OP_NODE(AtenCeil, ATENCEIL)
DECLARE_NNIR_OP_NODE(AtenClear, ATENCLEAR)
DECLARE_NNIR_OP_NODE(AtenCpu, ATENCPU)
DECLARE_NNIR_OP_NODE(AtenCuda, ATENCUDA)
DECLARE_NNIR_OP_NODE(AtenDim, ATENDIM)
DECLARE_NNIR_OP_NODE(AtenDiv, ATENDIV)
DECLARE_NNIR_OP_NODE(AtenEq, ATENEQ)
DECLARE_NNIR_OP_NODE(AtenEqual, ATENEQUAL)
DECLARE_NNIR_OP_NODE(AtenFill, ATENFILL)
DECLARE_NNIR_OP_NODE(AtenFloorDivide, ATENFLOORDIVIDE)
DECLARE_NNIR_OP_NODE(AtenGe, ATENGE)
DECLARE_NNIR_OP_NODE(AtenGt, ATENGT)
DECLARE_NNIR_OP_NODE(AtenIndex, ATENINDEX)
DECLARE_NNIR_OP_NODE(AtenInt, ATENINT)
DECLARE_NNIR_OP_NODE(AtenIs, ATENIS)
DECLARE_NNIR_OP_NODE(AtenItem, ATENITEM)
