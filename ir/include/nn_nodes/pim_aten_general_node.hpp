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

#define DECLARE_ATEN_OP_NODE(op_name, node_type)                                                         \
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

DECLARE_ATEN_OP_NODE(AtenAnd, ATENAND)
DECLARE_ATEN_OP_NODE(AtenAny, ATENANY)
DECLARE_ATEN_OP_NODE(AtenAppend, ATENAPPEND)
DECLARE_ATEN_OP_NODE(AtenBitwiseNot, ATENBITWISENOT)
DECLARE_ATEN_OP_NODE(AtenBmm, ATENBMM)
DECLARE_ATEN_OP_NODE(AtenBool, ATENBOOL)
DECLARE_ATEN_OP_NODE(AtenCeil, ATENCEIL)
DECLARE_ATEN_OP_NODE(AtenClear, ATENCLEAR)
DECLARE_ATEN_OP_NODE(AtenCpu, ATENCPU)
DECLARE_ATEN_OP_NODE(AtenCuda, ATENCUDA)
DECLARE_ATEN_OP_NODE(AtenDim, ATENDIM)
DECLARE_ATEN_OP_NODE(AtenDiv, ATENDIV)
DECLARE_ATEN_OP_NODE(AtenEq, ATENEQ)
DECLARE_ATEN_OP_NODE(AtenEqual, ATENEQUAL)
DECLARE_ATEN_OP_NODE(AtenFill, ATENFILL)
DECLARE_ATEN_OP_NODE(AtenFloorDivide, ATENFLOORDIVIDE)
DECLARE_ATEN_OP_NODE(AtenGe, ATENGE)
DECLARE_ATEN_OP_NODE(AtenGt, ATENGT)
DECLARE_ATEN_OP_NODE(AtenIndex, ATENINDEX)
DECLARE_ATEN_OP_NODE(AtenInt, ATENINT)
DECLARE_ATEN_OP_NODE(AtenIs, ATENIS)
DECLARE_ATEN_OP_NODE(AtenItem, ATENITEM)
DECLARE_ATEN_OP_NODE(AtenLen, ATENLEN)
DECLARE_ATEN_OP_NODE(AtenList, ATENLIST)
DECLARE_ATEN_OP_NODE(AtenLog, ATENLOG)
DECLARE_ATEN_OP_NODE(AtenLt, ATENLT)
DECLARE_ATEN_OP_NODE(AtenMaskedSelect, ATENMASKEDSELECT)
DECLARE_ATEN_OP_NODE(AtenMatmul, ATENMATMUL)
DECLARE_ATEN_OP_NODE(AtenMul, ATENMUL)
DECLARE_ATEN_OP_NODE(AtenNeg, ATENNEG)
DECLARE_ATEN_OP_NODE(AtenNe, ATENNE)
DECLARE_ATEN_OP_NODE(AtenNot, ATENNOT)
DECLARE_ATEN_OP_NODE(AtenPow, ATENPOW)
DECLARE_ATEN_OP_NODE(AtenRelu, ATENRELU)
DECLARE_ATEN_OP_NODE(AtenReshape, ATENRESHAPE)
DECLARE_ATEN_OP_NODE(AtenTanh, ATENTANH)
DECLARE_ATEN_OP_NODE(AtenTensor, ATENTENSOR)
DECLARE_ATEN_OP_NODE(AtenView, ATENVIEW)
DECLARE_ATEN_OP_NODE(AtenZeros, ATENZEROS)
DECLARE_ATEN_OP_NODE(AtenZerosLike, ATENZEROSLIKE)
