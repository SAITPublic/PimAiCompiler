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

#include "ir/include/control_node.hpp"
#include "ir/include/ir_types.hpp"

#define DECLARE_PRIM_OP_NODE(op_name, node_type)                                                         \
    namespace nn_compiler                                                                                \
    {                                                                                                    \
    namespace nn_ir                                                                                      \
    {                                                                                                    \
    class op_name##Node : public NodeMixin<op_name##Node, CONTROLNode>                                   \
    {                                                                                                    \
       public:                                                                                           \
        explicit op_name##Node(const NodeInfo& node_info) : NodeMixin(node_info, NodeType::node_type) {} \
        std::string getNodeTypeAsString(void) const override { return #op_name; }                        \
    };                                                                                                   \
    }                                                                                                    \
    }

DECLARE_PRIM_OP_NODE(PrimBlock, PRIMBLOCK)
DECLARE_PRIM_OP_NODE(PrimData, PRIMDATA)
DECLARE_PRIM_OP_NODE(PrimDevice, PRIMDEVICE)
DECLARE_PRIM_OP_NODE(PrimDtype, PRIMDTYPE)
DECLARE_PRIM_OP_NODE(PrimGetAttr, PRIMGETATTR)
DECLARE_PRIM_OP_NODE(PrimInput, PRIMINPUT)
DECLARE_PRIM_OP_NODE(PrimListConstruct, PRIMLISTCONSTRUCT)
DECLARE_PRIM_OP_NODE(PrimListUnpack, PRIMLISTUNPACK)
DECLARE_PRIM_OP_NODE(PrimOutput, PRIMOUTPUT)
DECLARE_PRIM_OP_NODE(PrimRaiseException, PRIMRAISEEXCEPTION)
DECLARE_PRIM_OP_NODE(PrimSetAttr, PRIMSETATTR)
DECLARE_PRIM_OP_NODE(PrimTupleConstruct, PRIMTUPLECONSTRUCT)
DECLARE_PRIM_OP_NODE(PrimTupleUnpack, PRIMTUPLEUNPACK)
DECLARE_PRIM_OP_NODE(PrimType, PRIMTYPE)
DECLARE_PRIM_OP_NODE(PrimUncheckedCast, PRIMUNCHECKEDCAST)
DECLARE_PRIM_OP_NODE(PrimUninitialized, PRIMUNINITIALIZED)
