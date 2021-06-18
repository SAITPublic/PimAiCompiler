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
 * @file.    global_concat_node.hpp
 * @brief.   This is GlobalConcat class
 * @details. This header defines GlobalConcat class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/blob.hpp"
#include "ir/include/data_edge.hpp"
#include "ir/include/global_node.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class GlobalConcatNode : public NodeMixin<GlobalConcatNode, GlobalNode> {
 public:
    GlobalConcatNode(const NodeInfo&         node_info,
                     NODE_ID_T               uid,
                     nn_ir::SyncType         sync_type,
                     nn_ir::SigType          sig_type,
                     const nn_ir::Shape4D&   ofm_starts,
                     nn_ir::GlobalConcatAxis concat_axis,
                     nn_ir::GlobalConcatType concat_type)
        : NodeMixin(node_info, NodeType::GCONCAT, uid, sync_type, sig_type),
          ofm_starts_({{.n = ofm_starts.n, .c = ofm_starts.c, .h = ofm_starts.h, .w = ofm_starts.w}}),
          concat_axis_(concat_axis), concat_type_(concat_type) {}

    std::string      getNodeTypeAsString() const override { return "GlobalConcat"; }
    Shape4D          getOriginalDim() const override { return getFirstOutEdge<DataEdge>()->getBlob()->getShape(); }
    MemoryInfo       getMemoryInfo() const override { return getFirstOutEdge<DataEdge>()->getFirstMemoryAllocation(); }
    Coord4D          getOfmStarts() const { return ofm_starts_; }
    GlobalConcatAxis getConcatAxis() const { return concat_axis_; }
    GlobalConcatType getConcatType() const { return concat_type_; }
    bool             isBranchConcat() const { return concat_type_ == GlobalConcatType::INTER; }
    bool             isTileConcat() const { return concat_type_ == GlobalConcatType::INTRA; }

 private:
    Coord4D          ofm_starts_;
    GlobalConcatAxis concat_axis_;
    GlobalConcatType concat_type_;
}; // class GlobalConcatNode

} // namespace nn_ir
} // namespace nn_compiler
