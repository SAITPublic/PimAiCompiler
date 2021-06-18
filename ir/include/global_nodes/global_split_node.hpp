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
 * @file.    global_split_node.hpp
 * @brief.   This is GlobalSplit class
 * @details. This header defines GlobalSplit class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/blob.hpp"
#include "ir/data_edge.hpp"
#include "ir/global_node.hpp"
#include "ir/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class GlobalSplitNode : public NodeMixin<GlobalSplitNode, GlobalNode> {
 public:
    GlobalSplitNode(const NodeInfo&       node_info,
                    NODE_ID_T             uid,
                    nn_ir::SyncType       sync_type,
                    nn_ir::SigType        sig_type,
                    const nn_ir::Shape4D& ifm_starts,
                    nn_ir::PartitionMode  partition_mode)
        : NodeMixin(node_info, NodeType::GSPLIT, uid, sync_type, sig_type),
          ifm_starts_({{.n = ifm_starts.n, .c = ifm_starts.c, .h = ifm_starts.h, .w = ifm_starts.w}}),
          partition_mode_(partition_mode) {}

    std::string   getNodeTypeAsString() const override { return "GlobalSplit"; }
    Shape4D       getOriginalDim() const override { return getFirstInEdge<DataEdge>()->getBlob()->getShape(); }
    MemoryInfo    getMemoryInfo() const override { return getFirstInEdge<DataEdge>()->getFirstMemoryAllocation(); }
    Coord4D       getIfmStarts() const { return ifm_starts_; }
    PartitionMode getPartitionMode() const { return partition_mode_; }

 private:
    Coord4D       ifm_starts_;
    PartitionMode partition_mode_;
}; // class GlobalSplitNode

} // namespace nn_ir
} // namespace nn_compiler
