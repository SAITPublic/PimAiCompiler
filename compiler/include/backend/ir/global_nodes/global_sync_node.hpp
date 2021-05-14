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
 * @file.    global_sync_node.hpp
 * @brief.   This is GlobalSync class
 * @details. This header defines GlobalSync class.
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

class GlobalSyncNode : public NodeMixin<GlobalSyncNode, GlobalNode> {
 public:
    GlobalSyncNode(const NodeInfo& node_info, NODE_ID_T uid, nn_ir::SyncType sync_type, nn_ir::SigType sig_type)
        : NodeMixin(node_info, NodeType::GSYNC, uid, sync_type, sig_type) {}

    std::string getNodeTypeAsString() const override { return "GlobalSync"; }
    MemoryInfo  getMemoryInfo() const override { return getFirstOutEdge<DataEdge>()->getFirstMemoryAllocation(); }
    Shape4D getOriginalDim() const override { return cast<nn_ir::DataEdge>(getFirstInEdge()).getBlob()->getShape(); }
}; // class GlobalSyncNode

} // namespace nn_ir
} // namespace nn_compiler
