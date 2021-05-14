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
 * @file.    vconcat_node.hpp
 * @brief.   This is VConcatNode class
 * @details. This header defines VConcatNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"
#include "ir/nn_ir.hpp"
#include "ir/v_node.hpp"
#include "ir/v_nodes/vsplit_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class VConcatNode : public NodeMixin<VConcatNode, VNode> {
 public:
    VConcatNode(const NodeInfo&       node_info,      // TODO(a.puschin): change the IR so that
                NODE_ID_T             vsplit_node_id, // VConcat Node has TileInfos and TileNumbers as VSplitNode
                std::vector<TileInfo> tile_infos = std::vector<TileInfo>())
        : NodeMixin(node_info, NodeType::VCONCAT), vsplit_node_id_(vsplit_node_id), tile_infos_(tile_infos) {}

    std::string getNodeTypeAsString() const override { return "VConcat"; }

    const NODE_ID_T& getVsplitNodeId() const { return vsplit_node_id_; }

    const nn_ir::VSplitNode& getVsplitNode() const {
        return cast<nn_ir::VSplitNode>(getGraph().getNode(vsplit_node_id_));
    }

    nn_ir::TileNumbers getTileNumbers() const {
        const auto num_predecessors       = getPredecessorsNum();
        const auto tile_numbers_at_vsplit = getVsplitNode().getNumTiles();
        const auto num_weight_tiles       = num_predecessors / tile_numbers_at_vsplit.getNumberOfTiles();
        return {{.n = tile_numbers_at_vsplit.n,
                 .c = num_weight_tiles,
                 .h = tile_numbers_at_vsplit.h,
                 .w = tile_numbers_at_vsplit.w}};
    }

    const std::vector<TileInfo>& getTileInfos() const { return tile_infos_; }

    const TileInfo& getTileInfoFor(const nn_ir::Node& predecessor_node) const {
        Log::IR::E_IF(tile_infos_.empty()) << "There's no TileInfos initialized in " << *this;
        const auto id = predecessor_node.getId();
        const auto tile_info =
            estd::find_if(tile_infos_, [id](const auto& tile_info) { return tile_info.node_id == id; });
        Log::IR::E_IF(tile_info == tile_infos_.end()) << predecessor_node << " is not predecessor of " << *this;
        return *tile_info;
    }

 private:
    NODE_ID_T vsplit_node_id_;
    // TODO(a.puschin): change the IR to keep TileInfos + add TileNumbers field
    std::vector<TileInfo> tile_infos_;
}; // class VConcatNode

} // namespace nn_ir
} // namespace nn_compiler
