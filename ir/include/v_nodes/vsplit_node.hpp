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
 * @brief.   This is VSplitNode class
 * @details. This header defines VSplitNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/v_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class VSplitNode : public NodeMixin<VSplitNode, VNode> {
 public:
    VSplitNode(const NodeInfo& node_info, TileNumbers num_tiles, const std::vector<TileInfo>& tile_infos)
        : NodeMixin(node_info, NodeType::VSPLIT), num_tiles_(num_tiles), tile_infos_(tile_infos) {}

    std::string getNodeTypeAsString() const override { return "VSplit"; }

    nn_ir::TileNumbers           getNumTiles() const { return num_tiles_; }
    const std::vector<TileInfo>& getTileInfos() const { return tile_infos_; }

    const TileInfo& getTileInfoFor(const nn_ir::Node& successor_node) const {
        Log::IR::E_IF(tile_infos_.empty()) << "There's no TileInfos initialized in " << *this;
        const auto id = successor_node.getId();
        const auto tile_info =
            estd::find_if(tile_infos_, [id](const auto& tile_info) { return tile_info.node_id == id; });
        Log::IR::E_IF(tile_info == tile_infos_.end()) << successor_node << " is not successor of " << *this;
        return *tile_info;
    }

 private:
    TileNumbers           num_tiles_;
    std::vector<TileInfo> tile_infos_;
}; // class VSplitNode

} // namespace nn_ir
} // namespace nn_compiler
