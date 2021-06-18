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

#include "ir/include/nn_ir.hpp"
#include "ir/include/node.hpp"

#include "common/include/cast.hpp"

namespace nn_compiler {

/// @class BlockIterator
/// @brief Iterates over blocks located between start and finish node in RPO order.
template <typename StartNode, typename FinishNode, bool include_open_intervals>
class BlockIterator {
 public:
    BlockIterator()
        : it(nn_ir::NNIR::NodeConstIt(nn_ir::NodesList::const_iterator())),
          end(nn_ir::NNIR::NodeConstIt(nn_ir::NodesList::const_iterator())) {}

    BlockIterator(const BlockIterator& other) : it(other.it), end(other.end) {}

    explicit BlockIterator(const nn_ir::NNIR& graph) : it(graph.getNodes().begin()), end(graph.getNodes().end()) {
        // Locate start of first block
        if (include_open_intervals) {
            auto jt = std::find_if(it, end, [](auto& node) { return is_any_of<StartNode, FinishNode>(node); });
            if (jt == end || isa<FinishNode>(*jt)) {
                return;
            }
        }
        ++*this;
    }

    BlockIterator& operator=(const BlockIterator& other) {
        it  = other.it;
        end = other.end;
        return *this;
    }

    bool operator==(const BlockIterator& other) const {
        // All completed iterators are equal
        if (it == end && other.it == other.end) {
            return true;
        }
        return it == other.it && end == other.end;
    }

    bool operator!=(const BlockIterator& other) const { return !(*this == other); }

    BlockIterator& operator++() {
        // Locate start of next block
        it = std::find_if(it, end, [](auto& node) { return isa<StartNode>(node); });
        // Eltwises have 2 vsplits...
        it = std::find_if_not(it, end, [](auto& node) { return isa<StartNode>(node); });
        return *this;
    }

    iterator_range<nn_ir::NNIR::NodeConstIt> operator*() {
        auto start = it;
        it         = std::find_if(it, end, [](auto& node) { return isa<FinishNode>(node); });
        return {start, it};
    }

 private:
    nn_ir::NNIR::NodeConstIt it, end;
};

using TileBlockIterator = BlockIterator<nn_ir::VSplitNode, nn_ir::VConcatNode, false>;

using UntiledNodeIterator = BlockIterator<nn_ir::VConcatNode, nn_ir::VSplitNode, true>;
} // namespace nn_compiler
