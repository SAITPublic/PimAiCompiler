/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/include/node.hpp"
#include "common/include/arithmetics.hpp"
#include "ir/include/blob.hpp"
#include "ir/include/common/log.hpp"
#include "ir/include/data_edge.hpp"
#include "ir/include/nn_node_type_traits.hpp"

#include "ir/include/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {

#define PROCESS_NODE(NODE_TYPE, NODE_CLASS, BASE_NODE_CLASS) \
    static_assert(std::is_base_of_v<BASE_NODE_CLASS, NODE_CLASS>, "invalid base node class");
#include "ir/include/nodes.def"

Node::Node(const NodeInfo& node_info, NodeType type) : Node(node_info.graph, type) {
    id_           = node_info.id;
    name_         = node_info.name;
    in_edge_ids_  = node_info.in_edge_ids;
    out_edge_ids_ = node_info.out_edge_ids;

    for (unsigned type = 0; type < unsigned(nn_ir::NodeExecutionStepType::COUNT); ++type) {
        steps_[type] = std::make_unique<NodeExecutionStep>(node_info.graph,
                                                           node_info.graph.getNextStepId(),
                                                           ExecutionStepType::NODE,
                                                           id_,
                                                           nn_ir::NodeExecutionStepType(type));
    }
}

Node::Node(const Node& other)
    : id_(other.graph_.getNextNodeId()), name_(other.name_), graph_(other.graph_), type_(other.type_),
      operation_mode_(other.operation_mode_), in_edge_ids_(other.in_edge_ids_), out_edge_ids_(other.out_edge_ids_),
      kernel_mem_info_(other.kernel_mem_info_), psum_mem_info_(other.psum_mem_info_),
      ifm_mem_info_(other.ifm_mem_info_), ofm_mem_info_(other.ofm_mem_info_), instr_mem_info_(other.instr_mem_info_),
      mapped_hw_block_(other.mapped_hw_block_), node_data_info_(other.node_data_info_) {

    for (unsigned type = 0; type < unsigned(nn_ir::NodeExecutionStepType::COUNT); ++type) {
        steps_[type] = std::make_unique<NodeExecutionStep>(
            graph_, graph_.getNextStepId(), ExecutionStepType::NODE, id_, nn_ir::NodeExecutionStepType(type));
    }
}

Node::~Node() = default;

NODE_ID_T Node::assignId(const NNIR& graph) {
    id_ = graph.getNextNodeId();
    return id_;
}

void Node::setId(NODE_ID_T id) {
    id_ = id;
    for (unsigned type = 0; type < unsigned(nn_ir::NodeExecutionStepType::COUNT); ++type) {
        steps_[type]->setNodeId(id);
    }
}

NodeInfo Node::getNodeInfo() const {
    NodeInfo node_info(id_, name_, graph_, in_edge_ids_, out_edge_ids_);
    return node_info;
}

std::vector<BLOB_ID_T> Node::getIFMBlobIds() const {
    std::vector<BLOB_ID_T> ret;
    for (auto in_edge_id : in_edge_ids_) {
        if (auto in_edge = cast_if<nn_ir::DataEdge>(graph_.getEdge(in_edge_id))) {
            ret.push_back(in_edge->getBlobId());
        }
    }
    return ret;
}
std::vector<BLOB_ID_T> Node::getOFMBlobIds(void) const {
    std::vector<BLOB_ID_T> ret;
    for (auto out_edge_id : out_edge_ids_) {
        if (auto out_edge = cast_if<nn_ir::DataEdge>(graph_.getEdge(out_edge_id))) {
            ret.push_back(out_edge->getBlobId());
        }
    }
    return ret;
}

DataType Node::getOutEdgeDataTypes(void) {
    auto out_edge = cast_if<nn_ir::DataEdge>(getFirstOutEdge());

    Log::IR::E_IF(!out_edge) << "this method supports only DATA Edge";
    return out_edge->getBlob()->getDataType();
}

void Node::addInEdgeIdsAfter(EDGE_ID_T target_id, const std::vector<EDGE_ID_T>& ids_to_add) {
    for (auto it = in_edge_ids_.begin(); it != in_edge_ids_.end(); ++it) {
        auto edge = *it;
        if (edge == target_id) {
            in_edge_ids_.insert(it, ids_to_add.begin(), ids_to_add.end());
            return;
        }
    }
    Log::IR::E() << __FUNCTION__ << ": " << *this << " cannot find input edge #" << target_id;
}

void Node::deleteOutEdgeId(EDGE_ID_T id) {
    for (auto it = out_edge_ids_.begin(); it != out_edge_ids_.end(); ++it) {
        auto edge = *it;
        if (edge == id) {
            out_edge_ids_.erase(it);
            return;
        }
    }
    Log::IR::E() << __FUNCTION__ << ": " << *this << " cannot find output edge #" << id;
}

void Node::deleteInEdgeId(EDGE_ID_T id) {
    for (auto it = in_edge_ids_.begin(); it != in_edge_ids_.end(); ++it) {
        auto edge = *it;
        if (edge == id) {
            in_edge_ids_.erase(it);
            return;
        }
    }
    Log::IR::E() << __FUNCTION__ << ": " << *this << " cannot find input edge #" << id;
}

void Node::setIfmMemInfoByInputNo(const std::vector<MemoryInfo>& ifm_mem_info, uint32_t input_no) {
    // Make sure that ifm_mem_info_ is non-empty only if it contains at least one non-empty MemoryInfo
    // This simplifies checks for MemoryInfo presence a lot
    if (input_no >= ifm_mem_info_.size()) {
        // Don't grow ifm_mem_info_ for empty slots
        if (ifm_mem_info.empty()) {
            return;
        }
        ifm_mem_info_.resize(input_no + 1);
    } else if (input_no == ifm_mem_info_.size() - 1 && ifm_mem_info.empty()) {
        // If we're clearing the last slot, we can shrink our vector.
        // input_no becomes new size.
        // There could me more empty slots before the last one. Strip all of them.
        while (input_no > 0 && ifm_mem_info_[input_no - 1].empty()) {
            input_no--;
        }

        ifm_mem_info_.resize(input_no);
        return;
    }

    ifm_mem_info_[input_no] = ifm_mem_info;
}

MEMORY_OFFSET_T Node::addInstrSize(MemoryType mem_type, MemoryDataType data_type, MEMORY_OFFSET_T offset) {
    MEMORY_OFFSET_T added_offset = 0;
    for (auto& mem_info : instr_mem_info_) {
        if (mem_info.first == data_type) {
            if (mem_type == MemoryType::DRAM) {
                mem_info.second.first.addr = offset;
                added_offset += mem_info.second.first.size;
            } else {
                mem_info.second.second.addr = offset;
                if (mem_info.second.second.mem_id == 0) {
                    added_offset += mem_info.second.second.size;
                }
            }
        }
    }
    return offset + added_offset;
}

MEMORY_OFFSET_T Node::addKernelMemSize(MEMORY_OFFSET_T offset) {
    MEMORY_OFFSET_T added_offset = 0;
    for (auto& mem_info : kernel_mem_info_) {
        mem_info.addr = offset;
        if (mem_info.mem_id == 0) {
            added_offset += mem_info.size;
        }
    }
    return offset + added_offset;
}

MEMORY_OFFSET_T Node::addPsumMemSize(MEMORY_OFFSET_T offset) {
    MEMORY_SIZE_T psum_size = getAlignedPsumMemSize();
    for (auto& mem_info : psum_mem_info_) {
        mem_info.addr = offset;
        Log::IR::E_IF(psum_size != mem_info.size) << "Node::setPsumOffset() : Psum size are different";
    }
    return offset + psum_size;
}

MEMORY_OFFSET_T Node::addIfmMemSize(EDGE_ID_T edge_id, MEMORY_OFFSET_T offset) {
    uint32_t input_no = findIndexOfElement(in_edge_ids_, edge_id);

    for (auto& mem_info : ifm_mem_info_[input_no]) {
        mem_info.addr = offset;
    }
    return offset + getAlignedIfmMemSize(edge_id);
}

MEMORY_OFFSET_T Node::addOfmMemSize(EDGE_ID_T edge_id, MEMORY_OFFSET_T offset) {
    // All outgoing edges are copies, so we use shared meminfo.
    // If this ever changes, change this to handle multiple outputs in a similar way
    // as we do for IFMs.
    for (auto& mem_info : ofm_mem_info_[0]) {
        mem_info.addr = offset;
    }
    return offset + getAlignedOfmMemSize(edge_id);
}

static MEMORY_SIZE_T calcSramSize(const std::vector<nn_ir::MemoryInfo>& mem_infos) {
    MEMORY_SIZE_T size = 0;

    for (const nn_ir::MemoryInfo& mem_info : mem_infos) {
        // Account only IDP #0. Multiple IDPs (if present) occupy
        // the same address space, so second IDP wouldn't bring in
        // any additional usage
        if (mem_info.mem_id == 0) {
            size += mem_info.size;
        }
    }

    return size;
}

MEMORY_SIZE_T Node::getAlignedIfmMemSize(EDGE_ID_T edge_id) const {
    uint32_t input_no = findIndexOfElement(in_edge_ids_, edge_id);

    Log::IR::E_IF(input_no >= ifm_mem_info_.size()) << *this << ": No IFM info for edge #" << edge_id;
    return calcSramSize(ifm_mem_info_[input_no]);
}

MEMORY_SIZE_T Node::getAlignedOfmMemSize(EDGE_ID_T edge_id) const {
    // All outgoing edges are copies, so we use shared meminfo.
    // If this ever changes, change this to handle multiple outputs in a similar way
    // as we do for IFMs.
    Log::IR::E_IF(ofm_mem_info_.empty()) << *this << ": No OFM info for edge #" << edge_id;
    return calcSramSize(ofm_mem_info_[0]);
}

MEMORY_SIZE_T Node::getTotalIfmMemSize() const {
    MEMORY_SIZE_T size = 0;

    for (auto& ifm_no : ifm_mem_info_) {
        size += calcSramSize(ifm_no);
    }

    return size;
}

MEMORY_SIZE_T Node::getTotalOfmMemSize() const {
    // All outgoing edges are copies, so we count only one of them
    // If this ever changes, this function would become virtual and be overridden
    // for certain Node subclacces
    Log::IR::E_IF(ofm_mem_info_.size() > 1) << "Nodes with multiple OFMs are not supported! " << *this;

    return ofm_mem_info_.empty() ? 0 : calcSramSize(ofm_mem_info_.front());
}

MEMORY_SIZE_T Node::getAlignedPsumMemSize() const { return psum_mem_info_.empty() ? 0 : psum_mem_info_[0].size; }
MEMORY_SIZE_T Node::getAlignedKernelMemSize(nn_ir::MemoryType mem_type) const {
    if (kernel_mem_info_.empty() || kernel_mem_info_[0].memory_type != mem_type) {
        return 0;
    }
    return kernel_mem_info_[0].size;
}

MEMORY_SIZE_T Node::getAlignedInstrMemSize(MemoryDataType data_type) const {
    for (auto& mem_info : instr_mem_info_) {
        if (mem_info.first == data_type) {
            return mem_info.second.second.size;
        }
    }

    return INVALID_SIZE;
}

// FIXME: We shouldn't be using traits here, but historically some users expect this to
// return false for non-executable nodes. Despite you can indeed say those have everything in
// DRAM for the most part. Need to find it out and remove the check.
bool Node::isDramToDram() const { return isExecutableNode(*this) && hasIfmInDram() && hasOfmInDram(); }

int32_t Node::getExecutionOrderNumByRPO() const {
    if (!graph_.isNodesNumberingValid()) {
        graph_.numberNodes();
    }
    return exec_number_;
}

Edge& Node::getFirstInEdge() const { return *graph_.getEdge(in_edge_ids_.at(0)); }

Edge& Node::getFirstOutEdge() const { return *graph_.getEdge(out_edge_ids_.at(0)); }

Edge* Node::getUniqueInEdge() const { return in_edge_ids_.size() == 1 ? graph_.getEdge(in_edge_ids_[0]) : nullptr; }

Edge* Node::getUniqueOutEdge() const { return out_edge_ids_.size() == 1 ? graph_.getEdge(out_edge_ids_[0]) : nullptr; }

Edge& Node::getInEdge(size_t i) const { return *graph_.getEdge(in_edge_ids_[i]); }

Edge& Node::getOutEdge(size_t i) const { return *graph_.getEdge(out_edge_ids_[i]); }

std::string Node::getOriginalNodeName() const {
    auto             name = getName();
    std::string_view result(name);

    std::string_view concat_suffix("_concat");
    if (name.size() >= concat_suffix.size() &&
        concat_suffix == result.substr(result.size() - concat_suffix.size(), concat_suffix.size())) {
        //  Matched pattern *_concat
        //
        result.remove_suffix(concat_suffix.size());
        return std::string(result);
    }

    auto pos = result.rfind('_');

    if (pos == std::string_view::npos || !std::all_of(result.begin() + pos + 1, result.end(), ::isdigit)) {
        // Does not match *_[0-9]+ pattern
        //
        return name;
    }

    // remove digits
    //
    result.remove_suffix(result.size() - pos);

    std::string_view split_suffix("_split");
    if (result.size() >= split_suffix.size() &&
        split_suffix == result.substr(result.size() - split_suffix.size(), split_suffix.size())) {
        // Matched pattern *_split_[0-9]+
        //
        result.remove_suffix(split_suffix.size());
        return std::string(result);
    }

    std::string_view splitted_prefix("splitted_");
    if (result.size() >= splitted_prefix.size() && splitted_prefix == result.substr(0, splitted_prefix.size())) {
        // Matched pattern splitted_*_[0-9]+
        //
        result.remove_prefix(splitted_prefix.size());
        return std::string(result);
    }

    // original node name matched pattern *_split_[0-9]+
    //
    return name;
}

Edge& Node::EdgeIterator::dereference() const {
    Log::IR::E_IF(*base() == INVALID_ID) << "EdgeIterator internal error";
    return *graph_.getEdge(*base());
}

const Node* Node::getFirstSuccessorNode() const { return getFirstOutEdge().getOutNode(); }

const Node* Node::getFirstPredecessorNode() const { return getFirstInEdge().getInNode(); }

Node* Node::getFirstSuccessorNode() { return getFirstOutEdge().getOutNode(); }

Node* Node::getFirstPredecessorNode() { return getFirstInEdge().getInNode(); }

const nn_ir::VConcatNode* getSuccessorVConcatOrNull(const nn_ir::Node& node) {
    auto successor = node.getFirstSuccessorNode();
    return successor ? cast_if<nn_ir::VConcatNode>(successor) : nullptr;
}
const nn_ir::VSplitNode* getPredecessorVSplitOrNull(const nn_ir::Node& node) {
    auto predeccessor = node.getFirstPredecessorNode();
    return predeccessor ? cast_if<nn_ir::VSplitNode>(predeccessor) : nullptr;
}

} // namespace nn_ir
} // namespace nn_compiler
