/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/iterators.hpp"
#include "common/include/types.hpp"

#include "ir/include/common/log.hpp"
#include "ir/include/instruction.hpp"
#include "ir/include/ir_types.hpp"
#include "ir/include/node_execution_step.hpp"
#include "ir/include/node_mixin.hpp"

#include "common/include/algorithm_ext.hpp"
#include "common/include/intrusive_list.hpp"
#include <unordered_map>

namespace nn_compiler {
namespace nn_ir {

class Node : public estd::IntrusiveListNode<Node>, public AbstractNodeMixin<Node> {
 public:
    // make this class abstract
    virtual ~Node() = 0;

    Node& operator=(const Node&) = delete;
    Node& operator=(Node&&) = delete;

    // to simplify stl algorithms (e.g std::find_if can be replaced with std::find in this case)
    friend bool operator==(const Node& l, const Node& r) { return l.getId() == r.getId(); }

    void setId(NODE_ID_T id);
    void setName(std::string name) { name_ = name; }
    void setOperationMode(const std::string& operation_mode) { operation_mode_ = operation_mode; }
    void setInEdgeIds(const std::vector<EDGE_ID_T>& ids) { in_edge_ids_ = ids; }
    void setOutEdgeIds(const std::vector<EDGE_ID_T>& ids) { out_edge_ids_ = ids; }
    void setType(NodeType type) { type_ = type; }
    void clearOutEdgeIds() { out_edge_ids_.clear(); }
    void clearInEdgeIds() { in_edge_ids_.clear(); }
    void addInEdgeIds(const std::vector<EDGE_ID_T>& ids) {
        in_edge_ids_.insert(in_edge_ids_.end(), ids.begin(), ids.end());
    }
    void addInEdgeIdsAfter(EDGE_ID_T target, const std::vector<EDGE_ID_T>& ids_to_add);
    void addOutEdgeIds(const std::vector<EDGE_ID_T>& ids) {
        out_edge_ids_.insert(out_edge_ids_.end(), ids.begin(), ids.end());
    }
    void deleteOutEdgeId(EDGE_ID_T id);
    void deleteInEdgeId(EDGE_ID_T id);

    void setKernelMemInfo(const std::vector<MemoryInfo>& kernel_mem_info) { kernel_mem_info_ = kernel_mem_info; }
    void setConstantMemInfo(const std::vector<MemoryInfo>& constant_mem_info) {
        constant_mem_info_ = constant_mem_info;
    }
    void setPsumMemInfo(const std::vector<MemoryInfo>& psum_mem_info) { psum_mem_info_ = psum_mem_info; }
    void setIfmMemInfo(EDGE_ID_T id, const std::vector<MemoryInfo>& ifm_mem_info) {
        uint32_t input_no = findIndexOfElement(in_edge_ids_, id);
#ifndef NDEBUG
        if (input_no < ifm_mem_info_.size()) {
            checkExisitingMemInfo(ifm_mem_info, ifm_mem_info_[input_no]);
        }
#endif
        setIfmMemInfoByInputNo(ifm_mem_info, input_no);
    }

    void setUniqueOfmMemInfo(const std::vector<MemoryInfo>& ofm_mem_info) {
        if (ofm_mem_info.empty()) {
            ofm_mem_info_.clear();
        } else {
            // Produce only one OFM by default.
            ofm_mem_info_.assign(1, ofm_mem_info);
        }
    }

    void setOfmMemInfo(const EDGE_ID_T&, const std::vector<MemoryInfo>& ofm_mem_info) {
#ifndef NDEBUG
        if (!ofm_mem_info_.empty()) {
            checkExisitingMemInfo(ofm_mem_info, ofm_mem_info_[0]);
        }
#endif
        setUniqueOfmMemInfo(ofm_mem_info);
    }

    void
    setInstrMemInfo(const std::vector<std::pair<MemoryDataType, std::pair<MemoryInfo, MemoryInfo>>>& instr_mem_info) {
        instr_mem_info_ = instr_mem_info;
    }

    void setStep(nn_ir::NodeExecutionStepType type, std::unique_ptr<NodeExecutionStep>& pstep) {
        steps_[unsigned(type)] = std::move(pstep);
    }

    void setInstructions(nn_ir::NodeExecutionStepType type, std::vector<std::unique_ptr<Instruction>> instr) {
        steps_[unsigned(type)]->setInstructions(std::move(instr));
    }

    NODE_ID_T                     getId() const { return id_; }
    const std::string&            getName() const { return name_; }
    std::string                   getOriginalNodeName() const;
    NodeType                      getNodeType() const { return type_; }
    const nn_ir::NNIR&            getGraph() const { return graph_; }
    std::string                   getOperationMode() const { return operation_mode_; }
    const std::vector<EDGE_ID_T>& getInEdgeIds() const { return in_edge_ids_; }
    const std::vector<EDGE_ID_T>& getOutEdgeIds() const { return out_edge_ids_; }

    // Consume all IFMs by default.
    virtual uint32_t getNumInputs() const { return in_edge_ids_.size(); }
    // Produce one OFM by default.
    virtual uint32_t getNumOutputs() const { return 1; }

    Edge& getFirstInEdge() const;
    Edge& getFirstOutEdge() const;

    /// @brief if node has a single input edge, return it, otherwise return nullptr
    Edge* getUniqueInEdge() const;

    /// @brief if node has a single output edge, return it, otherwise return nullptr
    Edge* getUniqueOutEdge() const;

    Edge& getOutEdge(size_t i) const;
    Edge& getInEdge(size_t i) const;

    class EdgeIterator : public IteratorAdaptor<EdgeIterator, std::vector<EDGE_ID_T>::const_iterator, Edge> {
        const NNIR& graph_;

     public:
        EdgeIterator(const NNIR& graph, std::vector<EDGE_ID_T>::const_iterator start)
            : IteratorAdaptor(start), graph_(graph) {}

        Edge& dereference() const;
    };

    EdgeIterator getEdgesBegin(bool in) const {
        auto& edges = in ? getInEdgeIds() : getOutEdgeIds();
        return EdgeIterator(graph_, edges.begin());
    }

    EdgeIterator getEdgesEnd(bool in) const {
        auto& edges = in ? getInEdgeIds() : getOutEdgeIds();
        return EdgeIterator(graph_, edges.end());
    }

    iterator_range<EdgeIterator> getInEdges() const {
        return iterator_range<EdgeIterator>(getEdgesBegin(true), getEdgesEnd(true));
    }

    iterator_range<EdgeIterator> getOutEdges() const {
        return iterator_range<EdgeIterator>(getEdgesBegin(false), getEdgesEnd(false));
    }

    template <typename ValueT>
    auto getInEdges() const {
        return makeFilteredRange<ValueT>(getInEdges());
    }

    template <typename ValueT>
    auto getOutEdges() const {
        return makeFilteredRange<ValueT>(getOutEdges());
    }

    template <typename ValueT>
    const ValueT* getFirstInEdge() const {
        auto range = getInEdges<ValueT>();
        return (range.begin() == range.end()) ? nullptr : &*range.begin();
    }

    template <typename ValueT>
    const ValueT* getFirstOutEdge() const {
        auto range = getOutEdges<ValueT>();
        return (range.begin() == range.end()) ? nullptr : &*range.begin();
    }

    const std::vector<MemoryInfo>& getKernelMemInfo() const { return kernel_mem_info_; }
    const std::vector<MemoryInfo>& getConstantMemInfo() const { return constant_mem_info_; }
    const std::vector<MemoryInfo>& getPsumMemInfo() const { return psum_mem_info_; }
    const std::vector<MemoryInfo>& getIfmMemInfoByInputNo(uint32_t input_no) const {
        return input_no < ifm_mem_info_.size() ? ifm_mem_info_[input_no] : empty_mem_info_;
    }
    const std::vector<MemoryInfo>& getOfmMemInfoByOutputNo(uint32_t output_no) const {
        return output_no < ofm_mem_info_.size() ? ofm_mem_info_[output_no] : empty_mem_info_;
    }

    const std::vector<MemoryInfo>& getIfmMemInfo(EDGE_ID_T id) const {
        uint32_t input_no = findIndexOfElement(in_edge_ids_, id);
        if (input_no < ifm_mem_info_.size()) {
            return ifm_mem_info_[input_no];
        }
        return empty_mem_info_;
    }

    const std::vector<MemoryInfo>& getOfmMemInfo(EDGE_ID_T) const {
        if (ofm_mem_info_.empty()) {
            return empty_mem_info_;
        }
        return ofm_mem_info_[0];
    }

    // We're using node's MemoryInfo only for transient allocations, like SRAM / FIFO buffers.
    // DRAM is allocated on Edges instead; and this isn't going to change because it's too
    // deep-rooted in our compiler and proven to be convenient over time. But should this change,
    // these method will need to be patched accordingly.
    bool hasIfmInDram() const { return ifm_mem_info_.empty(); }
    bool hasOfmInDram() const { return ofm_mem_info_.empty(); }
    bool isDramToDram() const;

    const std::vector<std::pair<MemoryDataType, std::pair<MemoryInfo, MemoryInfo>>>& getInstrMemInfos() const {
        return instr_mem_info_;
    }

    MEMORY_SIZE_T getTotalIfmMemSize() const;
    MEMORY_SIZE_T getTotalOfmMemSize() const;
    MEMORY_SIZE_T getTotalCuMemSize() const { return mapped_hw_block_ == "CU" ? node_data_info_[0].second : 0; }

    MEMORY_SIZE_T getTotalSramMemSize() const {
        return getTotalIfmMemSize() + getTotalOfmMemSize() + getAlignedKernelMemSize(nn_ir::MemoryType::SRAM) +
               getAlignedPsumMemSize() + getTotalCuMemSize();
    }

    std::vector<BLOB_ID_T> getIFMBlobIds() const;
    std::vector<BLOB_ID_T> getOFMBlobIds() const;

    DataType  getOutEdgeDataTypes();
    NodeInfo  getNodeInfo() const;
    NODE_ID_T assignId(const NNIR& graph);

    const NodeExecutionStep& getStep(nn_ir::NodeExecutionStepType type) const { return *steps_[unsigned(type)]; }

    std::vector<std::unique_ptr<Instruction>>& getInstructions(nn_ir::NodeExecutionStepType type) {
        return const_cast<std::vector<std::unique_ptr<Instruction>>&>(steps_[unsigned(type)]->getInstructions());
    }

    const std::vector<std::unique_ptr<Instruction>>& getInstructions(nn_ir::NodeExecutionStepType type) const {
        return steps_[unsigned(type)]->getInstructions();
    }

    virtual Shape4D getPreprocessedKernelBlobDim() const { return {{.n = 0, .c = 0, .h = 0, .w = 0}}; }
    virtual Shape4D getPreprocessedKernelBlobDim() { return {{.n = 0, .c = 0, .h = 0, .w = 0}}; }

    virtual std::vector<Shape4D> getPreprocessedWeightBlobDim() const { return {{{.n = 0, .c = 0, .h = 0, .w = 0}}}; }
    virtual std::vector<Shape4D> getPreprocessedWeightBlobDim() { return {{{.n = 0, .c = 0, .h = 0, .w = 0}}}; }

    MEMORY_OFFSET_T addInstrSize(MemoryType mem_type, MemoryDataType data_type, MEMORY_OFFSET_T offset);
    MEMORY_OFFSET_T addKernelMemSize(MEMORY_OFFSET_T offset);
    MEMORY_OFFSET_T addPsumMemSize(MEMORY_OFFSET_T offset);
    MEMORY_OFFSET_T addIfmMemSize(EDGE_ID_T edge_id, MEMORY_OFFSET_T offset);
    MEMORY_OFFSET_T addOfmMemSize(EDGE_ID_T edge_id, MEMORY_OFFSET_T offset);

    ///@brief get per IDP SRAM size for psum/kernel/instr
    MEMORY_SIZE_T getAlignedIfmMemSize(EDGE_ID_T edge_id) const;
    MEMORY_SIZE_T getAlignedOfmMemSize(EDGE_ID_T edge_id) const;
    MEMORY_SIZE_T getAlignedPsumMemSize() const;
    MEMORY_SIZE_T getAlignedKernelMemSize(nn_ir::MemoryType mem_type) const;
    MEMORY_SIZE_T getAlignedInstrMemSize(MemoryDataType data_type) const;

    virtual std::string getNodeTypeAsString() const { return "None"; }

    void        setMappedHWName(std::string hw_block) { mapped_hw_block_ = hw_block; }
    std::string getMappedHWName() const { return mapped_hw_block_; }

    void setNodeDataInfo(std::vector<std::pair<MemoryDataType, uint32_t>> node_data) { node_data_info_ = node_data; }
    std::vector<std::pair<MemoryDataType, uint32_t>> getNodeDataInfo() const { return node_data_info_; }

    int32_t getExecutionOrderNumByRPO() const;

    uint32_t getPredecessorsNum() const { return in_edge_ids_.size(); }
    uint32_t getSuccessorsNum() const { return out_edge_ids_.size(); }

    const Node* getFirstSuccessorNode() const;
    const Node* getFirstPredecessorNode() const;
    Node*       getFirstPredecessorNode();
    Node*       getFirstSuccessorNode();

 protected:
    explicit Node(const NodeInfo& node_info, NodeType type);
    explicit Node(const NNIR& graph, NodeType type) : graph_(graph), type_(type) {}
    Node(const Node&);
    Node(Node&&) = default;

 private:
    void setIfmMemInfoByInputNo(const std::vector<MemoryInfo>& ifm_mem_info, uint32_t input_no);

    uint32_t findIndexOfElement(const std::vector<EDGE_ID_T>& ids_vec, EDGE_ID_T id) const {
        const auto it = estd::find(ids_vec, id);
        Log::IR::E_IF(it == ids_vec.end()) << "Can't find edge id " << id << " in provided vector!\n";
        return std::distance(std::begin(ids_vec), it);
    }

    void checkExisitingMemInfo(const std::vector<MemoryInfo>& new_mem_info,
                               const std::vector<MemoryInfo>& existing_mem_info) const {
        if (!existing_mem_info.empty()) {
            Log::IR::E_IF(existing_mem_info.size() != new_mem_info.size())
                << __FUNCTION__ << ": Meminfo vector size mismatch! Existing: " << existing_mem_info.size()
                << ", provided: " << new_mem_info.size() << "\n";
            for (uint32_t i = 0; i < existing_mem_info.size(); ++i) {
                Log::IR::E_IF(existing_mem_info[i].size != INVALID_SIZE &&
                              existing_mem_info[i].size != new_mem_info[i].size)
                    << __FUNCTION__ << ": Meminfo size mismatch! Existing: " << existing_mem_info[i].size
                    << ", provided: " << new_mem_info[i].size << ", index: " << i << "\n";
            }
        }
    }

    NODE_ID_T              id_;
    std::string            name_;
    const NNIR&            graph_;
    NodeType               type_;
    std::string            operation_mode_;
    std::vector<EDGE_ID_T> in_edge_ids_;
    std::vector<EDGE_ID_T> out_edge_ids_;

    friend NNIR;
    mutable uint32_t exec_number_ = 0;

    std::unique_ptr<NodeExecutionStep> steps_[unsigned(nn_ir::NodeExecutionStepType::COUNT)];

    // FIXME: Need to find a way to integrate into ir schema (since it is hardware dependent info)
    std::vector<MemoryInfo> kernel_mem_info_;
    std::vector<MemoryInfo> constant_mem_info_;
    std::vector<MemoryInfo> psum_mem_info_;

    std::vector<std::vector<MemoryInfo>> ifm_mem_info_;
    std::vector<std::vector<MemoryInfo>> ofm_mem_info_;

    std::vector<MemoryInfo>                                                   empty_mem_info_;
    std::vector<std::pair<MemoryDataType, std::pair<MemoryInfo, MemoryInfo>>> instr_mem_info_;

    std::string                                      mapped_hw_block_;
    std::vector<std::pair<MemoryDataType, uint32_t>> node_data_info_;
}; // class Node

inline std::ostream& operator<<(std::ostream& s, const Node& node) {
    s << "Node #" << node.getId() << " \"" << node.getName() << "\" Type: " << node.getNodeTypeAsString();
    return s;
}

const nn_ir::VConcatNode* getSuccessorVConcatOrNull(const nn_ir::Node& node);
const nn_ir::VSplitNode*  getPredecessorVSplitOrNull(const nn_ir::Node& node);

} // namespace nn_ir
} // namespace nn_compiler
