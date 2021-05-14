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
 * @file.    nn_ir.hpp
 * @brief.   This is nn_ir::NNIR class
 * @details. This header defines nn_ir::NNIR class.
 * @version. 0.1.
 */

#pragma once

#include "ir/blob.hpp"
#include "ir/edge.hpp"
#include "ir/execution_step.hpp"
#include "ir/ir_intrusive_list_traits.hpp"
#include "ir/node.hpp"

#include "common/algorithm_ext.hpp"
#include "common/types.hpp"

#include <type_traits>
#include <unordered_map>

namespace nn_compiler {
namespace nn_ir {

using NodesList = estd::intrusive_list<Node, NNIRIntrusiveListTraits<NNIR>>;

class NNIR {
 public:
    /**
     * @brief.      Constructor of NNIR.
     * @details.    This function constructs NNIR
     * @param[in].
     * @param[out].
     * @returns.
     */
    explicit NNIR(GRAPH_ID_T id, std::string name) : id_(id), name_(name), ir_factory_(*this) {}

    NNIR(const NNIR&) = delete;
    NNIR& operator=(const NNIR&) = delete;

    template <typename EdgeT, typename... TArgs>
    EdgeT* createEdge(TArgs&&... args) {
        return ir_factory_.createEdge<EdgeT>(std::forward<TArgs>(args)...);
    }

    template <typename EdgeT>
    auto cloneEdge(EdgeT&& edge) {
        return ir_factory_.cloneEdge<EdgeT>(std::forward<EdgeT>(edge));
    }

    template <typename BlobT, typename... TArgs>
    BlobT* createBlob(TArgs&&... args) {
        return ir_factory_.createBlob<BlobT>(std::forward<TArgs>(args)...);
    }

    template <typename BlobT>
    auto cloneBlob(BlobT&& blob) {
        return ir_factory_.cloneBlob<BlobT>(std::forward<BlobT>(blob));
    }

    template <typename NodeT, typename... TArgs>
    NodeT* createNode(TArgs&&... args) {
        return ir_factory_.createNode<NodeT>(std::forward<TArgs>(args)...);
    }

    template <typename NodeT>
    auto cloneNode(NodeT&& node) {
        return ir_factory_.cloneNode<NodeT>(std::forward<NodeT>(node));
    }

    void addNode(std::unique_ptr<Node> node) {
#ifndef NDEBUG
        checkExistingNode(*node);
#endif
        nodes_.push_back(std::move(node));
    }

    void addEdge(std::pair<EDGE_ID_T, std::unique_ptr<Edge>> edge) { edges_.insert(std::move(edge)); }

    void addBlob(std::unique_ptr<Blob> blob);

    NodesList::iterator insertNodeBefore(const Node& target, const Node& node) {
        auto node_owner = ir_factory_.takeNodeOwnership(&node);
        auto it         = nodes_.getNodeIterator(target);
#ifndef NDEBUG
        checkExistingNode(node);
#endif
        return nodes_.insert(it, std::move(node_owner));
    }

    NodesList::iterator insertNodeAfter(const Node& target, const Node& node) {
        auto node_owner = ir_factory_.takeNodeOwnership(&node);
        auto it         = nodes_.getNodeIterator(target);
#ifndef NDEBUG
        checkExistingNode(node);
#endif
        return nodes_.insert(std::next(it), std::move(node_owner));
    }

#include "private/node_iterator.inc"

    auto getNodes() { return iterator_range<NodeIt>(NodeIt(nodes_.begin()), NodeIt(nodes_.end())); }
    auto getNodes() const {
        return iterator_range<NodeConstIt>(NodeConstIt(nodes_.cbegin()), NodeConstIt(nodes_.cend()));
    }
    auto getNodesReverse() {
        return iterator_range<NodeReverseIt>(NodeReverseIt(nodes_.rbegin()), NodeReverseIt(nodes_.rend()));
    }
    auto getNodesReverse() const {
        return iterator_range<NodeConstReverseIt>(NodeConstReverseIt(nodes_.crbegin()),
                                                  NodeConstReverseIt(nodes_.crend()));
    }

    template <typename ValueT>
    auto getNodes() {
        return makeFilteredRange<ValueT>(getNodes());
    }

    template <typename ValueT>
    auto getNodes() const {
        return makeFilteredRange<const ValueT>(getNodes());
    }

    template <typename PredT>
    auto getNodes(const PredT& pred) {
        return makeUnaryPredFilteredRange<Node>(getNodes(), pred);
    }

    template <typename PredT>
    auto getNodes(const PredT& pred) const {
        return makeUnaryPredFilteredRange<const Node>(getNodes(), pred);
    }

    template <typename ValueT, typename PredT>
    auto getNodes(const PredT& pred) {
        return makeUnaryPredFilteredRange<ValueT>(getNodes(), pred);
    }

    template <typename ValueT, typename PredT>
    auto getNodes(const PredT& pred) const {
        return makeUnaryPredFilteredRange<const ValueT>(getNodes(), pred);
    }

#include "private/edge_blob_iterator.inc"

    auto getEdges() { return iterator_range<EdgeIt>(EdgeIt(edges_.begin()), EdgeIt(edges_.end())); }
    auto getEdges() const {
        return iterator_range<EdgeConstIt>(EdgeConstIt(edges_.cbegin()), EdgeConstIt(edges_.cend()));
    }

    template <typename ValueT>
    auto getEdges() {
        return makeFilteredRange<ValueT>(getEdges());
    }

    template <typename ValueT>
    auto getEdges() const {
        return makeFilteredRange<const ValueT>(getEdges());
    }

    template <typename PredT>
    auto getEdges(const PredT& pred) {
        return makeUnaryPredFilteredRange<Edge, EdgeIt, PredT>(getEdges(), pred);
    }

    template <typename PredT>
    auto getEdges(const PredT& pred) const {
        return makeUnaryPredFilteredRange<const Edge, EdgeConstIt, PredT>(getEdges(), pred);
    }

    template <typename ValueT, typename PredT>
    auto getEdges(const PredT& pred) {
        return makeUnaryPredFilteredRange<ValueT>(getEdges(), pred);
    }

    template <typename ValueT, typename PredT>
    auto getEdges(const PredT& pred) const {
        return makeUnaryPredFilteredRange<const ValueT>(getEdges(), pred);
    }

    auto getBlobs() { return iterator_range<BlobIt>(BlobIt(blobs_.begin()), BlobIt(blobs_.end())); }
    auto getBlobs() const {
        return iterator_range<BlobConstIt>(BlobConstIt(blobs_.cbegin()), BlobConstIt(blobs_.cend()));
    }

    Node* getNode(NODE_ID_T id) const {
        auto it = id_to_node_.find(id);
        Log::IR::E_IF(it == id_to_node_.end()) << "NNIR::getNode: node #" << id << " doesn't exist";
        return it->second;
    }

    Edge* getEdge(EDGE_ID_T id) const {
        auto it = edges_.find(id);
        if (it == edges_.end()) {
            // WORKAROUND: Avoid using E_IF() here because unconditional instantiation of a stream appears
            // to take some time, and on large networks it substantially increases compilation time. This
            // causes FlowNetS_1024_1920_A16W8 +FLC to timeout on CI.
            // TODO(m-cherkashin): Solve the problem by re-introducing macros
            Log::IR::E() << "NNIR::getEdge: edge #" << id << " doesn't exist";
        }
        return it->second.get();
    }

    Blob* getBlob(BLOB_ID_T id) const {
        auto find_it = blobs_.find(id);
        if (find_it == blobs_.end()) {
            // WORKAROUND: See getEdge()
            Log::IR::E() << "NNIR::getBlob: blob #" << id << " doesn't exist";
        }
        return find_it->second.get();
    }

    bool haveBlob(BLOB_ID_T id) const { return estd::contains(blobs_, id); }

    std::size_t getNodeCount() const { return nodes_.size(); }
    NODE_ID_T   getNextNodeId() const { return ++next_node_id_; }
    NODE_ID_T   getNextEdgeId() const { return ++next_edge_id_; }
    NODE_ID_T   getNextBlobId() const { return ++next_blob_id_; }
    NODE_ID_T   getNextStepId() const { return ++next_step_id_; }
    NODE_ID_T   getNextInstrId() const { return ++next_instr_id_; }

    NODE_ID_T getCurrentMaxNodeId() const { return next_node_id_; }

    void setNextNodeId(NODE_ID_T id) { next_node_id_ = (id > next_node_id_ ? id : next_node_id_); }
    void setNextEdgeId(EDGE_ID_T id) { next_edge_id_ = (id > next_edge_id_ ? id : next_edge_id_); }
    void setNextBlobId(BLOB_ID_T id) { next_blob_id_ = (id > next_blob_id_ ? id : next_blob_id_); }

    GRAPH_ID_T  getId() const { return id_; }
    std::string getName() const { return name_; }

    void clearNodes() { nodes_.clear(); }

    const NodesList& getNodesList() const { return nodes_; }
    NodesList&       getNodesList() { return nodes_; }

    NodesList::iterator deleteNode(const Node& target);
    void                deleteEdge(EDGE_ID_T id) { edges_.erase(id); }
    void                deleteBlob(BLOB_ID_T id) { blobs_.erase(id); }

    void                            addExecutionStep(STEP_ID_T step_id) { execution_steps_.push_back(step_id); }
    const std::vector<STEP_ID_T>&   getExecutionSteps(void) const { return execution_steps_; }
    const nn_ir::Node*              getIfmNodeForPropagation(const NODE_ID_T node_id) const;
    const nn_ir::Node*              getOfmNodeForPropagation(const NODE_ID_T node_id) const;
    std::vector<const nn_ir::Node*> findMultipleOutputEdgeNodes(iterator_range<nn_ir::NNIR::NodeIt> nodes);
    std::vector<const nn_ir::Node*> findMultipleInputEdgeNodes(iterator_range<nn_ir::NNIR::NodeIt> nodes);

    std::vector<nn_ir::Edge*> findInEdges(iterator_range<nn_ir::NNIR::EdgeIt> edges);
    std::vector<nn_ir::Edge*> findOutEdges(iterator_range<nn_ir::NNIR::EdgeIt> edges);

    NO_DISCARD bool isNodesNumberingValid() const { return is_nodes_numbering_valid_; }

    void numberNodes() const {
        uint32_t exec_number = 0;
        for (auto& node : nodes_) {
            node.exec_number_ = exec_number++;
        }
        is_nodes_numbering_valid_ = true;
    }

 private:
    template <typename T>
    friend struct NNIRIntrusiveListTraits;

    static void updateNodeDataOnAdd(const NNIR& graph, Node& node) {
        Log::IR::E_IF(graph.id_to_node_.count(node.getId())) << "attempt to insert existing node in graph: " << node;
        graph.id_to_node_.emplace(node.getId(), &node);
        graph.is_nodes_numbering_valid_ = false;
    }

    static void updateNodeDataOnDelete(const NNIR& graph, Node& node) {
        Log::IR::E_IF(!graph.id_to_node_.count(node.getId())) << "attempt to delete missing node in graph: " << node;
        graph.id_to_node_.erase(node.getId());
        // FIXME: actually it's not necessary but optimized_scheduler assumes that exec numbers of nodes are sequential
        graph.is_nodes_numbering_valid_ = false;
    }

    static void updateNodeDataOnMove(const NNIR& graph, Node& first_moved, Node& last_moved) {
        // Note. if needed it can be possible to make partial renumbering instead of full invalidation
        graph.is_nodes_numbering_valid_ = false;
    }

#ifndef NDEBUG
    void checkExistingNode(const Node& node) {
        Log::IR::E_IF(estd::contains(nodes_, node))
            << "NNIR::addNode:: attempting to insert node with existing id " << node.getId();
    }
#endif
    GRAPH_ID_T  id_;
    std::string name_;

    std::map<EDGE_ID_T, std::unique_ptr<Edge>> edges_;
    std::map<BLOB_ID_T, std::unique_ptr<Blob>> blobs_;

    std::vector<STEP_ID_T> execution_steps_;

    mutable NODE_ID_T  next_node_id_  = 0;
    mutable EDGE_ID_T  next_edge_id_  = 0;
    mutable BLOB_ID_T  next_blob_id_  = 0;
    mutable STEP_ID_T  next_step_id_  = 0;
    mutable INSTR_ID_T next_instr_id_ = 0;

    mutable std::unordered_map<NODE_ID_T, Node*> id_to_node_;
    mutable bool                                 is_nodes_numbering_valid_ = false;

    // Node list should keep the inserted order.
    // It must be the last field because it calls hooks of NNIR that touch
    // other fields and it's possible that hooks will be called in dtor of NNIR
    // and to prevent access to already destroyed fields we must guarantee that
    // dtor of NodesList will be called first. For this reason we register it last
    NodesList nodes_;

    /// @brief A simple class that encapsulates nodes/edges creation logic fo NNIR.
    ///        NNIR uses an instance of this object to hide implementation details from
    ///        its own createNode/createEdge methods that just proxy user request to corresponding
    ///        method of NNIRFactory object.
    class NNIRFactory {
     public:
        explicit NNIRFactory(NNIR& graph) : graph_(graph) {}

        /// @brief create an edge from given arguments and return pointer to user
        /// @return returns a pointer to the created edge
        template <typename EdgeT, typename... TArgs>
        EdgeT* createEdge(TArgs&&... args) {
            auto   new_edge                = std::make_unique<EdgeT>(std::forward<TArgs>(args)...);
            EdgeT* handle                  = new_edge.get();
            graph_.edges_[handle->getId()] = std::move(new_edge);
            return handle;
        }

        /// @return returns a pointer to the cloned edge
        template <typename EdgeT>
        auto cloneEdge(EdgeT&& edge) {
            auto new_edge = std::forward<EdgeT>(edge).clone();
            auto edgeId   = new_edge->getId();
            return static_cast<std::decay_t<EdgeT>*>(
                graph_.edges_.insert({edgeId, std::move(new_edge)}).first->second.get());
        }

        /// @brief create a blob from given arguments and return pointer to user
        /// @return returns a pointer to the created blob
        template <typename BlobT, typename... TArgs>
        BlobT* createBlob(TArgs&&... args) {
            auto new_blob = std::make_unique<BlobT>(std::forward<TArgs>(args)...);
            auto blobId   = new_blob->getId();
            return static_cast<std::decay_t<BlobT>*>(
                graph_.blobs_.insert({blobId, std::move(new_blob)}).first->second.get());
        }

        /// @return returns a pointer to the cloned blob
        template <typename BlobT>
        auto cloneBlob(BlobT&& blob) {
            auto new_blob = std::forward<BlobT>(blob).clone();
            auto blobId   = new_blob->getId();
            return static_cast<std::decay_t<BlobT>*>(
                graph_.blobs_.insert({blobId, std::move(new_blob)}).first->second.get());
        }

        /// @brief create a node from given arguments and return pointer to user
        /// @return returns a pointer to the created node
        template <typename NodeT, typename... TArgs>
        NodeT* createNode(TArgs&&... args) {
            billet_nodes_.emplace_back(std::make_unique<NodeT>(std::forward<TArgs>(args)...));
            return static_cast<NodeT*>(billet_nodes_.back().get());
        }

        /// @return returns a pointer to the cloned node
        template <typename NodeT>
        auto cloneNode(NodeT&& node) {
            billet_nodes_.emplace_back(std::forward<NodeT>(node).clone());
            return static_cast<std::decay_t<NodeT>*>(billet_nodes_.back().get());
        }

        /// @brief given a pointer to node, return an owning object to tranfer ownership from
        ///        NNIRFactory to NNIR.
        /// @returns unique_ptr to node
        std::unique_ptr<Node> takeNodeOwnership(const Node* node) {
            auto pool_it =
                estd::find_if(billet_nodes_, [node](const std::unique_ptr<Node>& n) { return node == n.get(); });
            Log::IR::E_IF(pool_it == billet_nodes_.end())
                << "NNIR::NNIRFactory::takeNodeOwnership:: attempting to insert a node that either had been inerted "
                << "before or hadn't created by createNode() or cloneNode() API: " << *node;

            return std::move(*pool_it);
        }

     private:
        // A pool of 'billet' nodes that are already created, but yet
        // not inserted into the graph. Used to hide ownership stuff
        // from user.
        // TODO(chefmax): this list won't be needed if we separate execution order in a separate list in graph.
        std::vector<std::unique_ptr<Node>> billet_nodes_;
        // A graph that owns the factory.
        NNIR& graph_;
    }; // class NNIRFactory

    NNIRFactory ir_factory_;
}; // class NNIR

} // namespace nn_ir
} // namespace nn_compiler
