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

#include <memory>

namespace nn_compiler::nn_ir {

/// @brief this class represents CRTP mixin class for base nodes (NNNode, HWNode, QNode, OpNode, VNode)
///        to avoid boilerplate code
template <typename DerivedNodeT, typename BaseNodeT = void>
struct AbstractNodeMixin : public BaseNodeT {
 public:
    std::unique_ptr<DerivedNodeT> clone() const& {
        return std::unique_ptr<DerivedNodeT>(static_cast<DerivedNodeT*>(this->cloneImpl()));
    }
    std::unique_ptr<DerivedNodeT> clone() && {
        return std::unique_ptr<DerivedNodeT>(static_cast<DerivedNodeT*>(this->cloneImpl()));
    }

    /// @brief methods used by casting infrastructure
    template <typename T>
    static bool classof(const AbstractNodeMixin<Node, void>* node);

 protected:
    template <typename... Ts>
    explicit AbstractNodeMixin(Ts&&... args) : BaseNodeT(std::forward<Ts>(args)...) {
        static_assert(std::is_base_of_v<AbstractNodeMixin, DerivedNodeT>, "must be used via CRTP");
    }

 private:
    virtual AbstractNodeMixin* cloneImpl() const& = 0;
    virtual AbstractNodeMixin* cloneImpl() &&     = 0;
};

/// @brief this class represents CRTP mixin class for Node class to avoid boilerplate code
template <typename DerivedNodeT>
struct AbstractNodeMixin<DerivedNodeT, void> {
 public:
    std::unique_ptr<DerivedNodeT> clone() const& {
        return std::unique_ptr<DerivedNodeT>(static_cast<DerivedNodeT*>(this->cloneImpl()));
    }
    std::unique_ptr<DerivedNodeT> clone() && {
        return std::unique_ptr<DerivedNodeT>(static_cast<DerivedNodeT*>(this->cloneImpl()));
    }

 protected:
    AbstractNodeMixin() { static_assert(std::is_base_of_v<AbstractNodeMixin, DerivedNodeT>, "must be used via CRTP"); }

 private:
    virtual AbstractNodeMixin* cloneImpl() const& = 0;
    virtual AbstractNodeMixin* cloneImpl() &&     = 0;
};

/// @brief methods used by casting infrastructure
template <typename DerivedNodeT, typename BaseNodeT>
template <typename T>
bool AbstractNodeMixin<DerivedNodeT, BaseNodeT>::classof(const AbstractNodeMixin<Node, void>* node) {
    static_assert(std::is_same<T, DerivedNodeT>::value, "incorrect type");
#define PROCESS_BASE_NODE(NODE_TYPE, NODE_CLASS, BASE_NODE_CLASS)                                        \
    if constexpr (std::is_same_v<T, NODE_CLASS>) {                                                       \
        auto node_type = static_cast<const DerivedNodeT*>(node)->getNodeType();                          \
        return node_type >= nn_ir::NodeType::NODE_TYPE && node_type <= nn_ir::NodeType::Last##NODE_TYPE; \
    }
#include "ir/include/nodes.def"
    Log::IR::E() << "Unreachable code!";
}

/// @brief this class represents CRTP mixin class for concrete nodes
///        (like Convolution, EltWise etc) to avoid boilerplate code
template <typename DerivedNodeT, typename BaseNodeT>
class NodeMixin : public BaseNodeT {
 public:
    std::unique_ptr<DerivedNodeT> clone() const& {
        return std::unique_ptr<DerivedNodeT>(static_cast<DerivedNodeT*>(this->cloneImpl()));
    }
    std::unique_ptr<DerivedNodeT> clone() && {
        return std::unique_ptr<DerivedNodeT>(static_cast<DerivedNodeT*>(this->cloneImpl()));
    }

    /// @brief methods used by casting infrastructure
    template <typename T>
    static bool classof(const AbstractNodeMixin<Node, void>* node) {
        static_assert(std::is_same<T, DerivedNodeT>::value, "incorrect type");
#define PROCESS_NODE(NODE_TYPE, NODE_CLASS, BASE_NODE_CLASS)                                        \
    if constexpr (std::is_same_v<DerivedNodeT, NODE_CLASS>) {                                       \
        return static_cast<const DerivedNodeT*>(node)->getNodeType() == nn_ir::NodeType::NODE_TYPE; \
    }
#include "ir/include/nodes.def"
        Log::IR::E() << "Unreachable code!";
    }

 protected:
    template <typename... Ts>
    explicit NodeMixin(Ts&&... args) : BaseNodeT(std::forward<Ts>(args)...) {
        static_assert(std::is_base_of_v<NodeMixin, DerivedNodeT>, "must be used via CRTP");
    }

 private:
    NodeMixin* cloneImpl() const& override { return new DerivedNodeT(static_cast<const DerivedNodeT&>(*this)); }
    NodeMixin* cloneImpl() && override { return new DerivedNodeT(static_cast<DerivedNodeT&&>(*this)); }
};

} // namespace nn_compiler::nn_ir
