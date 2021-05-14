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

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/all_nodes.hpp"

namespace nn_compiler::nn_ir {

/// @brief this class implements static visitor via CRTP.
///        Every derived class must hide necessary visit* functions that are named by the following pattern:
///           visit'NodeClass' where NodeClass is name of specific node class
///        Users have to use visitor classes as follows:
///          ```
///             SomeVisitor v;
///             v.visit(graph); // visit all graph nodes
///             // or
///             std::vector<nn_ir::Node*> nodes;
///             v.visit(nodes.begin(), nodes.end());
///             // or
///             void f(nn_ir::Node& node) {
///                SomeVisitor v;
///                v.visit(node);
//              }
///          ```
///        if 'is_const' parameter == true then visit* functions will take node parameter
///        by const reference 'RetT' parameter defines return type of visit* functions
template <typename DerivedT, bool is_const = false, typename RetT = void>
class Visitor {
 private:
    // declare wrapper types for graph and for each node depends on is_const parameter
    using GraphType = std::conditional_t<is_const, const NNIR, NNIR>;
    using NodeType  = std::conditional_t<is_const, const Node, Node>;

#define PROCESS_NODE(NODE_TYPE, NODE_CLASS, BASE_NODE_CLASS) \
    using NODE_CLASS##Type = std::conditional_t<is_const, const NODE_CLASS, NODE_CLASS>;
#include "ir/nodes.def"

 public:
    /// @brief this function invokes specific visit* function depends on real type of node
    RetT visit(NodeType& node) {
        static_assert(std::is_base_of_v<Visitor, DerivedT>, "must be used via CRTP");

        switch (node.getNodeType()) {
            default:
                Log::IR::E() << "Unknown node type: " << node;
#define PROCESS_NODE(NODE_TYPE, NODE_CLASS, BASE_NODE_CLASS) \
    case nn_ir::NodeType::NODE_TYPE:                         \
        return static_cast<DerivedT*>(this)->visit##NODE_CLASS(static_cast<NODE_CLASS##Type&>(node));
#include "ir/nodes.def"
        }
    }

    // generate all visit* functions (for each node declared in nodes.def)
    // these functions can be "overridden" (hidden) by inherited classes
#define PROCESS_NODE(NODE_TYPE, NODE_CLASS, BASE_NODE_CLASS)               \
    RetT visit##NODE_CLASS(NODE_CLASS##Type& node) {                       \
        return static_cast<DerivedT*>(this)->visit##BASE_NODE_CLASS(node); \
    }
#include "ir/nodes.def"

    /// @brief the top level visit function that will be invoked by all visit*
    /// functions if they were not hidden in derived class
    ///
    /// @note if RetT != void this function must be hidden in derived class to return necessary value
    void visitNode(NodeType& node) {}

    /// @brief visit nodes within [first, last) range
    template <typename InputIteratorT>
    void visit(InputIteratorT first, InputIteratorT last) {
        for (; first != last; ++first) {
            visit(*first);
        }
    }

    /// @brief visit all nodes of graph
    void visit(GraphType& graph) {
        auto nodes = graph.getNodes();
        visit(nodes.begin(), nodes.end());
    }
};

} // namespace nn_compiler::nn_ir
