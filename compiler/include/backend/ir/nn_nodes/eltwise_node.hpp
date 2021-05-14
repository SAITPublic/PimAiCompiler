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

#include "ir/ir_types.hpp"
#include "ir/nn_node.hpp"

#include "ir/ir_includes.hpp"

#include "common/common.hpp"
#include "common/types.hpp"

namespace nn_compiler::nn_ir {

class EltwiseNode : public NodeMixin<EltwiseNode, NNNode> {
 public:
    explicit EltwiseNode(const NodeInfo&            node_info,
                         EltwiseType                elt_type,
                         bool                       stable_prod_grad,
                         std::unique_ptr<ShiftNode> shift_node,
                         std::unique_ptr<ShiftNode> shift_in1_node,
                         std::unique_ptr<ShiftNode> shift_in2_node,
                         uint16_t                   multi_scale)
        : NodeMixin(node_info, NodeType::ELTWISE), elt_type_(elt_type), stable_prod_grad_(stable_prod_grad),
          shift_node_(std::move(shift_node)), shift_in1_node_(std::move(shift_in1_node)),
          shift_in2_node_(std::move(shift_in2_node)), multi_scale_(multi_scale) {}

    std::string getNodeTypeAsString() const override { return "Eltwise"; }

    bool        getStableProdGrad() const { return stable_prod_grad_; }
    EltwiseType getEltType() const { return elt_type_; }

    void setKernelNodeParameters(const KernelNodeParameters& kernel_node_parameters) {
        kernel_node_parameters_ = kernel_node_parameters;
    }

    KernelNodeParameters&       getKernelNodeParameters() { return kernel_node_parameters_; }
    const KernelNodeParameters& getKernelNodeParameters() const { return kernel_node_parameters_; }

    void setShiftNode(std::unique_ptr<ShiftNode> shift_node) { shift_node_ = std::move(shift_node); }
    void setShiftIn1Node(std::unique_ptr<ShiftNode> shift_node) { shift_in1_node_ = std::move(shift_node); }
    void setShiftIn2Node(std::unique_ptr<ShiftNode> shift_node) { shift_in2_node_ = std::move(shift_node); }
    void setEltType(EltwiseType elt_type) { elt_type_ = elt_type; }
    void setMultiScale(uint16_t multi_scale) { multi_scale_ = multi_scale; }

    const ShiftNode* getShiftNode() const { return shift_node_.get(); }
    const ShiftNode* getShiftIn1Node() const { return shift_in1_node_.get(); }
    const ShiftNode* getShiftIn2Node() const { return shift_in2_node_.get(); }
    ShiftNode*       getShiftNode() { return shift_node_.get(); }
    uint16_t         getMultiScale() const { return multi_scale_; }

    bool hasANNQuantization() const {
        const auto& in_edges = this->getInEdgeIds();
        Log::IR::E_IF(in_edges.size() < 2) << "Eltwise requires 2 input edges";
        const auto* ifm0_edge = getGraph().getEdge(in_edges[0]);
        const auto* ifm1_edge = getGraph().getEdge(in_edges[1]);
        const auto& ofm_edge  = this->getFirstOutEdge();

        const auto& ifm0_blob = cast<nn_ir::DataEdge>(ifm0_edge).getBlob();
        const auto& ifm1_blob = cast<nn_ir::DataEdge>(ifm1_edge).getBlob();
        const auto& ofm_blob  = cast<nn_ir::DataEdge>(ofm_edge).getBlob();
        return ifm0_blob->getFracLen().empty() && ifm1_blob->getFracLen().empty() && ofm_blob->getFracLen().empty();
    }

    EltwiseNode(const EltwiseNode&);
    EltwiseNode(EltwiseNode&&) = default;

 private:
    EltwiseType          elt_type_;
    bool                 stable_prod_grad_;
    KernelNodeParameters kernel_node_parameters_;

    std::unique_ptr<ShiftNode> shift_node_;
    std::unique_ptr<ShiftNode> shift_in1_node_;
    std::unique_ptr<ShiftNode> shift_in2_node_;
    uint16_t                   multi_scale_;
}; // class EltwiseNode
} // namespace nn_compiler::nn_ir
