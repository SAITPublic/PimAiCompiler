/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "ir/hw_node.hpp"
#include "ir/ir_types.hpp"

#include "ir/ir_includes.hpp"

#include "common/common.hpp"
#include "common/types.hpp"

namespace nn_compiler::nn_ir {

class MAAEltwiseNode : public NodeMixin<MAAEltwiseNode, HWNode> {
 public:
    explicit MAAEltwiseNode(const NodeInfo&                 node_info,
                            EltwiseType                     elt_type,
                            bool                            stable_prod_grad,
                            std::unique_ptr<ShiftNode>      shift_node,
                            std::unique_ptr<ActivationNode> activation_node,
                            BLOB_ID_T                       kernel_blob_id,
                            BLOB_ID_T                       bias_blob_id)
        : NodeMixin(node_info, NodeType::MAAELTWISE), elt_type_(elt_type), stable_prod_grad_(stable_prod_grad),
          shift_node_(std::move(shift_node)), activation_node_(std::move(activation_node)),
          kernel_blob_id_(kernel_blob_id), bias_blob_id_(bias_blob_id) {}

    std::string getNodeTypeAsString() const override { return "MAAEltwise"; }

    bool        getStableProdGrad() const { return stable_prod_grad_; }
    EltwiseType getEltType() const { return elt_type_; }

    /// @brief method used by casting infrastructure
    template <typename T>
    static bool classof(const Node* node) {
        static_assert(std::is_same<T, MAAEltwiseNode>::value, "incorrect type");
        return node->getNodeType() == NodeType::MAAELTWISE;
    }

    void setKernelNodeParameters(const KernelNodeParameters& kernel_node_parameters) {
        kernel_node_parameters_ = kernel_node_parameters;
    }

    KernelNodeParameters&       getKernelNodeParameters() { return kernel_node_parameters_; }
    const KernelNodeParameters& getKernelNodeParameters() const { return kernel_node_parameters_; }

    BLOB_ID_T getKernelBlobId() const { return kernel_blob_id_; }
    BLOB_ID_T getBiasBlobId() const { return bias_blob_id_; }
    Blob*     getKernelBlob() const { return getGraph().getBlob(kernel_blob_id_); }
    Blob*     getBiasBlob() const { return getGraph().getBlob(bias_blob_id_); }

    void setKernelBlobId(BLOB_ID_T id) { kernel_blob_id_ = id; }
    void setBiasBlobId(BLOB_ID_T id) { bias_blob_id_ = id; }
    void setShiftNode(std::unique_ptr<ShiftNode> shift_node) { shift_node_ = std::move(shift_node); }
    void setEltType(EltwiseType elt_type) { elt_type_ = elt_type; }
    void setActivationNode(std::unique_ptr<ActivationNode> activation_node) {
        activation_node_ = std::move(activation_node);
    }

    const ShiftNode* getShiftNode() const { return shift_node_.get(); }
    ShiftNode*       getShiftNode() { return shift_node_.get(); }

    const ActivationNode* getActivationNode() const { return activation_node_.get(); }

    MAAEltwiseNode(const MAAEltwiseNode&);
    MAAEltwiseNode(MAAEltwiseNode&&) = default;
    explicit MAAEltwiseNode(EltwiseNode&);

 private:
    EltwiseType          elt_type_;
    bool                 stable_prod_grad_;
    KernelNodeParameters kernel_node_parameters_;

    std::unique_ptr<ShiftNode>      shift_node_;
    std::unique_ptr<ActivationNode> activation_node_;
    BLOB_ID_T                       kernel_blob_id_;
    BLOB_ID_T                       bias_blob_id_;
}; // class MAAEltwiseNode
} // namespace nn_compiler::nn_ir
