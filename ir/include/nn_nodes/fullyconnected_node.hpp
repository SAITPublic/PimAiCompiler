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
 * @file.    fullyconnected_node.hpp
 * @brief.   This is FullyConnectedNode class
 * @details. This header defines FullyConnectedNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"

#include "ir/nn_ir.hpp"
#include "ir/nn_node.hpp"

#include "ir/ir_includes.hpp"

namespace nn_compiler {
namespace nn_ir {

class FullyConnectedNode : public NodeMixin<FullyConnectedNode, NNNode> {
 public:
    explicit FullyConnectedNode(const NodeInfo&                 node_info,
                                std::unique_ptr<ActivationNode> activation_node,
                                std::unique_ptr<ShiftNode>      shift_node,
                                nn_ir::Axis                     axis,
                                bool                            transpose,
                                BLOB_ID_T                       weight_blob_id,
                                BLOB_ID_T                       bias_blob_id);

    std::string getNodeTypeAsString() const override { return "FullyConnected"; }

    void setKernelBlobId(BLOB_ID_T id) { kernel_blob_id_ = id; }
    void setBiasBlobId(BLOB_ID_T id) { bias_blob_id_ = id; }
    void setActivationNode(std::unique_ptr<ActivationNode> activation_node) {
        activation_node_ = std::move(activation_node);
    }
    void setShiftNode(std::unique_ptr<ShiftNode> shift_node) { shift_node_ = std::move(shift_node); }

    const nn_ir::Axis& getAxis() const { return axis_; }
    const bool&        getTranspose() const { return transpose_; }
    BLOB_ID_T          getKernelBlobId() const { return kernel_blob_id_; }
    BLOB_ID_T          getBiasBlobId() const { return bias_blob_id_; }
    Blob*              getKernelBlob() const { return getGraph().getBlob(kernel_blob_id_); }
    Blob* getBiasBlob() const { return (bias_blob_id_ == INVALID_ID) ? nullptr : getGraph().getBlob(bias_blob_id_); }

    const ActivationNode* getActivationNode() const { return activation_node_.get(); }
    const ShiftNode*      getShiftNode() const { return shift_node_.get(); }
    ShiftNode*            getShiftNode() { return shift_node_.get(); }

    FullyConnectedNode(const FullyConnectedNode&);
    FullyConnectedNode(FullyConnectedNode&&) = default;

 private:
    std::unique_ptr<ActivationNode> activation_node_;
    std::unique_ptr<ShiftNode>      shift_node_;
    nn_ir::Axis                     axis_;
    bool                            transpose_;
    BLOB_ID_T                       kernel_blob_id_;
    BLOB_ID_T                       bias_blob_id_;
}; // class FullyConnectedNode

} // namespace nn_ir
} // namespace nn_compiler
