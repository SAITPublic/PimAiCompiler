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
 * @file.    convolution_node.hpp
 * @brief.   This is ConvolutionNode class
 * @details. This header defines ConvolutionNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"

#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

#include "ir/include/ir_includes.hpp"

namespace nn_compiler {
namespace nn_ir {

class ConvolutionNode : public NodeMixin<ConvolutionNode, NNNode> {
 public:
    explicit ConvolutionNode(const NodeInfo&                 node_info,
                             std::unique_ptr<ActivationNode> activation_node,
                             std::unique_ptr<ShiftNode>      shift_node,
                             Shape2D                         kernel_size,
                             Shape2D                         stride_size,
                             Shape2D                         dilation_size,
                             Pad4                            padding_size,
                             BLOB_ID_T                       kernel_blob_id,
                             BLOB_ID_T                       bias_blob_id);

    std::string getNodeTypeAsString() const override { return "Convolution"; }

    void setKernelBlobId(BLOB_ID_T id) { kernel_blob_id_ = id; }
    void setBiasBlobId(BLOB_ID_T id) { bias_blob_id_ = id; }

    void setShiftNode(std::unique_ptr<ShiftNode> shift_node) { shift_node_ = std::move(shift_node); }
    void setActivationNode(std::unique_ptr<ActivationNode> activation_node) {
        activation_node_ = std::move(activation_node);
    }

    void setKernelNodeParameters(const KernelNodeParameters& kernel_node_parameters) {
        kernel_node_parameters_ = kernel_node_parameters;
    }

    KernelNodeParameters&       getKernelNodeParameters() { return kernel_node_parameters_; }
    const KernelNodeParameters& getKernelNodeParameters() const { return kernel_node_parameters_; }

    BLOB_ID_T getKernelBlobId() const { return kernel_blob_id_; }
    BLOB_ID_T getBiasBlobId() const { return bias_blob_id_; }
    Blob*     getKernelBlob() const { return getGraph().getBlob(kernel_blob_id_); }
    Blob* getBiasBlob() const { return (bias_blob_id_ == INVALID_ID) ? nullptr : getGraph().getBlob(bias_blob_id_); }

    const ActivationNode* getActivationNode() const { return activation_node_.get(); }
    const ShiftNode*      getShiftNode() const { return shift_node_.get(); }

    ShiftNode* getShiftNode() { return shift_node_.get(); }

    Shape4D getPreprocessedKernelBlobDim() const override { return getGraph().getBlob(kernel_blob_id_)->getShape(); }

    ConvolutionNode(const ConvolutionNode&);
    ConvolutionNode(ConvolutionNode&&) = default;

 private:
    std::unique_ptr<ActivationNode> activation_node_;
    std::unique_ptr<ShiftNode>      shift_node_;
    KernelNodeParameters            kernel_node_parameters_;
    BLOB_ID_T                       kernel_blob_id_;
    BLOB_ID_T                       bias_blob_id_;
}; // class ConvolutionNode

} // namespace nn_ir
} // namespace nn_compiler
