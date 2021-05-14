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
 * @brief.   This is DeconvolutionNode class
 * @details. This header defines DeconvolutionNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"
#include "ir/nn_node.hpp"

#include "ir/ir_includes.hpp"

namespace nn_compiler {
namespace nn_ir {

class DeconvolutionNode : public NodeMixin<DeconvolutionNode, NNNode> {
 public:
    explicit DeconvolutionNode(const NodeInfo&                 node_info,
                               std::unique_ptr<ActivationNode> activation_node,
                               std::unique_ptr<ShiftNode>      shift_node,
                               Shape2D                         kernel_size,
                               Shape2D                         stride_size,
                               Shape2D                         dilation_size,
                               Pad4                            padding_size,
                               BLOB_ID_T                       kernel_blob_id,
                               BLOB_ID_T                       bias_blob_id);

    std::string getNodeTypeAsString() const override { return "Deconvolution"; }

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
    ShiftNode*            getShiftNode() { return shift_node_.get(); }

    /// Methods and structures to represent Deconvolution as several Convolutions + BatchToSpace

    typedef struct DeconvAsConvsExecutionParameters_ {
        nn_ir::Shape2D deconv_ofm_dim;
        nn_ir::Shape2D conv_ofm_dim;
        nn_ir::Shape2D merged_convs_ofm_dim;
        nn_ir::Pad4    conv_actual_pad;
        nn_ir::Pad4    merged_convs_ofm_area_to_trim;
    } DeconvAsConvsExecutionParameters;

    nn_ir::Shape2D getConvolutionKernel(const nn_ir::Shape2D& deconv_kernel, const nn_ir::Shape2D deconv_stride) const {
        Log::IR::E_IF(!deconv_stride.isValid()) << "Zero stride was found in Deconvolution node";
        return {{.h = divUp(deconv_kernel.h, deconv_stride.h), .w = divUp(deconv_kernel.w, deconv_stride.w)}};
    }
    nn_ir::Shape2D getConvolutionKernel(const nn_ir::Shape4D& deconv_kernel, const nn_ir::Shape2D deconv_stride) const {
        return getConvolutionKernel(nn_ir::Shape2D{{.h = deconv_kernel.h, .w = deconv_kernel.w}}, deconv_stride);
    }
    nn_ir::Shape2D getConvolutionKernel() const {
        return getConvolutionKernel(kernel_node_parameters_.getKernelSize(), kernel_node_parameters_.getStrideSize());
    }
    DeconvAsConvsExecutionParameters
    getOfmAndPadsAsMultipleConvolutions(const nn_ir::Shape4D& ifm_dim, const nn_ir::KernelNodeParameters& params) const;

    DeconvolutionNode(const DeconvolutionNode&);
    DeconvolutionNode(DeconvolutionNode&&) = default;

 private:
    std::unique_ptr<ActivationNode> activation_node_;
    std::unique_ptr<ShiftNode>      shift_node_;
    KernelNodeParameters            kernel_node_parameters_;
    BLOB_ID_T                       kernel_blob_id_;
    BLOB_ID_T                       bias_blob_id_;
}; // class DeconvolutionNode

} // namespace nn_ir
} // namespace nn_compiler
