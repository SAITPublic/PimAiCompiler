/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/include/nn_nodes/deconvolution_node.hpp"
#include "ir/include/common/log.hpp"

#include "common/include/arithmetics.hpp"

namespace nn_compiler {
namespace nn_ir {

DeconvolutionNode::DeconvolutionNode(const NodeInfo&                 node_info,
                                     std::unique_ptr<ActivationNode> activation_node,
                                     std::unique_ptr<ShiftNode>      shift_node,
                                     Shape2D                         kernel_size,
                                     Shape2D                         stride_size,
                                     Shape2D                         dilation_size,
                                     Pad4                            padding_size,
                                     BLOB_ID_T                       kernel_blob_id,
                                     BLOB_ID_T                       bias_blob_id)
    : NodeMixin(node_info, NodeType::DECONVOLUTION), activation_node_(std::move(activation_node)),
      shift_node_(std::move(shift_node)),
      kernel_node_parameters_(padding_size, kernel_size, stride_size, dilation_size), kernel_blob_id_(kernel_blob_id),
      bias_blob_id_(bias_blob_id) {}

DeconvolutionNode::DeconvolutionNode(const DeconvolutionNode& copied_deconv)
    : NodeMixin(copied_deconv),
      activation_node_(copied_deconv.activation_node_ ? copied_deconv.activation_node_->clone() : nullptr),
      shift_node_(copied_deconv.shift_node_->clone()), kernel_node_parameters_(copied_deconv.kernel_node_parameters_),
      kernel_blob_id_(copied_deconv.kernel_blob_id_), bias_blob_id_(copied_deconv.bias_blob_id_) {}

DeconvolutionNode::DeconvAsConvsExecutionParameters
DeconvolutionNode::getOfmAndPadsAsMultipleConvolutions(const nn_ir::Shape4D&              ifm_dim,
                                                       const nn_ir::KernelNodeParameters& params) const {
    const auto deconv_stride = params.getStrideSize();
    const auto deconv_kernel = params.getOriginKernelSize();
    const auto deconv_pad    = params.getPaddingSize();

    // Get corresponding Convolution kernel
    const auto conv_kernel_for_execution = getConvolutionKernel(deconv_kernel, deconv_stride);
    const auto conv_ops_numbers          = deconv_stride;

    /* First, calculate everything for pad="full" */
    // The mathematically expected OFM shape with pad="full", not trimmed via "pad" parameter yet
    const auto deconv_ofm_untrimmed_with_full_pad =
        nn_ir::Shape2D{{.h = deconv_stride.h * (ifm_dim.h - 1) + deconv_kernel.h,
                        .w = deconv_stride.w * (ifm_dim.w - 1) + deconv_kernel.w}};
    // The mathematically expected final OFM shape after trimming with "pad" parameter
    const auto deconv_ofm = nn_ir::Shape2D{{.h = deconv_ofm_untrimmed_with_full_pad.h - (deconv_pad.t + deconv_pad.b),
                                            .w = deconv_ofm_untrimmed_with_full_pad.w - (deconv_pad.r + deconv_pad.l)}};
    // The padding value required to run Convolutions with reformatted kernels to get full OFM
    const auto conv_full_pad = nn_ir::Pad4{.t = conv_kernel_for_execution.h - 1,
                                           .b = conv_kernel_for_execution.h - 1,
                                           .l = conv_kernel_for_execution.w - 1,
                                           .r = conv_kernel_for_execution.w - 1};
    const auto conv_ofm_with_full_pad =
        nn_ir::Shape2D{{.h = (ifm_dim.h + conv_full_pad.t + conv_full_pad.b) - conv_kernel_for_execution.h + 1,
                        .w = (ifm_dim.w + conv_full_pad.l + conv_full_pad.r) - conv_kernel_for_execution.w + 1}};
    const auto merged_conv_ofms_with_full_pads = nn_ir::Shape2D{
        {.h = conv_ofm_with_full_pad.h * conv_ops_numbers.h, .w = conv_ofm_with_full_pad.w * conv_ops_numbers.w}};
    // If multiple Convolutions produce more data than necessary, reconfigure Pad parameters to trim the excess
    auto merged_conv_ofms_with_full_pads_area_to_trim = deconv_pad;
    if (merged_conv_ofms_with_full_pads.w > deconv_ofm_untrimmed_with_full_pad.w) {
        const auto extra_pixels = merged_conv_ofms_with_full_pads.w - deconv_ofm_untrimmed_with_full_pad.w;
        merged_conv_ofms_with_full_pads_area_to_trim.r += extra_pixels;
    }
    if (merged_conv_ofms_with_full_pads.h > deconv_ofm_untrimmed_with_full_pad.h) {
        const auto extra_pixels = merged_conv_ofms_with_full_pads.h - deconv_ofm_untrimmed_with_full_pad.h;
        merged_conv_ofms_with_full_pads_area_to_trim.b += extra_pixels;
    }

    /* We could just generate the whole OFM with "full" IFM padding and then trim the excess but it's inefficient
     * Let's try to reduce the IFM zero-padding thus generate smaller OFM so there'd be less to trim after all
     * Each step of (stride * stride) Convolution kernels would generate (stride * stride) OFM pixels group
     * So, the area expected to be trimmed (Deconvolution "pad" parameter) could be reduced by any integer (N * stride)
     */
    const auto trimmable_ofm_border_pixels =
        nn_ir::Pad4{.t = alignDown(merged_conv_ofms_with_full_pads_area_to_trim.t, conv_ops_numbers.h),
                    .b = alignDown(merged_conv_ofms_with_full_pads_area_to_trim.b, conv_ops_numbers.h),
                    .l = alignDown(merged_conv_ofms_with_full_pads_area_to_trim.l, conv_ops_numbers.w),
                    .r = alignDown(merged_conv_ofms_with_full_pads_area_to_trim.r, conv_ops_numbers.w)};
    // We can only reduce OFM pixels by non-creating them at all, so non-adding extra zero-padding in advance
    const auto reduceable_conv_zero_pads = nn_ir::Pad4{.t = conv_full_pad.t * conv_ops_numbers.h,
                                                       .b = conv_full_pad.b * conv_ops_numbers.h,
                                                       .l = conv_full_pad.l * conv_ops_numbers.w,
                                                       .r = conv_full_pad.r * conv_ops_numbers.w};
    // We should reduce the OFM pixels produced at its borders as min(wanted, possible)
    const auto border_pixels_to_reduce =
        nn_ir::Pad4{.t = std::min(trimmable_ofm_border_pixels.t, reduceable_conv_zero_pads.t),
                    .b = std::min(trimmable_ofm_border_pixels.b, reduceable_conv_zero_pads.b),
                    .l = std::min(trimmable_ofm_border_pixels.l, reduceable_conv_zero_pads.l),
                    .r = std::min(trimmable_ofm_border_pixels.r, reduceable_conv_zero_pads.r)};
    // The padding value required to run Convolutions with reformatted kernels to get the same result
    const auto conv_reduced_pad = nn_ir::Pad4{.t = conv_full_pad.t - border_pixels_to_reduce.t / conv_ops_numbers.h,
                                              .b = conv_full_pad.b - border_pixels_to_reduce.b / conv_ops_numbers.h,
                                              .l = conv_full_pad.l - border_pixels_to_reduce.l / conv_ops_numbers.w,
                                              .r = conv_full_pad.r - border_pixels_to_reduce.r / conv_ops_numbers.w};

    // Calculate OFM shape for Convolutions with those paddings and kernel
    const auto conv_ofm_with_reduced_pad =
        nn_ir::Shape2D{{.h = (ifm_dim.h + conv_reduced_pad.t + conv_reduced_pad.b) - conv_kernel_for_execution.h + 1,
                        .w = (ifm_dim.w + conv_reduced_pad.l + conv_reduced_pad.r) - conv_kernel_for_execution.w + 1}};
    // Get the OFM produced by all Convolutions
    const auto merged_conv_ofms_with_reduced_pads = nn_ir::Shape2D{
        {.h = conv_ofm_with_reduced_pad.h * conv_ops_numbers.h, .w = conv_ofm_with_reduced_pad.w * conv_ops_numbers.w}};

    // If multiple Convolutions produce more data than necessary, reconfigure Pad parameters to trim the excess
    auto merged_convs_ofm_area_to_trim =
        nn_ir::Pad4{.t = merged_conv_ofms_with_full_pads_area_to_trim.t - border_pixels_to_reduce.t,
                    .b = merged_conv_ofms_with_full_pads_area_to_trim.b - border_pixels_to_reduce.b,
                    .l = merged_conv_ofms_with_full_pads_area_to_trim.l - border_pixels_to_reduce.l,
                    .r = merged_conv_ofms_with_full_pads_area_to_trim.r - border_pixels_to_reduce.r};
    Log::IR::E_IF((merged_convs_ofm_area_to_trim.t / conv_ops_numbers.h > 0 && conv_reduced_pad.t > 0) ||
                  (merged_convs_ofm_area_to_trim.b / conv_ops_numbers.h > 0 && conv_reduced_pad.b > 0) ||
                  (merged_convs_ofm_area_to_trim.l / conv_ops_numbers.w > 0 && conv_reduced_pad.l > 0) ||
                  (merged_convs_ofm_area_to_trim.r / conv_ops_numbers.w > 0 && conv_reduced_pad.r > 0))
        << "There's some useless area left to be trimmed which we could not even produce";
    return {.deconv_ofm_dim                = deconv_ofm,
            .conv_ofm_dim                  = conv_ofm_with_reduced_pad,
            .merged_convs_ofm_dim          = merged_conv_ofms_with_reduced_pads,
            .conv_actual_pad               = conv_reduced_pad,
            .merged_convs_ofm_area_to_trim = merged_convs_ofm_area_to_trim};
}

} // namespace nn_ir
} // namespace nn_compiler
