/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/include/ir_nnnode_parser.hpp"
#include "ir/include/common/log.hpp"
#include "ir/include/ir_includes.hpp"
#include "ir/include/ir_tools.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_InputNode>(const IR::NnNode*      ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto input_node = ir_node->nn_node_as_InputNode();
    Log::IR::E_IF(input_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::InputNode>() => wrong node type!";

    nn_ir::IR_Node_Config_Type_ ir_type = input_node->type();
    auto                        ir_mean = input_node->mean();
    float                       scale   = input_node->scale();
    bool                        mirror  = input_node->mirror();

    nn_ir::InputType input_type = std::get<nn_ir::InputType>(nn_ir::parseConfigType(ir_type));

    std::vector<float> mean;
    if (ir_mean != nullptr) {
        auto raw_data  = ir_mean->data();
        auto data_size = ir_mean->size();
        mean.assign(raw_data, raw_data + data_size);
    }

    return std::make_unique<nn_ir::InputNode>(node_info, input_type, mean, scale, mirror);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_ConvNode>(const IR::NnNode* ir_node, const nn_ir::NodeInfo& node_info) {
    auto conv_node = ir_node->nn_node_as_ConvNode();
    Log::IR::E_IF(conv_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::ConvNode>() => wrong node type!";

    auto act_node   = getActNode(node_info, conv_node->activation());
    auto shift_node = getShiftNode(node_info, conv_node->shift());

    nn_ir::Shape2D kernel_size   = std::get<nn_ir::Shape2D>(nn_ir::parseParam(conv_node->kernel_size()));
    nn_ir::Shape2D stride_size   = std::get<nn_ir::Shape2D>(nn_ir::parseParam(conv_node->stride_size()));
    nn_ir::Shape2D dilation_size = std::get<nn_ir::Shape2D>(nn_ir::parseParam(conv_node->dilation_size()));
    nn_ir::Pad4    padding_size  = std::get<nn_ir::Pad4>(nn_ir::parseParam(conv_node->padding_size()));

    BLOB_ID_T kernel_blob_id = conv_node->kernel_blob_id();
    BLOB_ID_T bias_blob_id   = conv_node->bias_blob_id();

    return std::make_unique<nn_ir::ConvolutionNode>(node_info,
                                                    std::move(act_node),
                                                    std::move(shift_node),
                                                    kernel_size,
                                                    stride_size,
                                                    dilation_size,
                                                    padding_size,
                                                    kernel_blob_id,
                                                    bias_blob_id);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_DeConvNode>(const IR::NnNode*      ir_node,
                                                            const nn_ir::NodeInfo& node_info) {
    auto deconv_node = ir_node->nn_node_as_DeConvNode();
    Log::IR::E_IF(deconv_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::DeConvNode>() => wrong node type!";

    auto act_node   = getActNode(node_info, deconv_node->activation());
    auto shift_node = getShiftNode(node_info, deconv_node->shift());

    nn_ir::Shape2D kernel_size   = std::get<nn_ir::Shape2D>(nn_ir::parseParam(deconv_node->kernel_size()));
    nn_ir::Shape2D stride_size   = std::get<nn_ir::Shape2D>(nn_ir::parseParam(deconv_node->stride_size()));
    nn_ir::Shape2D dilation_size = std::get<nn_ir::Shape2D>(nn_ir::parseParam(deconv_node->dilation_size()));
    nn_ir::Pad4    padding_size  = std::get<nn_ir::Pad4>(nn_ir::parseParam(deconv_node->padding_size()));

    BLOB_ID_T kernel_blob_id = deconv_node->kernel_blob_id();
    BLOB_ID_T bias_blob_id   = deconv_node->bias_blob_id();

    return std::make_unique<nn_ir::DeconvolutionNode>(node_info,
                                                      std::move(act_node),
                                                      std::move(shift_node),
                                                      kernel_size,
                                                      stride_size,
                                                      dilation_size,
                                                      padding_size,
                                                      kernel_blob_id,
                                                      bias_blob_id);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_PoolNode>(const IR::NnNode* ir_node, const nn_ir::NodeInfo& node_info) {
    auto pool_node = ir_node->nn_node_as_PoolNode();
    Log::IR::E_IF(pool_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::PoolNode>() => wrong node type!";

    nn_ir::IR_Node_Config_Type_ ir_pool_type     = pool_node->type();
    nn_ir::IR_Node_Config_Type_ ir_pad_calc_type = pool_node->pad_calc();

    nn_ir::PoolType    pool_type     = std::get<nn_ir::PoolType>(nn_ir::parseConfigType(ir_pool_type));
    nn_ir::Shape2D     kernel_size   = std::get<nn_ir::Shape2D>(nn_ir::parseParam(pool_node->kernel_size()));
    nn_ir::Shape2D     stride_size   = std::get<nn_ir::Shape2D>(nn_ir::parseParam(pool_node->stride_size()));
    nn_ir::Shape2D     dilation_size = std::get<nn_ir::Shape2D>(nn_ir::parseParam(pool_node->dilation_size()));
    nn_ir::Pad4        padding_size  = std::get<nn_ir::Pad4>(nn_ir::parseParam(pool_node->padding_size()));
    nn_ir::PadCalcType pad_calc_type = std::get<nn_ir::PadCalcType>(nn_ir::parseConfigType(ir_pad_calc_type));

    return std::make_unique<nn_ir::PoolNode>(
        node_info, pool_type, kernel_size, stride_size, dilation_size, padding_size, pad_calc_type);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_ActivationNode>(const IR::NnNode*      ir_node,
                                                                const nn_ir::NodeInfo& node_info) {
    auto act_node = ir_node->nn_node_as_ActivationNode();
    Log::IR::E_IF(act_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::ActivationNode>() => wrong node type!";

    nn_ir::IR_Node_Config_Type_ ir_type  = act_node->type();
    nn_ir::ActivationType       act_type = std::get<nn_ir::ActivationType>(nn_ir::parseConfigType(ir_type));

    float slope          = act_node->slope();
    float negative_slope = act_node->negative_slope();
    float min            = act_node->min();
    float max            = act_node->max();

    auto shift_node = getShiftNode(node_info, act_node->shift());
    return std::make_unique<nn_ir::ActivationNode>(
        node_info, std::move(shift_node), act_type, slope, negative_slope, min, max);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_ConcatNode>(const IR::NnNode*      ir_node,
                                                            const nn_ir::NodeInfo& node_info) {
    auto concat_node = ir_node->nn_node_as_ConcatNode();
    Log::IR::E_IF(concat_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::ConcatNode>() => wrong node type!";

    const auto axis = static_cast<nn_ir::Axis>(concat_node->axis());

    return std::make_unique<nn_ir::ConcatNode>(node_info, axis);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_SoftmaxNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto softmax_node = ir_node->nn_node_as_SoftmaxNode();
    Log::IR::E_IF(softmax_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::SoftmaxNode>() => wrong node type!";

    const auto axis                 = static_cast<nn_ir::Axis>(softmax_node->axis());
    int32_t    exp_lut_blob_id      = softmax_node->exp_lut_blob_id();
    float      exp_scale            = softmax_node->exp_scale();
    float      exp_bias             = softmax_node->exp_bias();
    int32_t    softmax_lut_blob_id  = softmax_node->softmax_lut_blob_id();
    float      softmax_scale_ex     = softmax_node->softmax_scale_ex();
    int32_t    softmax_max_sum_ex   = softmax_node->softmax_max_sum_ex();
    int32_t    softmax_max_ex       = softmax_node->softmax_max_ex();
    float      softmax_scale_sum_ex = softmax_node->softmax_scale_sum_ex();
    bool       has_mask             = softmax_node->has_mask();

    return std::make_unique<nn_ir::SoftmaxNode>(node_info,
                                                axis,
                                                exp_lut_blob_id,
                                                exp_scale,
                                                exp_bias,
                                                softmax_lut_blob_id,
                                                softmax_scale_ex,
                                                softmax_max_sum_ex,
                                                softmax_max_ex,
                                                softmax_scale_sum_ex,
                                                has_mask);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_TileNode>(const IR::NnNode* ir_node, const nn_ir::NodeInfo& node_info) {
    auto tile_node = ir_node->nn_node_as_TileNode();
    Log::IR::E_IF(!tile_node) << "IRNNNodeParser::parseNNNode<NN::TileNode>() => wrong node type!";

    const auto axis  = static_cast<nn_ir::Axis>(tile_node->axis());
    const auto tiles = static_cast<uint32_t>(tile_node->tiles());

    return std::make_unique<nn_ir::TileNode>(node_info, axis, tiles);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_FullyConnectedNode>(const IR::NnNode*      ir_node,
                                                                    const nn_ir::NodeInfo& node_info) {
    auto fc_node = ir_node->nn_node_as_FullyConnectedNode();
    Log::IR::E_IF(!fc_node) << "IRNNNodeParser::parseNNNode<NN::FullyConnectedNode>() => wrong node type!";

    auto act_node   = getActNode(node_info, fc_node->activation());
    auto shift_node = getShiftNode(node_info, fc_node->shift());

    auto axis      = static_cast<nn_ir::Axis>(fc_node->axis());
    bool transpose = fc_node->transpose();

    BLOB_ID_T weight_blob_id = fc_node->kernel_blob_id();
    BLOB_ID_T bias_blob_id   = fc_node->bias_blob_id();

    return std::make_unique<nn_ir::FullyConnectedNode>(
        node_info, std::move(act_node), std::move(shift_node), axis, transpose, weight_blob_id, bias_blob_id);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_EltwiseNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto elt_node = ir_node->nn_node_as_EltwiseNode();
    Log::IR::E_IF(elt_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::EltwiseNode>() => wrong node type!";

    nn_ir::IR_Node_Config_Type_ ir_type     = elt_node->operation();
    nn_ir::EltwiseType          elt_type    = std::get<nn_ir::EltwiseType>(nn_ir::parseConfigType(ir_type));
    uint16_t                    multi_scale = elt_node->multi_scale();

    // TODO(dongguen.lim): add coeff
    auto stable_prod_grad = elt_node->stable_prod_grad();
    auto shift_node       = getShiftNode(node_info, elt_node->shift());
    auto shift_in1_node   = getShiftNode(node_info, elt_node->in1shift());
    auto shift_in2_node   = getShiftNode(node_info, elt_node->in2shift());
    return std::make_unique<nn_ir::EltwiseNode>(node_info,
                                                elt_type,
                                                stable_prod_grad,
                                                std::move(shift_node),
                                                std::move(shift_in1_node),
                                                std::move(shift_in2_node),
                                                multi_scale);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_BatchNormNode>(const IR::NnNode*      ir_node,
                                                               const nn_ir::NodeInfo& node_info) {
    auto bn_node = ir_node->nn_node_as_BatchNormNode();
    Log::IR::E_IF(bn_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::BatchNormNode>() => wrong node type!";

    auto ir_std_data_arr  = bn_node->std_arr();
    auto ir_mean_data_arr = bn_node->mean_arr();

    auto use_global_stats = bn_node->use_global_stats();
    auto eps              = bn_node->eps();
    auto scale            = bn_node->scale();
    auto axis             = static_cast<nn_ir::Axis>(bn_node->axis());

    auto std_data_arr  = makeDataArrFromTypedArray<float>(ir_std_data_arr, nn_ir::DataType::FLOAT32);
    auto mean_data_arr = makeDataArrFromTypedArray<float>(ir_mean_data_arr, nn_ir::DataType::FLOAT32);

    return std::make_unique<nn_ir::BatchNormNode>(
        node_info, axis, use_global_stats, eps, scale, std_data_arr, mean_data_arr);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_DataFormatNode>(const IR::NnNode*      ir_node,
                                                                const nn_ir::NodeInfo& node_info) {
    auto data_format_node = ir_node->nn_node_as_DataFormatNode();
    Log::IR::E_IF(data_format_node == nullptr) << "IRImporter::parseNNNode<NN::DataFormatNode>() => wrong node type!";

    auto                        ir_format_direction = data_format_node->direction();
    nn_ir::DataFormatConversion format_direction;
    switch (ir_format_direction) {
        case IR::NNNode::DataFormatConversion::DataFormatConversion_TENSOR2CELL:
            format_direction = nn_ir::DataFormatConversion::TENSOR2CELL;
            break;
        case IR::NNNode::DataFormatConversion::DataFormatConversion_CELL2TENSOR:
            format_direction = nn_ir::DataFormatConversion::CELL2TENSOR;
            break;
        default:
            Log::IR::E() << "IRImporter::parseNNNode<NN::DataFormatNode>() => wrong format direction!";
            break;
    }
    nn_ir::IR_Node_Param_ ir_shape = data_format_node->shape();
    auto                  shape    = std::get<nn_ir::Shape4D>(nn_ir::parseParam(ir_shape));

    return std::make_unique<nn_ir::DataFormatNode>(node_info, format_direction, shape);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_ReshapeNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto reshape_node = ir_node->nn_node_as_ReshapeNode();
    Log::IR::E_IF(reshape_node == nullptr) << "IRImporter::parseNNNode<NN::ReshapeNode>() => wrong node type!";

    nn_ir::IR_Node_Param_ ir_shape = reshape_node->shape();
    auto                  shape    = std::get<nn_ir::Shape4D>(nn_ir::parseParam(ir_shape));

    return std::make_unique<nn_ir::ReshapeNode>(node_info, shape);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_PermuteNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto permute_node = ir_node->nn_node_as_PermuteNode();
    Log::IR::E_IF(permute_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::PermuteNode>() => wrong node type!";

    nn_ir::IR_Node_Param_ permute_order = permute_node->order();
    auto                  order         = std::get<nn_ir::Shape4D>(nn_ir::parseParam(permute_order));
    auto                  in_shape      = permute_node->reinterpreted_shape();
    nn_ir::Shape4D        shape         = {{.n = 1, .c = 1, .h = 1, .w = 1}};
    if (in_shape) {
        shape = std::get<nn_ir::Shape4D>(nn_ir::parseParam(in_shape));
    }
    return std::make_unique<nn_ir::PermuteNode>(node_info, order, shape);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_PriorBoxNode>(const IR::NnNode*      ir_node,
                                                              const nn_ir::NodeInfo& node_info) {
    auto priorbox_node = ir_node->nn_node_as_PriorBoxNode();
    Log::IR::E_IF(priorbox_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::PriorBoxNode>() => wrong node type!";

    auto      ir_min_sizes     = priorbox_node->min_sizes();
    auto      ir_max_sizes     = priorbox_node->max_sizes();
    auto      ir_aspect_ratios = priorbox_node->aspect_ratios();
    bool      flip             = priorbox_node->flip();
    bool      clip             = priorbox_node->clip();
    auto      ir_variance      = priorbox_node->variance();
    float     step_h           = priorbox_node->step_h();
    float     step_w           = priorbox_node->step_w();
    float     offset           = priorbox_node->offset();
    BLOB_ID_T blob_id          = priorbox_node->blob_id();

    auto min_sizes     = makeDataArrFromTypedArray<float>(ir_min_sizes, nn_ir::DataType::FLOAT32);
    auto max_sizes     = makeDataArrFromTypedArray<float>(ir_max_sizes, nn_ir::DataType::FLOAT32);
    auto aspect_ratios = makeDataArrFromTypedArray<float>(ir_aspect_ratios, nn_ir::DataType::FLOAT32);
    auto variance      = makeDataArrFromTypedArray<float>(ir_variance, nn_ir::DataType::FLOAT32);

    nn_ir::IR_Node_Config_Type_ ir_type       = priorbox_node->type();
    nn_ir::PriorboxType         priorbox_type = std::get<nn_ir::PriorboxType>(nn_ir::parseConfigType(ir_type));

    return std::make_unique<nn_ir::PriorBoxNode>(node_info,
                                                 min_sizes,
                                                 max_sizes,
                                                 aspect_ratios,
                                                 flip,
                                                 clip,
                                                 variance,
                                                 step_h,
                                                 step_w,
                                                 offset,
                                                 priorbox_type,
                                                 blob_id);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_ScaleNode>(const IR::NnNode*      ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto scale_node = ir_node->nn_node_as_ScaleNode();
    Log::IR::E_IF(scale_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::ScaleNode>() => wrong node type!";

    auto ir_alpha_data_arr = scale_node->alpha_arr();
    auto ir_beta_data_arr  = scale_node->beta_arr();
    auto bias_term         = scale_node->bias_term();

    auto alpha_data_arr = makeDataArrFromTypedArray<float>(ir_alpha_data_arr, nn_ir::DataType::FLOAT32);
    auto beta_data_arr  = makeDataArrFromTypedArray<float>(ir_beta_data_arr, nn_ir::DataType::FLOAT32);

    return std::make_unique<nn_ir::ScaleNode>(node_info, bias_term, alpha_data_arr, beta_data_arr);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_SliceNode>(const IR::NnNode*      ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto slice_node = ir_node->nn_node_as_SliceNode();
    Log::IR::E_IF(!slice_node) << "IRNNNodeParser::parseNNNode<NN::SliceNode>() => wrong node type!";

    const auto axis   = static_cast<nn_ir::Axis>(slice_node->axis());
    const auto points = makeDataArrFromTypedArray<uint8_t>(slice_node->points(), nn_ir::DataType::UINT8);

    return std::make_unique<nn_ir::SliceNode>(node_info, axis, points);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_SpaceToDepthNode>(const IR::NnNode*      ir_node,
                                                                  const nn_ir::NodeInfo& node_info) {
    auto s2d_node = ir_node->nn_node_as_SpaceToDepthNode();
    Log::IR::E_IF(!s2d_node) << "IRNNNodeParser::parseNNNode<NN::SpaceToDepthNode>() => wrong node type!";

    int block_size = s2d_node->block_size();
    return std::make_unique<nn_ir::SpaceToDepthNode>(node_info, block_size);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_DepthToSpaceNode>(const IR::NnNode*      ir_node,
                                                                  const nn_ir::NodeInfo& node_info) {
    auto d2s_node = ir_node->nn_node_as_DepthToSpaceNode();
    Log::IR::E_IF(!d2s_node) << "IRNNNodeParser::parseNNNode<NN::DepthToSpaceNode>() => wrong node type!";

    int block_size = d2s_node->block_size();
    return std::make_unique<nn_ir::DepthToSpaceNode>(node_info, block_size);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_MatMulNode>(const IR::NnNode*      ir_node,
                                                            const nn_ir::NodeInfo& node_info) {
    auto matmul_node = ir_node->nn_node_as_MatMulNode();
    Log::IR::E_IF(!matmul_node) << "IRNNNodeParser::parseNNNode<NN::MatMulNode>() => wrong node type!";

    std::unique_ptr<nn_ir::ShiftNode> shift_node;
    shift_node = getShiftNode(node_info, matmul_node->shift());

    return std::make_unique<nn_ir::MatMulNode>(node_info, std::move(shift_node));
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_DummyNode>(const IR::NnNode*      ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto dummy_node = ir_node->nn_node_as_DummyNode();
    Log::IR::E_IF(dummy_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::DummyNode>() => wrong node type!";

    return std::make_unique<nn_ir::DummyNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_CopyNode>(const IR::NnNode* ir_node, const nn_ir::NodeInfo& node_info) {
    auto copy_node = ir_node->nn_node_as_CopyNode();
    Log::IR::E_IF(copy_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::CopyNode>() => wrong node type!";

    return std::make_unique<nn_ir::CopyNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenAppendNode>(const IR::NnNode* ir_node,
                                                                const nn_ir::NodeInfo& node_info) {
    auto aten_append_node = ir_node->nn_node_as_AtenAppendNode();
    Log::IR::E_IF(aten_append_node == nullptr)
     << "IRNNNodeParser::parseNNNode<NN::AtenAppendNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenAppendNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenAddNode>(const IR::NnNode* ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_add_node = ir_node->nn_node_as_AtenAddNode();
    Log::IR::E_IF(aten_add_node == nullptr)
     << "IRNNNodeParser::parseNNNode<NN::AtenAddNode>() => wrong node type!";
    return std::make_unique<nn_ir::AtenAddNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenAddmmNode>(const IR::NnNode* ir_node,
                                                               const nn_ir::NodeInfo& node_info) {
    auto aten_addmm_node = ir_node->nn_node_as_AtenAddmmNode();
    Log::IR::E_IF(aten_addmm_node == nullptr)
     << "IRNNNodeParser::parseNNNode<NN::AtenAddmmNode>() => wrong node type!";
    return std::make_unique<nn_ir::AtenAddmmNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenCatNode>(const IR::NnNode* ir_node,
                                                              const nn_ir::NodeInfo& node_info) {
    auto aten_cat_node = ir_node->nn_node_as_AtenCatNode();
    Log::IR::E_IF(aten_cat_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenCatNode>() => wrong node type!";

    int64_t dim = aten_cat_node->dim();
    return std::make_unique<nn_ir::AtenCatNode>(node_info, dim);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenCeilNode>(const IR::NnNode* ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_ceil_node = ir_node->nn_node_as_AtenCeilNode();
    Log::IR::E_IF(aten_ceil_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenCeilNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenCeilNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenCopyNode>(const IR::NnNode* ir_node,
                                                              const nn_ir::NodeInfo& node_info) {
    auto aten_copy_node = ir_node->nn_node_as_AtenCopyNode();
    Log::IR::E_IF(aten_copy_node == nullptr)
     << "IRNNNodeParser::parseNNNode<NN::AtenCopyNode>() => wrong node type!";

    bool non_blocking = aten_copy_node->non_blocking();

    return std::make_unique<nn_ir::AtenCopyNode>(node_info, non_blocking);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenDeriveIndexNode>(const IR::NnNode* ir_node, const nn_ir::NodeInfo& node_info) {
    auto aten_zero_like_node = ir_node->nn_node_as_AtenDeriveIndexNode();
    Log::IR::E_IF(aten_zero_like_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenDeriveIndexNode>() => wrong node type!";
    int64_t step = aten_zero_like_node->step();
    int64_t start = aten_zero_like_node->start();
    return std::make_unique<nn_ir::AtenDeriveIndexNode>(node_info, start, step);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenDimNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_dim_node = ir_node->nn_node_as_AtenDimNode();
    Log::IR::E_IF(aten_dim_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenDimNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenDimNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenDivNode>(const IR::NnNode* ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_div_node = ir_node->nn_node_as_AtenDivNode();
    Log::IR::E_IF(aten_div_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenDivNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenDivNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenDropoutNode>(const IR::NnNode*      ir_node,
                                                                 const nn_ir::NodeInfo& node_info) {
    auto aten_dropout_node = ir_node->nn_node_as_AtenDropoutNode();
    Log::IR::E_IF(aten_dropout_node == nullptr)
    << "IRNNNodeParser::parseNNNode<NN::AtenDropoutNode>() => wrong node type!";

    float proportion = aten_dropout_node->proportion();
    bool  train      = aten_dropout_node->train();
    return std::make_unique<nn_ir::AtenDropoutNode>(node_info, proportion, train);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenEmbeddingNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_embedding_node = ir_node->nn_node_as_AtenEmbeddingNode();
    Log::IR::E_IF(aten_embedding_node == nullptr)
    << "IRNNNodeParser::parseNNNode<NN::AtenEmbeddingNode>() => wrong node type!";

    int64_t padding_idx = aten_embedding_node->padding_idx();
    bool    scale_grad_by_freq = aten_embedding_node->scale_grad_by_freq();
    bool    sparse = aten_embedding_node->sparse();

    return std::make_unique<nn_ir::AtenEmbeddingNode>(node_info, padding_idx, scale_grad_by_freq, sparse);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenEqNode>(const IR::NnNode*      ir_node,
                                                            const nn_ir::NodeInfo& node_info) {
    auto aten_eq_node = ir_node->nn_node_as_AtenEqNode();
    Log::IR::E_IF(aten_eq_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenEqNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenEqNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenExpandNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_expand_node = ir_node->nn_node_as_AtenExpandNode();
    Log::IR::E_IF(aten_expand_node == nullptr)
    << "IRNNNodeParser::parseNNNode<NN::AtenExpandNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenExpandNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenFormatNode>(const IR::NnNode* ir_node, const nn_ir::NodeInfo& node_info) {
    auto aten_format_node = ir_node->nn_node_as_AtenFormatNode();
    Log::IR::E_IF(aten_format_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenFormatNode>() => wrong node type!";
    auto assembly_format = aten_format_node->assembly_format()->c_str();

    return std::make_unique<nn_ir::AtenFormatNode>(node_info, assembly_format);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenGetItemNode>(const IR::NnNode*      ir_node,
                                                                 const nn_ir::NodeInfo& node_info) {
    auto aten_get_item_node = ir_node->nn_node_as_AtenGetItemNode();
    Log::IR::E_IF(aten_get_item_node == nullptr)
        << "IRNNNodeParser::parseNNNode<NN::AtenGetItemNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenGetItemNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenGtNode>(const IR::NnNode*      ir_node,
                                                            const nn_ir::NodeInfo& node_info) {
    auto aten_gt_node = ir_node->nn_node_as_AtenGtNode();
    Log::IR::E_IF(aten_gt_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenGtNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenGtNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenIntNode>(const IR::NnNode* ir_node,
                                                            const nn_ir::NodeInfo& node_info) {
    auto aten_int_node = ir_node->nn_node_as_AtenIntNode();
    Log::IR::E_IF(aten_int_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenIntNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenIntNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenIsNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_is_node = ir_node->nn_node_as_AtenIsNode();
    Log::IR::E_IF(aten_is_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenIsNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenIsNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenItemNode>(const IR::NnNode* ir_node,
                                                              const nn_ir::NodeInfo& node_info) {
    auto aten_item_node = ir_node->nn_node_as_AtenItemNode();
    Log::IR::E_IF(aten_item_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenItemNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenItemNode>(node_info);
}


template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenLenNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_len_node = ir_node->nn_node_as_AtenLenNode();
    Log::IR::E_IF(aten_len_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenLenNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenLenNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenListNode>(const IR::NnNode* ir_node, const nn_ir::NodeInfo& node_info) {
    auto aten_list_node = ir_node->nn_node_as_AtenListNode();
    Log::IR::E_IF(aten_list_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenListNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenListNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenLSTMNode>(const IR::NnNode*      ir_node,
                                                              const nn_ir::NodeInfo& node_info) {
    auto aten_lstm_node = ir_node->nn_node_as_AtenLSTMNode();
    Log::IR::E_IF(aten_lstm_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenLSTMNode>() => wrong node type!";

    float   has_biases     = aten_lstm_node->has_biases();
    int64_t num_layer      = aten_lstm_node->num_layers();
    double  dropout        = aten_lstm_node->dropout();
    bool    train          = aten_lstm_node->train();
    bool    bidirectional  = aten_lstm_node->bidirectional();
    bool    batch_first    = aten_lstm_node->batch_first();

    auto weight_blob_id = makeDataArrFromVector<int64_t>(aten_lstm_node->weight_blob_ids());
    auto bias_blob_id   = makeDataArrFromVector<int64_t>(aten_lstm_node->bias_blob_ids());

    return std::make_unique<nn_ir::AtenLSTMNode>(node_info, has_biases, num_layer, dropout, train,
                                              bidirectional, batch_first, weight_blob_id, bias_blob_id);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenLtNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_lt_node = ir_node->nn_node_as_AtenLtNode();
    Log::IR::E_IF(aten_lt_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenLtNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenLtNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenMatmulNode>(const IR::NnNode* ir_node,
                                                                const nn_ir::NodeInfo& node_info) {
    auto aten_matmul_node = ir_node->nn_node_as_AtenMatmulNode();
    Log::IR::E_IF(aten_matmul_node == nullptr)
     << "IRNNNodeParser::parseNNNode<NN::AtenMatmulNode>() => wrong node type!";
    return std::make_unique<nn_ir::AtenMatmulNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenMaxNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_max_node = ir_node->nn_node_as_AtenMaxNode();
    Log::IR::E_IF(aten_max_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenMaxNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenMaxNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenNeNode>(const IR::NnNode*      ir_node,
                                                            const nn_ir::NodeInfo& node_info) {
    auto aten_ne_node = ir_node->nn_node_as_AtenNeNode();
    Log::IR::E_IF(aten_ne_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenNeNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenNeNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenNegNode>(const IR::NnNode* ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_neg_node = ir_node->nn_node_as_AtenNegNode();
    Log::IR::E_IF(aten_neg_node == nullptr)
     << "IRNNNodeParser::parseNNNode<NN::AtenNegNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenNegNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenReluNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_relu_node = ir_node->nn_node_as_AtenReluNode();
    Log::IR::E_IF(aten_relu_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenReluNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenReluNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenSelectNode>(const IR::NnNode* ir_node,
                                                              const nn_ir::NodeInfo& node_info) {
    auto aten_select_node = ir_node->nn_node_as_AtenSelectNode();
    Log::IR::E_IF(aten_select_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenSelectNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenSelectNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenSizeNode>(const IR::NnNode* ir_node,
                                                              const nn_ir::NodeInfo& node_info) {
    auto aten_size_node = ir_node->nn_node_as_AtenSizeNode();
    Log::IR::E_IF(aten_size_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenSizeNode>() => wrong node type!";

    int64_t dim = aten_size_node->dim();
    return std::make_unique<nn_ir::AtenSizeNode>(node_info, dim);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenSliceNode>(const IR::NnNode* ir_node, const nn_ir::NodeInfo& node_info) {
    auto aten_slice_like_node = ir_node->nn_node_as_AtenSliceNode();
    Log::IR::E_IF(aten_slice_like_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenSliceNode>() => wrong node type!";
    int64_t dim = aten_slice_like_node->dim();
    int64_t step = aten_slice_like_node->step();
    int64_t start = aten_slice_like_node->start();
    int64_t end = aten_slice_like_node->end();
    return std::make_unique<nn_ir::AtenSliceNode>(node_info, dim, start, end, step);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenSubNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_sub_node = ir_node->nn_node_as_AtenSubNode();
    Log::IR::E_IF(aten_sub_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenSubNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenSubNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenTensorNode>(const IR::NnNode*      ir_node,
                                                             const nn_ir::NodeInfo& node_info) {
    auto aten_tensor_node = ir_node->nn_node_as_AtenTensorNode();
    Log::IR::E_IF(aten_tensor_node == nullptr)
        << "IRNNNodeParser::parseNNNode<NN::AtenTensorNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenTensorNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenToNode>(const IR::NnNode* ir_node,
                                                            const nn_ir::NodeInfo& node_info) {
    auto aten_to_node = ir_node->nn_node_as_AtenToNode();
    Log::IR::E_IF(aten_to_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenToNode>() => wrong node type!";

    // Fixme(SRCX): Cast should be done in GraphGen ir builder?
    nn_ir::DataType dtype = convertIrTypeToNNIr(static_cast<IR::Type::DataType>(aten_to_node->dtype()));
    bool non_blocking = aten_to_node->non_blocking();
    bool copy = aten_to_node->copy();
    int64_t optional_memory_format = aten_to_node->optional_memory_format();
    return std::make_unique<nn_ir::AtenToNode>(node_info, dtype, non_blocking, copy, optional_memory_format);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenTransposeNode>(const IR::NnNode* ir_node,
                                                              const nn_ir::NodeInfo& node_info) {
    auto aten_transpose_node = ir_node->nn_node_as_AtenTransposeNode();
    Log::IR::E_IF(aten_transpose_node == nullptr) 
    << "IRNNNodeParser::parseNNNode<NN::AtenTransposeNode>() => wrong node type!";

    int64_t dim0 = aten_transpose_node->dim0();
    int64_t dim1 = aten_transpose_node->dim1();
    return std::make_unique<nn_ir::AtenTransposeNode>(node_info, dim0, dim1);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenUnsqueezeNode>(const IR::NnNode* ir_node,
                                                                   const nn_ir::NodeInfo& node_info) {
    auto aten_unsqueeze_node = ir_node->nn_node_as_AtenUnsqueezeNode();
    Log::IR::E_IF(aten_unsqueeze_node == nullptr)
     << "IRNNNodeParser::parseNNNode<NN::AtenUnsqueezeNode>() => wrong node type!";

    int64_t dim = aten_unsqueeze_node->dim();
    return std::make_unique<nn_ir::AtenUnsqueezeNode>(node_info, dim);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenZerosLikeNode>(const IR::NnNode* ir_node, const nn_ir::NodeInfo& node_info) {
    auto aten_zero_like_node = ir_node->nn_node_as_AtenZerosLikeNode();
    Log::IR::E_IF(aten_zero_like_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenZerosLikeNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenZerosLikeNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::NNNode>
IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_AtenZerosNode>(const IR::NnNode* ir_node,
                                                               const nn_ir::NodeInfo& node_info) {
    auto aten_zeros_node = ir_node->nn_node_as_AtenZerosNode();
    Log::IR::E_IF(aten_zeros_node == nullptr) << "IRNNNodeParser::parseNNNode<NN::AtenZerosNode>() => wrong node type!";

    return std::make_unique<nn_ir::AtenZerosNode>(node_info);
}

std::unique_ptr<nn_ir::ActivationNode> IRNNNodeParser::getActNode(const nn_ir::NodeInfo&            node_info,
                                                                  const IR::NNNode::ActivationNode* ir_act_node) {
    if (ir_act_node == nullptr) {
        return nullptr;
    }
    nn_ir::NodeInfo node_info_(-1, node_info.name + "_activation", node_info.graph);
    auto            ir_act_type = ir_act_node->type();

    auto slope          = ir_act_node->slope();
    auto negative_slope = ir_act_node->negative_slope();
    auto min            = ir_act_node->min();
    auto max            = ir_act_node->max();
    auto shift_node     = getShiftNode(node_info, ir_act_node->shift());

    nn_ir::ActivationType act_type = parseActivation(ir_act_type);
    return std::make_unique<nn_ir::ActivationNode>(
        node_info_, std::move(shift_node), act_type, slope, negative_slope, min, max);
}

std::unique_ptr<nn_ir::ShiftNode> IRNNNodeParser::getShiftNode(const nn_ir::NodeInfo&       node_info,
                                                               const IR::OPNode::ShiftNode* ir_shift_node) {
    if (ir_shift_node == nullptr) {
        return nullptr;
    }

    nn_ir::NodeInfo shift_node_info(-1, node_info.name + "_shift", node_info.graph);
    return parseShiftNode(shift_node_info, ir_shift_node);
}

} // namespace nn_compiler
