/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/ir_parser.hpp"
#include "ir/common/log.hpp"
#include "ir/ir_hwnode_parser.hpp"
#include "ir/ir_includes.hpp"
#include "ir/ir_nnnode_parser.hpp"
#include "ir/ir_tools.hpp"

namespace nn_compiler {

std::unique_ptr<nn_ir::Node> IRParser::parseNode(const IR::Node* node, const nn_ir::NNIR& graph) {
    auto ir_in_edges  = node->in_edges_id();
    auto ir_out_edges = node->out_edges_id();

    std::vector<EDGE_ID_T> in_edge_ids;
    if (ir_in_edges != nullptr) {
        auto raw_data  = ir_in_edges->data();
        auto data_size = ir_in_edges->size();
        in_edge_ids.assign(raw_data, raw_data + data_size);
    }

    std::vector<EDGE_ID_T> out_edge_ids;
    if (ir_out_edges != nullptr) {
        auto raw_data  = ir_out_edges->data();
        auto data_size = ir_out_edges->size();
        out_edge_ids.assign(raw_data, raw_data + data_size);
    }

    nn_ir::NodeInfo              node_info(node->id(), node->name()->str(), graph, in_edge_ids, out_edge_ids);
    std::unique_ptr<nn_ir::Node> g_node;

#define NODE(NNIR_TYPE, NODE_MAP, NODE_AS_GETTER, NODE_TYPE_GETTER)                                                    \
    case NNIR_TYPE: {                                                                                                  \
        auto typed_node = node->NODE_AS_GETTER();                                                                      \
        Log::IR::E_IF(!NODE_MAP.count(typed_node->NODE_TYPE_GETTER())) << "IRParser::parseNode => unknown node type!"; \
        const auto& parser = NODE_MAP[typed_node->NODE_TYPE_GETTER()];                                                 \
        g_node             = (this->*parser)(typed_node, node_info);                                                   \
        break;                                                                                                         \
    }

    switch (node->node_type()) {
        // NODE(IR::AnyNode_NnNode, nn_node_parse_func_map_, node_as_NnNode, nn_node_type)
        case IR::AnyNode_NnNode: {
            auto typed_node = node->node_as_NnNode();
            Log::IR::E_IF(!nn_node_parse_func_map_.count(typed_node->nn_node_type()))
                << "IRParser::parseNode => unknown node type!";
            const auto&    parser = nn_node_parse_func_map_[typed_node->nn_node_type()];
            IRNNNodeParser obj;
            g_node = (obj.*parser)(typed_node, node_info);
            break;
        }
        case IR::AnyNode_HwNode: {
            auto typed_node = node->node_as_HwNode();
            Log::IR::E_IF(!hw_node_parse_func_map_.count(typed_node->hw_node_type()))
                << "IRParser::parseNode => unknown node type!";
            const auto&    parser = hw_node_parse_func_map_[typed_node->hw_node_type()];
            IRHWNodeParser obj;
            g_node = (obj.*parser)(typed_node, node_info);
            break;
        }
            NODE(IR::AnyNode_OpNode, op_node_parse_func_map_, node_as_OpNode, op_node_type)
            NODE(IR::AnyNode_vNode, v_node_parse_func_map_, node_as_vNode, v_node_type)
            NODE(IR::AnyNode_qNode, q_node_parse_func_map_, node_as_qNode, q_node_type)
            NODE(IR::AnyNode_globalNode, global_node_parse_func_map_, node_as_globalNode, global_node_type)
        default: {
            Log::IR::E() << "IRParser::parseNode => unknown node type!";
        }
    }
#undef NODE

    return g_node;
}

template <>
std::unique_ptr<nn_ir::OPNode> IRParser::parseOPNode<IR::OPNode::AnyType_ShiftNode>(const IR::OpNode*      ir_node,
                                                                                    const nn_ir::NodeInfo& node_info) {
    auto shift_node = ir_node->op_node_as_ShiftNode();
    Log::IR::E_IF(shift_node == nullptr) << "IRParser::parseOPNode<OP::ShiftNode>() => wrong node type!";
    return parseShiftNode(node_info, shift_node);
}

template <>
std::unique_ptr<nn_ir::GlobalNode>
IRParser::parseGlobalNode<IR::GlobalNode::AnyType_GlobalSplitNode>(const IR::globalNode*  ir_node,
                                                                   const nn_ir::NodeInfo& node_info) {
    auto gsplit = ir_node->global_node_as_GlobalSplitNode();
    Log::IR::E_IF(gsplit == nullptr) << "IRParser::parseGlobalNode<Global::SplitNode() => wrong node type!";

    nn_ir::SyncType gsync_type = nn_ir::SyncType::NONE;
    nn_ir::SigType  sig_type   = nn_ir::SigType::NONE;
    if (gsplit->sync_node() != nullptr) {
        nn_ir::IR_Node_Config_Type_ ir_sync_type = gsplit->sync_node()->sync_type();
        gsync_type                               = std::get<nn_ir::SyncType>(nn_ir::parseConfigType(ir_sync_type));
        nn_ir::IR_Node_Config_Type_ ir_sig_type  = gsplit->sync_node()->sig_type();
        sig_type                                 = std::get<nn_ir::SigType>(nn_ir::parseConfigType(ir_sig_type));
    }

    auto                        gsync_uid         = gsplit->uid();
    nn_ir::IR_Node_Config_Type_ ir_partition_mode = gsplit->partition_mode();
    nn_ir::PartitionMode partition_mode = std::get<nn_ir::PartitionMode>(nn_ir::parseConfigType(ir_partition_mode));
    nn_ir::Shape4D       ifm_starts     = std::get<nn_ir::Shape4D>(nn_ir::parseParam(gsplit->ifm_split_pos()));

    return std::make_unique<nn_ir::GlobalSplitNode>(
        node_info, gsync_uid, gsync_type, sig_type, ifm_starts, partition_mode);
}

template <>
std::unique_ptr<nn_ir::GlobalNode>
IRParser::parseGlobalNode<IR::GlobalNode::AnyType_GlobalConcatNode>(const IR::globalNode*  ir_node,
                                                                    const nn_ir::NodeInfo& node_info) {
    auto gconcat = ir_node->global_node_as_GlobalConcatNode();
    Log::IR::E_IF(gconcat == nullptr) << "IRParser::parseGlobalNode<Global::ConcatNode() => wrong node type!";

    nn_ir::SyncType gsync_type = nn_ir::SyncType::NONE;
    nn_ir::SigType  sig_type   = nn_ir::SigType::NONE;
    if (gconcat->sync_node() != nullptr) {
        nn_ir::IR_Node_Config_Type_ ir_sync_type = gconcat->sync_node()->sync_type();
        gsync_type                               = std::get<nn_ir::SyncType>(nn_ir::parseConfigType(ir_sync_type));
        nn_ir::IR_Node_Config_Type_ ir_sig_type  = gconcat->sync_node()->sig_type();
        sig_type                                 = std::get<nn_ir::SigType>(nn_ir::parseConfigType(ir_sig_type));
    }
    auto                        gsync_uid      = gconcat->uid();
    nn_ir::IR_Node_Config_Type_ ir_concat_axis = gconcat->concat_axis();
    nn_ir::GlobalConcatAxis     concat_axis = std::get<nn_ir::GlobalConcatAxis>(nn_ir::parseConfigType(ir_concat_axis));
    nn_ir::IR_Node_Config_Type_ ir_concat_type = gconcat->type();
    nn_ir::GlobalConcatType     concat_type = std::get<nn_ir::GlobalConcatType>(nn_ir::parseConfigType(ir_concat_type));

    nn_ir::Shape4D ofm_starts = std::get<nn_ir::Shape4D>(nn_ir::parseParam(gconcat->ofm_split_pos()));
    return std::make_unique<nn_ir::GlobalConcatNode>(
        node_info, gsync_uid, gsync_type, sig_type, ofm_starts, concat_axis, concat_type);
}

template <>
std::unique_ptr<nn_ir::GlobalNode>
IRParser::parseGlobalNode<IR::GlobalNode::AnyType_GlobalSyncNode>(const IR::globalNode*  ir_node,
                                                                  const nn_ir::NodeInfo& node_info) {
    auto gsync = ir_node->global_node_as_GlobalSyncNode();
    Log::IR::E_IF(gsync == nullptr) << "IRParser::parseGlobalNode<Global::SyncNode() => wrong node type!";

    nn_ir::SyncType gsync_type = nn_ir::SyncType::NONE;
    nn_ir::SigType  sig_type   = nn_ir::SigType::NONE;

    nn_ir::IR_Node_Config_Type_ ir_sync_type = gsync->sync_type();
    gsync_type                               = std::get<nn_ir::SyncType>(nn_ir::parseConfigType(ir_sync_type));
    nn_ir::IR_Node_Config_Type_ ir_sig_type  = gsync->sig_type();
    sig_type                                 = std::get<nn_ir::SigType>(nn_ir::parseConfigType(ir_sig_type));
    auto gsync_uid                           = gsync->uid();
    return std::make_unique<nn_ir::GlobalSyncNode>(node_info, gsync_uid, gsync_type, sig_type);
}

template <>
std::unique_ptr<nn_ir::VNode> IRParser::parseVNode<IR::VNode::AnyType_VSplitNode>(const IR::vNode*       ir_node,
                                                                                  const nn_ir::NodeInfo& node_info) {
    auto vsplit_node = ir_node->v_node_as_VSplitNode();
    Log::IR::E_IF(vsplit_node == nullptr) << "IRParser::parseVNode<V::SplitNode() => wrong node type!";

    nn_ir::IR_Node_Param_ ir_num_tiles      = vsplit_node->num_tiles();
    auto                  ir_tile_positions = vsplit_node->tile_positions();

    nn_ir::TileNumbers num_tiles = std::get<nn_ir::TileNumbers>(nn_ir::parseParam(ir_num_tiles));

    std::vector<nn_ir::TileInfo> tile_infos;
    for (const auto& ir_tile_position : *ir_tile_positions) {
        tile_infos.emplace_back(std::get<nn_ir::TileInfo>(nn_ir::parseParam(ir_tile_position)));
    }

    Log::IR::I() << "IRParser::parseVNode<V:SplitNode>() : num tiles = " << num_tiles.getNumberOfTiles();

    for (nn_ir::TileInfo tile : tile_infos) {
        Log::IR::I() << "IRParser::parseVNode<V:SplitNode>() : tile id = " << tile.node_id;
        Log::IR::I() << "IRParser::parseVNode<V:SplitNode>() : tile index (n c h w) = " << tile.position.n << " "
                     << tile.position.c << " " << tile.position.h << " " << tile.position.w;
        Log::IR::I() << "IRParser::parseVNode<V:SplitNode>() : tile first coord (w h c n) = "
                     << tile.first_value_coord.w << " " << tile.first_value_coord.h << " " << tile.first_value_coord.c
                     << " " << tile.first_value_coord.n;
    }

    return std::make_unique<nn_ir::VSplitNode>(node_info, num_tiles, tile_infos);
}

template <>
std::unique_ptr<nn_ir::VNode> IRParser::parseVNode<IR::VNode::AnyType_VConcatNode>(const IR::vNode*       ir_node,
                                                                                   const nn_ir::NodeInfo& node_info) {
    auto vconcat_node = ir_node->v_node_as_VConcatNode();
    Log::IR::E_IF(vconcat_node == nullptr) << "IRParser::parseVNode<V::ConcatNode() => wrong node type!";

    auto vsplit_node_id = vconcat_node->vsplit_node_id();

    return std::make_unique<nn_ir::VConcatNode>(node_info, vsplit_node_id);
}

template <typename Qnode>
std::tuple<std::vector<float>, std::vector<int32_t>, std::vector<int8_t>> parseQParams(const Qnode*     q_node,
                                                                                       nn_ir::QuantType quant_type) {
    std::vector<float>   output_scales;
    std::vector<int32_t> output_zp;
    std::vector<int8_t>  frac_len;
    if (quant_type == nn_ir::QuantType::ASYMMETRIC) {
        output_scales =
            makeDataArrFromTypedArray<float>(q_node->param_as_AsymQuantParam()->scale(), nn_ir::DataType::FLOAT32);
        output_zp =
            makeDataArrFromTypedArray<int32_t>(q_node->param_as_AsymQuantParam()->zero_point(), nn_ir::DataType::INT32);
    } else {
        frac_len =
            makeDataArrFromTypedArray<int8_t>(q_node->param_as_SymmQuantParam()->frac_len(), nn_ir::DataType::INT8);
    }
    return {output_scales, output_zp, frac_len};
}

template <>
std::unique_ptr<nn_ir::QNode> IRParser::parseQNode<IR::QNode::AnyType_QuantNode>(const IR::qNode*       ir_node,
                                                                                 const nn_ir::NodeInfo& node_info) {
    auto quant_node = ir_node->q_node_as_QuantNode();
    Log::IR::E_IF(!quant_node) << "IRParser::parseQNode<Q::QuantNode>() => wrong node type!\n";
    nn_ir::QuantType quant_type               = static_cast<nn_ir::QuantType>(quant_node->type());
    auto [output_scales, output_zp, frac_len] = parseQParams<IR::QNode::QuantNode>(quant_node, quant_type);
    return std::make_unique<nn_ir::QuantNode>(node_info, quant_type, output_scales, output_zp, frac_len);
}

template <>
std::unique_ptr<nn_ir::QNode> IRParser::parseQNode<IR::QNode::AnyType_DequantNode>(const IR::qNode*       ir_node,
                                                                                   const nn_ir::NodeInfo& node_info) {
    auto dequant_node = ir_node->q_node_as_DequantNode();
    Log::IR::E_IF(!dequant_node) << "IRParser::parseQNode<Q::QuantNode>() => wrong node type!\n";
    nn_ir::QuantType quant_type               = static_cast<nn_ir::QuantType>(dequant_node->type());
    auto [output_scales, output_zp, frac_len] = parseQParams<IR::QNode::DequantNode>(dequant_node, quant_type);
    return std::make_unique<nn_ir::DequantNode>(node_info, quant_type, output_scales, output_zp, frac_len);
}

/**
 * @brief.      Constructor of IRParser.
 * @details.    This function constructs IRParser
 * @param[in].
 * @param[out].
 * @returns.
 */
IRParser::IRParser() {
    nn_node_parse_func_map_ = {
        {IR::NNNode::AnyType_InputNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_InputNode>},
        {IR::NNNode::AnyType_ConvNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_ConvNode>},
        {IR::NNNode::AnyType_PoolNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_PoolNode>},
        {IR::NNNode::AnyType_ActivationNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_ActivationNode>},
        {IR::NNNode::AnyType_ConcatNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_ConcatNode>},
        {IR::NNNode::AnyType_SoftmaxNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_SoftmaxNode>},
        {IR::NNNode::AnyType_TileNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_TileNode>},
        {IR::NNNode::AnyType_FullyConnectedNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_FullyConnectedNode>},
        {IR::NNNode::AnyType_EltwiseNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_EltwiseNode>},
        {IR::NNNode::AnyType_BatchNormNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_BatchNormNode>},
        {IR::NNNode::AnyType_ScaleNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_ScaleNode>},
        {IR::NNNode::AnyType_DeConvNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_DeConvNode>},
        {IR::NNNode::AnyType_ReshapeNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_ReshapeNode>},
        {IR::NNNode::AnyType_DataFormatNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_DataFormatNode>},
        {IR::NNNode::AnyType_PermuteNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_PermuteNode>},
        {IR::NNNode::AnyType_PriorBoxNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_PriorBoxNode>},
        {IR::NNNode::AnyType_SliceNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_SliceNode>},
        {IR::NNNode::AnyType_SpaceToDepthNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_SpaceToDepthNode>},
        {IR::NNNode::AnyType_DepthToSpaceNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_DepthToSpaceNode>},
        {IR::NNNode::AnyType_MatMulNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_MatMulNode>},
        {IR::NNNode::AnyType_DummyNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_DummyNode>},
        {IR::NNNode::AnyType_CopyNode, &IRNNNodeParser::parseNNNode<IR::NNNode::AnyType_CopyNode>},
    };

    op_node_parse_func_map_ = {
        {IR::OPNode::AnyType_ShiftNode, &IRParser::parseOPNode<IR::OPNode::AnyType_ShiftNode>},
    };

    q_node_parse_func_map_ = {
        {IR::QNode::AnyType_QuantNode, &IRParser::parseQNode<IR::QNode::AnyType_QuantNode>},
        {IR::QNode::AnyType_DequantNode, &IRParser::parseQNode<IR::QNode::AnyType_DequantNode>},
    };

    global_node_parse_func_map_ = {
        {IR::GlobalNode::AnyType_GlobalSplitNode, &IRParser::parseGlobalNode<IR::GlobalNode::AnyType_GlobalSplitNode>},
        {IR::GlobalNode::AnyType_GlobalConcatNode,
         &IRParser::parseGlobalNode<IR::GlobalNode::AnyType_GlobalConcatNode>},
        {IR::GlobalNode::AnyType_GlobalSyncNode, &IRParser::parseGlobalNode<IR::GlobalNode::AnyType_GlobalSyncNode>},
    };

    v_node_parse_func_map_ = {
        {IR::VNode::AnyType_VSplitNode, &IRParser::parseVNode<IR::VNode::AnyType_VSplitNode>},
        {IR::VNode::AnyType_VConcatNode, &IRParser::parseVNode<IR::VNode::AnyType_VConcatNode>},
    };

    hw_node_parse_func_map_ = {
        {IR::HWNode::AnyType_MAAEltwiseNode, &IRHWNodeParser::parseHWNode<IR::HWNode::AnyType_MAAEltwiseNode>},
    };
}
} // namespace nn_compiler