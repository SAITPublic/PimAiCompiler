import numpy as np

import IR.Blob
import IR.NNNode.ActivationNode
import IR.NNNode.AnyType
import IR.NNNode.BatchNormNode
import IR.NNNode.ConcatNode
import IR.NNNode.ConvNode
import IR.NNNode.CopyNode
import IR.NNNode.DeConvNode
import IR.NNNode.EltwiseNode
import IR.NNNode.FullyConnectedNode
import IR.NNNode.InputNode
import IR.NNNode.PoolNode
import IR.NNNode.ScaleNode
import IR.NNNode.PermuteNode
import IR.NNNode.ReshapeNode
import IR.NnNode
import IR.Node
import IR.Type.TypedValue
import IR.OpNode
import IR.OPNode.AnyType
import IR.OPNode.AddNode
import IR.NNNode.MatMulNode
import IR.OPNode.ShiftNode
import ir_enums
import IR.NNNode.DataFormatNode
import IR.NNNode.SliceNode
import IR.NNNode.TileNode

import IR.NNNode.AtenAddmmNode
import IR.NNNode.AtenAddNode
import IR.NNNode.AtenAndNode
import IR.NNNode.AtenAnyNode
import IR.NNNode.AtenAppendNode
import IR.NNNode.AtenArangeNode
import IR.NNNode.AtenAsTensorNode
import IR.NNNode.AtenBitwiseNotNode
import IR.NNNode.AtenBmmNode
import IR.NNNode.AtenBoolNode
import IR.NNNode.AtenCatNode
import IR.NNNode.AtenCeilNode
import IR.NNNode.AtenChunkNode
import IR.NNNode.AtenClampNode
import IR.NNNode.AtenClearNode
import IR.NNNode.AtenContiguousNode
import IR.NNNode.AtenConv2dNode
import IR.NNNode.AtenCopyNode
import IR.NNNode.AtenCpuNode
import IR.NNNode.AtenCudaNode
import IR.NNNode.AtenDeriveIndexNode
import IR.NNNode.AtenDimNode
import IR.NNNode.AtenDivNode
import IR.NNNode.AtenDropoutNode
import IR.NNNode.AtenEmbeddingNode
import IR.NNNode.AtenEqNode
import IR.NNNode.AtenEqualNode
import IR.NNNode.AtenExpandNode
import IR.NNNode.AtenFillNode
import IR.NNNode.AtenFloorDivideNode
import IR.NNNode.AtenFormatNode
import IR.NNNode.AtenGatherNode
import IR.NNNode.AtenGeNode
import IR.NNNode.AtenGetItemNode
import IR.NNNode.AtenGtNode
import IR.NNNode.AtenIndexNode
import IR.NNNode.AtenIndexPutNode
import IR.NNNode.AtenIndexSelectNode
import IR.NNNode.AtenIntNode
import IR.NNNode.AtenIsNode
import IR.NNNode.AtenItemNode
import IR.NNNode.AtenLeakyReluNode
import IR.NNNode.AtenLenNode
import IR.NNNode.AtenLinearNode
import IR.NNNode.AtenListNode
import IR.NNNode.AtenLogNode
import IR.NNNode.AtenLogSoftmaxNode
import IR.NNNode.AtenLSTMNode
import IR.NNNode.AtenLtNode
import IR.NNNode.AtenMaskedFillNode
import IR.NNNode.AtenMaskedSelectNode
import IR.NNNode.AtenMatmulNode
import IR.NNNode.AtenMaxNode
import IR.NNNode.AtenMaxPool2dNode
import IR.NNNode.AtenMinNode
import IR.NNNode.AtenMulNode
import IR.NNNode.AtenNegNode
import IR.NNNode.AtenNeNode
import IR.NNNode.AtenNotNode
import IR.NNNode.AtenOnesNode
import IR.NNNode.AtenPackPaddedSequenceNode
import IR.NNNode.AtenPadPackedSequenceNode
import IR.NNNode.AtenPowNode
import IR.NNNode.AtenReluNode
import IR.NNNode.AtenSelectNode
import IR.NNNode.AtenSetItemNode
import IR.NNNode.AtenSizeNode
import IR.NNNode.AtenSliceNode
import IR.NNNode.AtenSoftmaxNode
import IR.NNNode.AtenSqueezeNode
import IR.NNNode.AtenSubNode
import IR.NNNode.AtenSumNode
import IR.NNNode.AtenTanhNode
import IR.NNNode.AtenTensorNode
import IR.NNNode.AtenToNode
import IR.NNNode.AtenTopkNode
import IR.NNNode.AtenTransposeNode
import IR.NNNode.AtenUnsqueezeNode
import IR.NNNode.AtenViewNode
import IR.NNNode.AtenWarnNode
import IR.NNNode.AtenZerosLikeNode
import IR.NNNode.AtenZerosNode

import IR.ControlNode
import IR.CONTROLNode.AnyType
import IR.CONTROLNode.PrimBlockNode
import IR.CONTROLNode.PrimCallMethodNode
import IR.CONTROLNode.PrimConstantNode
import IR.CONTROLNode.PrimDataNode
import IR.CONTROLNode.PrimDeviceNode
import IR.CONTROLNode.PrimDtypeNode
import IR.CONTROLNode.PrimEndIfNode
import IR.CONTROLNode.PrimEndLoopNode
import IR.CONTROLNode.PrimGetAttrNode
import IR.CONTROLNode.PrimIfNode
import IR.CONTROLNode.PrimListConstructNode
import IR.CONTROLNode.PrimListUnpackNode
import IR.CONTROLNode.PrimLoopIndexNode
import IR.CONTROLNode.PrimLoopNode
import IR.CONTROLNode.PrimRaiseExceptionNode
import IR.CONTROLNode.PrimSetAttrNode
import IR.CONTROLNode.PrimTupleConstructNode
import IR.CONTROLNode.PrimTupleIndexNode
import IR.CONTROLNode.PrimTupleUnpackNode
import IR.CONTROLNode.PrimTypeNode
import IR.CONTROLNode.PrimUncheckedCastNode
import IR.CONTROLNode.PrimUninitializedNode
import IR.CONTROLNode.PrimVariableNode

import torch_ops


def node_name(graph_id: int, node_id: int) -> str:
    return str(graph_id) + '_node' + str(node_id)


def blob_node_name(graph_id: int, blob_id: int) -> str:
    return str(graph_id) + '_blob' + str(blob_id)


def edge_name(graph_id: int, edge_id: int) -> str:
    return str(graph_id) + '_edge' + str(edge_id)


def head_node_name(graph_id: int) -> str:
    return str(graph_id) + '_head'


def tail_node_name(graph_id: int) -> str:
    return str(graph_id) + '_tail'


# Labels for NNNode -> AnyType
def nn_node_label(nn_node: IR.NnNode.NnNode) -> (object, str):
    node_type = nn_node.NnNodeType()
    if node_type == IR.NNNode.AnyType.AnyType().InputNode:
        nn_node_input_node = IR.NNNode.InputNode.InputNode()
        nn_node_input_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_input_node, input_node_label(nn_node_input_node)

    elif node_type == IR.NNNode.AnyType.AnyType().ConvNode:
        nn_node_conv_node = IR.NNNode.ConvNode.ConvNode()
        nn_node_conv_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_conv_node, conv_node_label(nn_node_conv_node)

    elif node_type == IR.NNNode.AnyType.AnyType().DeConvNode:
        nn_node_deconv_node = IR.NNNode.DeConvNode.DeConvNode()
        nn_node_deconv_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_deconv_node, deconv_node_label(nn_node_deconv_node)

    elif node_type == IR.NNNode.AnyType.AnyType().PoolNode:
        nn_node_pool_node = IR.NNNode.PoolNode.PoolNode()
        nn_node_pool_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_pool_node, pool_node_label(nn_node_pool_node)

    elif node_type == IR.NNNode.AnyType.AnyType().ActivationNode:
        nn_node_activation_node = IR.NNNode.ActivationNode.ActivationNode()
        nn_node_activation_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_activation_node, activation_node_label(nn_node_activation_node)

    elif node_type == IR.NNNode.AnyType.AnyType().FullyConnectedNode:
        nn_node_fc_node = IR.NNNode.FullyConnectedNode.FullyConnectedNode()
        nn_node_fc_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_fc_node, fully_connected_node_label(nn_node_fc_node)

    elif node_type == IR.NNNode.AnyType.AnyType().ConcatNode:
        nn_node_cc_node = IR.NNNode.ConcatNode.ConcatNode()
        nn_node_cc_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_cc_node, concat_node_label(nn_node_cc_node)

    elif node_type == IR.NNNode.AnyType.AnyType().EltwiseNode:
        nn_node_ew_node = IR.NNNode.EltwiseNode.EltwiseNode()
        nn_node_ew_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_ew_node, eltwise_node_label(nn_node_ew_node)

    elif node_type == IR.NNNode.AnyType.AnyType().BatchNormNode:
        nn_node_bn_node = IR.NNNode.BatchNormNode.BatchNormNode()
        nn_node_bn_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_bn_node, batch_norm_node_label(nn_node_bn_node)

    elif node_type == IR.NNNode.AnyType.AnyType().ScaleNode:
        nn_node_scale_node = IR.NNNode.ScaleNode.ScaleNode()
        nn_node_scale_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_scale_node, scale_node_label(nn_node_scale_node)

    elif node_type == IR.NNNode.AnyType.AnyType().PermuteNode:
        nn_node_permute_node = IR.NNNode.PermuteNode.PermuteNode()
        nn_node_permute_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_permute_node, permute_node_label(nn_node_permute_node)

    elif node_type == IR.NNNode.AnyType.AnyType().ReshapeNode:
        nn_node_reshape_node = IR.NNNode.ReshapeNode.ReshapeNode()
        nn_node_reshape_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_reshape_node, reshape_node_label(nn_node_reshape_node)

    elif node_type == IR.NNNode.AnyType.AnyType().SliceNode:
        slice_node = IR.NNNode.SliceNode.SliceNode()
        slice_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return slice_node, slice_node_label(slice_node)

    elif node_type == IR.NNNode.AnyType.AnyType().TileNode:
        tile_node = IR.NNNode.TileNode.TileNode()
        tile_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return tile_node, tile_node_label(tile_node)

    elif node_type == IR.NNNode.AnyType.AnyType().MatMulNode:
        mm_node = IR.NNNode.MatMulNode.MatMulNode()
        mm_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return mm_node, list_table(['MatMul Node'])

    elif node_type == IR.NNNode.AnyType.AnyType().CopyNode:
        copy_node = IR.NNNode.CopyNode.CopyNode()
        copy_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return copy_node, list_table(['Copy Node'])
    
    # torch AtenNode
    # these AtenNode with attribute
    elif node_type == IR.NNNode.AnyType.AnyType().AtenCatNode:
        aten_cat_node = IR.NNNode.AtenCatNode.AtenCatNode()
        aten_cat_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_cat_node, aten_cat_node_label(aten_cat_node)

    elif node_type == IR.NNNode.AnyType.AnyType().AtenDropoutNode:
        aten_dropout_node = IR.NNNode.AtenDropoutNode.AtenDropoutNode()
        aten_dropout_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_dropout_node, aten_dropout_node_label(aten_dropout_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenExpandNode:
        aten_expand_node = IR.NNNode.AtenExpandNode.AtenExpandNode()
        aten_expand_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_expand_node, aten_expand_node_label(aten_expand_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenLSTMNode:
        aten_lstm_node = IR.NNNode.AtenLSTMNode.AtenLSTMNode()
        aten_lstm_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_lstm_node, aten_lstm_node_label(aten_lstm_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenSelectNode:
        aten_select_node = IR.NNNode.AtenSelectNode.AtenSelectNode()
        aten_select_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_select_node, aten_select_node_label(aten_select_node)

    elif node_type == IR.NNNode.AnyType.AnyType().AtenSizeNode:
        aten_size_node = IR.NNNode.AtenSizeNode.AtenSizeNode()
        aten_size_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_size_node, aten_size_node_label(aten_size_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenSliceNode:
        aten_slice_node = IR.NNNode.AtenSliceNode.AtenSliceNode()
        aten_slice_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_slice_node, aten_slice_node_label(aten_slice_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenToNode:
        aten_to_node = IR.NNNode.AtenToNode.AtenToNode()
        aten_to_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_to_node, aten_to_node_label(aten_to_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenTransposeNode:
        aten_transpose_node = IR.NNNode.AtenTransposeNode.AtenTransposeNode()
        aten_transpose_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_transpose_node, aten_transpose_node_label(aten_transpose_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenUnsqueezeNode:
        aten_unsqueeze_node = IR.NNNode.AtenUnsqueezeNode.AtenUnsqueezeNode()
        aten_unsqueeze_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_unsqueeze_node, aten_unsqueeze_node_label(aten_unsqueeze_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenCopyNode:
        aten_copy_node = IR.NNNode.AtenCopyNode.AtenCopyNode()
        aten_copy_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_copy_node, aten_copy_node_label(aten_copy_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenDeriveIndexNode:
        aten_derive_index_node = IR.NNNode.AtenDeriveIndexNode.AtenDeriveIndexNode()
        aten_derive_index_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_derive_index_node, aten_derive_index_node_label(aten_derive_index_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenEmbeddingNode:
        aten_embedding_node = IR.NNNode.AtenEmbeddingNode.AtenEmbeddingNode()
        aten_embedding_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_embedding_node, aten_embedding_node_label(aten_embedding_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenFormatNode:
        aten_format_node = IR.NNNode.AtenFormatNode.AtenFormatNode()
        aten_format_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_format_node, aten_format_node_label(aten_format_node)

    elif node_type == IR.NNNode.AnyType.AnyType().AtenPackPaddedSequenceNode:
        aten_pack_padded_sequence_node = IR.NNNode.AtenPackPaddedSequenceNode.AtenPackPaddedSequenceNode()
        aten_pack_padded_sequence_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_pack_padded_sequence_node, aten_pack_padded_sequence_node_label(aten_pack_padded_sequence_node)

    elif node_type == IR.NNNode.AnyType.AnyType().AtenPadPackedSequenceNode:
        aten_pad_packed_sequence_node = IR.NNNode.AtenPadPackedSequenceNode.AtenPadPackedSequenceNode()
        aten_pad_packed_sequence_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_pad_packed_sequence_node, aten_pad_packed_sequence_node_label(aten_pad_packed_sequence_node)

    elif node_type == IR.NNNode.AnyType.AnyType().AtenSetItemNode:
        aten_set_item_node = IR.NNNode.AtenSetItemNode.AtenSetItemNode()
        aten_set_item_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_set_item_node, aten_set_item_node_label(aten_set_item_node)

    elif node_type == IR.NNNode.AnyType.AnyType().AtenArangeNode:
        aten_arange_node = IR.NNNode.AtenArangeNode.AtenArangeNode()
        aten_arange_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_arange_node, aten_arange_node_label(aten_arange_node)

    elif node_type == IR.NNNode.AnyType.AnyType().AtenAsTensorNode:
        aten_as_tensor_node = IR.NNNode.AtenAsTensorNode.AtenAsTensorNode()
        aten_as_tensor_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_as_tensor_node, aten_as_tensor_node_label(aten_as_tensor_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenChunkNode:
        aten_chunk_node = IR.NNNode.AtenChunkNode.AtenChunkNode()
        aten_chunk_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_chunk_node, aten_chunk_node_label(aten_chunk_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenClampNode:
        aten_clamp_node = IR.NNNode.AtenClampNode.AtenClampNode()
        aten_clamp_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_clamp_node, aten_clamp_node_label(aten_clamp_node)

    elif node_type == IR.NNNode.AnyType.AnyType().AtenContiguousNode:
        aten_contiguous_node = IR.NNNode.AtenContiguousNode.AtenContiguousNode()
        aten_contiguous_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_contiguous_node, aten_contiguous_node_label(aten_contiguous_node)

    elif node_type == IR.NNNode.AnyType.AnyType().AtenGatherNode:
        aten_gather_node = IR.NNNode.AtenGatherNode.AtenGatherNode()
        aten_gather_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_gather_node, aten_gather_node_label(aten_gather_node)

    elif node_type == IR.NNNode.AnyType.AnyType().AtenIndexPutNode:
        aten_index_put_node = IR.NNNode.AtenIndexPutNode.AtenIndexPutNode()
        aten_index_put_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_index_put_node, aten_index_put_node_label(aten_index_put_node)

    elif node_type == IR.NNNode.AnyType.AnyType().AtenLeakyReluNode:
        aten_leaky_relu_node = IR.NNNode.AtenLeakyReluNode.AtenLeakyReluNode()
        aten_leaky_relu_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_leaky_relu_node, aten_leaky_relu_node_label(aten_leaky_relu_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenLogSoftmaxNode:
        aten_log_softmax_node = IR.NNNode.AtenLogSoftmaxNode.AtenLogSoftmaxNode()
        aten_log_softmax_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_log_softmax_node, aten_log_softmax_node_label(aten_log_softmax_node)
     
    elif node_type == IR.NNNode.AnyType.AnyType().AtenMaskedFillNode:
        aten_masked_fill_node = IR.NNNode.AtenMaskedFillNode.AtenMaskedFillNode()
        aten_masked_fill_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_masked_fill_node, aten_masked_fill_node_label(aten_masked_fill_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenMinNode:
        aten_min_node = IR.NNNode.AtenMinNode.AtenMinNode()
        aten_min_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_min_node, aten_min_node_label(aten_min_node)

    elif node_type == IR.NNNode.AnyType.AnyType().AtenMulNode:
        aten_mul_node = IR.NNNode.AtenMulNode.AtenMulNode()
        aten_mul_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_mul_node, aten_mul_node_label(aten_mul_node)
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenOnesNode:
        aten_ones_node = IR.NNNode.AtenOnesNode.AtenOnesNode()
        aten_ones_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_ones_node, aten_ones_node_label(aten_ones_node)   
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenSoftmaxNode:
        aten_softmax_node = IR.NNNode.AtenSoftmaxNode.AtenSoftmaxNode()
        aten_softmax_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_softmax_node, aten_softmax_node_label(aten_softmax_node) 

    elif node_type == IR.NNNode.AnyType.AnyType().AtenSqueezeNode:
        aten_squeeze_node = IR.NNNode.AtenSqueezeNode.AtenSqueezeNode()
        aten_squeeze_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_squeeze_node, aten_squeeze_node_label(aten_squeeze_node) 

    elif node_type == IR.NNNode.AnyType.AnyType().AtenSumNode:
        aten_sum_node = IR.NNNode.AtenSumNode.AtenSumNode()
        aten_sum_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_sum_node, aten_sum_node_label(aten_sum_node) 
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenTopkNode:
        aten_topk_node = IR.NNNode.AtenTopkNode.AtenTopkNode()
        aten_topk_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_topk_node, aten_topk_node_label(aten_topk_node) 
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenWarnNode:
        aten_warn_node = IR.NNNode.AtenWarnNode.AtenWarnNode()
        aten_warn_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_warn_node, aten_warn_node_label(aten_warn_node) 
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenConv2dNode:
        aten_conv2d_node = IR.NNNode.AtenConv2dNode.AtenConv2dNode()
        aten_conv2d_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_conv2d_node, aten_conv2d_node_label(aten_conv2d_node) 
    
    elif node_type == IR.NNNode.AnyType.AnyType().AtenMaxPool2dNode:
        aten_maxpool2d_node = IR.NNNode.AtenMaxPool2dNode.AtenMaxPool2dNode()
        aten_maxpool2d_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_maxpool2d_node, aten_maxpool2d_node_label(aten_maxpool2d_node) 

    elif node_type == IR.NNNode.AnyType.AnyType().AtenIndexSelectNode:
        aten_index_select_node = IR.NNNode.AtenIndexSelectNode.AtenIndexSelectNode()
        aten_index_select_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_index_select_node, aten_index_select_node_label(aten_index_select_node) 

    # aten Ops without attribute, lookup table
    op_name = torch_ops.aten_ops_dict[node_type]
    if op_name in torch_ops.aten_ops_no_attr_dict:      # has key
        aten_op_ = torch_ops.aten_ops_no_attr_dict[op_name]
        aten_op_.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return aten_op_, list_table(['{} Node'.format(op_name[:-4])])

    else:
        return None, list_table(['Unknown NNNode Type'])


def control_node_label(ctl_node: IR.ControlNode.ControlNode) -> (object, str):
    node_type = ctl_node.ControlNodeType()

    # torch prim ops with attrs
    if node_type == IR.CONTROLNode.AnyType.AnyType().PrimCallMethodNode:
        prim_call_method_node = IR.CONTROLNode.PrimCallMethodNode.PrimCallMethodNode()
        prim_call_method_node.Init(ctl_node.ControlNode().Bytes, ctl_node.ControlNode().Pos)
        return prim_call_method_node, prim_call_method_node_label(prim_call_method_node)

    elif node_type == IR.CONTROLNode.AnyType.AnyType().PrimConstantNode:
        prim_constant_node = IR.CONTROLNode.PrimConstantNode.PrimConstantNode()
        prim_constant_node.Init(ctl_node.ControlNode().Bytes, ctl_node.ControlNode().Pos)
        return prim_constant_node, prim_constant_node_label(prim_constant_node)
    
    elif node_type == IR.CONTROLNode.AnyType.AnyType().PrimIfNode:
        prim_if_node = IR.CONTROLNode.PrimIfNode.PrimIfNode()
        prim_if_node.Init(ctl_node.ControlNode().Bytes, ctl_node.ControlNode().Pos)
        return prim_if_node, prim_if_node_label(prim_if_node)
    
    elif node_type == IR.CONTROLNode.AnyType.AnyType().PrimLoopIndexNode:
        prim_loop_index_node = IR.CONTROLNode.PrimLoopIndexNode.PrimLoopIndexNode()
        prim_loop_index_node.Init(ctl_node.ControlNode().Bytes, ctl_node.ControlNode().Pos)
        return prim_loop_index_node, prim_loop_index_node_label(prim_loop_index_node)
    
    elif node_type == IR.CONTROLNode.AnyType.AnyType().PrimLoopNode:
        prim_loop_node = IR.CONTROLNode.PrimLoopNode.PrimLoopNode()
        prim_loop_node.Init(ctl_node.ControlNode().Bytes, ctl_node.ControlNode().Pos)
        return prim_loop_node, prim_loop_node_label(prim_loop_node)
    
    elif node_type == IR.CONTROLNode.AnyType.AnyType().PrimTupleIndexNode:
        prim_tuple_index_node = IR.CONTROLNode.PrimTupleIndexNode.PrimTupleIndexNode()
        prim_tuple_index_node.Init(ctl_node.ControlNode().Bytes, ctl_node.ControlNode().Pos)
        return prim_tuple_index_node, prim_tuple_index_node_label(prim_tuple_index_node)
    
    elif node_type == IR.CONTROLNode.AnyType.AnyType().PrimVariableNode :
        prim_variable_node = IR.CONTROLNode.PrimVariableNode .PrimVariableNode ()
        prim_variable_node.Init(ctl_node.ControlNode().Bytes, ctl_node.ControlNode().Pos)
        return prim_variable_node, prim_variable_node_label(prim_variable_node)

    # torch prim OPs without attrs
    op_name = torch_ops.prim_ops_dict[node_type]
    if op_name in torch_ops.prim_ops_no_attr_dict:
        prim_op_ = torch_ops.prim_ops_no_attr_dict[op_name]
        prim_op_.Init(ctl_node.ControlNode().Bytes, ctl_node.ControlNode().Pos)
        return prim_op_, list_table(['{} Node'.format(op_name[:-4])])


def op_node_label(op_node: IR.OpNode.OpNode) -> (object, str):
    node_type = op_node.OpNodeType()
    if node_type == IR.OPNode.AnyType.AnyType().AddNode:
        op_node_add_node = IR.OPNode.AddNode.AddNode()
        op_node_add_node.Init(op_node.OpNode().Bytes, op_node.OpNode().Pos)
        return op_node_add_node, list_table(['Add Node'])
    elif node_type == IR.OPNode.AnyType.AnyType().ShiftNode:
        op_node_shift_node = IR.OPNode.ShiftNode.ShiftNode()
        op_node_shift_node.Init(op_node.OpNode().Bytes, op_node.OpNode().Pos)
        return op_node_shift_node, shift_node_label(op_node_shift_node)
    else:
        return None, list_table(['Unknown OpNode Type'])


def input_node_label(input_node: IR.NNNode.InputNode.InputNode) -> str:
    retval = 'Input Node ({})<br/>'.format(ir_enums.InputType.get_name(input_node.Type()))
    if type(input_node.MeanAsNumpy()) is np.ndarray:
        retval += 'mean: {}<br/>'.format(input_node.MeanAsNumpy())
    retval += 'scale:{}<br/>'.format(input_node.Scale())
    retval += 'mirror: {}'.format(input_node.Mirror())

    return list_table([retval])


def conv_node_label(conv_node: IR.NNNode.ConvNode.ConvNode) -> str:
    kernel_size = conv_node.KernelSize()
    stride_size = conv_node.StrideSize()
    dilation_size = conv_node.DilationSize()
    padding_size = conv_node.PaddingSize()
    activation_node = conv_node.Activation()
    shift_node = conv_node.Shift()

    retval = 'Conv Node<br/>'
    if kernel_size is not None:
        retval += 'Kernel Size: {}<br/>'.format(dim2_label(kernel_size))
    if stride_size is not None:
        retval += 'Stride Size: {}<br/>'.format(dim2_label(stride_size))
    if dilation_size is not None:
        retval += 'Dilation Size: {}<br/>'.format(dim2_label(dilation_size))
    if padding_size is not None:
        retval += 'Padding Size: {}<br/>'.format(pad4_label(padding_size))

    activation_label = None
    if activation_node is not None:
        activation_label = activation_node_label(activation_node)
    shift_label = None
    if shift_node is not None:
        shift_label = shift_node_label(shift_node)

    return list_table([retval, activation_label, shift_label])


def deconv_node_label(deconv_node: IR.NNNode.DeConvNode.DeConvNode) -> str:
    kernel_size = deconv_node.KernelSize()
    stride_size = deconv_node.StrideSize()
    dilation_size = deconv_node.DilationSize()
    padding_size = deconv_node.PaddingSize()
    sampling_size = deconv_node.SamplingSize()
    output_padding = deconv_node.OutputPadding()
    activation_node = deconv_node.Activation()
    shift_node = deconv_node.Shift()

    retval = 'DeConv Node<br/>'
    if kernel_size is not None:
        retval += 'Kernel Size: {}<br/>'.format(dim2_label(kernel_size))
    if stride_size is not None:
        retval += 'Stride Size: {}<br/>'.format(dim2_label(stride_size))
    if dilation_size is not None:
        retval += 'Dilation Size: {}<br/>'.format(dim2_label(dilation_size))
    if padding_size is not None:
        retval += 'Padding Size: {}<br/>'.format(pad4_label(padding_size))

    if sampling_size is not None:
        retval += 'Sampling Size: {}<br/>'.format(dim2_label(sampling_size))
    retval += 'Interpolation: {}<br/>'.format(deconv_node.Interpolation())
    if output_padding is not None:
        retval += 'Output Padding: {}<br/>'.format(dim2_label(output_padding))

    activation_label = None
    if activation_node is not None:
        activation_label = activation_node_label(activation_node)

    shift_label = None
    if shift_node is not None:
        shift_label = shift_node_label(shift_node)

    return list_table([retval, activation_label, shift_label])


def pool_node_label(pool_node: IR.NNNode.PoolNode.PoolNode) -> str:
    kernel_size = pool_node.KernelSize()
    stride_size = pool_node.StrideSize()
    dilation_size = pool_node.DilationSize()
    padding_size = pool_node.PaddingSize()

    retval = 'Pool Node ({})<br/>'.format(ir_enums.PoolType.get_name(pool_node.Type()))
    if kernel_size is not None:
        retval += 'KernelSize: {}<br/>'.format(dim2_label(kernel_size))
    if stride_size is not None:
        retval += 'StrideSize: {}<br/>'.format(dim2_label(stride_size))
    if dilation_size is not None:
        retval += 'DilationSize: {}<br/>'.format(dim2_label(dilation_size))
    if padding_size is not None:
        retval += 'PaddingSize: {}<br/>'.format(pad4_label(padding_size))
    retval += 'PadCalculation: {}<br/>'.format(ir_enums.PadCalculation.get_name(pool_node.PadCalc()))
    return list_table([retval])


def activation_node_label(activation_node: IR.NNNode.ActivationNode.ActivationNode) -> str:
    retval = 'Activation Node ({})<br/>'.format(ir_enums.ActivationType.get_name(activation_node.Type()))
    retval += 'Slope: {:.2f}<br/>'.format(activation_node.Slope())
    retval += 'NegativeSlope: {:.2f}<br/>'.format(activation_node.NegativeSlope())
    retval += 'Min: {:.2f}, Max: {:.2f}<br/>'.format(activation_node.Min(), activation_node.Max())
    return list_table([retval])


def fully_connected_node_label(fc_node: IR.NNNode.FullyConnectedNode.FullyConnectedNode) -> str:
    activation_node = fc_node.Activation()
    shift_node = fc_node.Shift()

    retval = 'Fully Connected Node<br/>'
    retval += 'Axis: {}<br/>'.format(fc_node.Axis())
    retval += 'Transpose: {}<br/>'.format(fc_node.Transpose())

    activation_label = None
    if activation_node is not None:
        activation_label = activation_node_label(activation_node)

    shift_label = None
    if shift_node is not None:
        shift_label = shift_node_label(shift_node)

    return list_table([retval, activation_label, shift_label])


def concat_node_label(concat_node: IR.NNNode.ConcatNode.ConcatNode) -> str:
    retval = 'Concat Node<br/>'
    retval += 'Axis: {}'.format(concat_node.Axis())
    return list_table([retval])


def eltwise_node_label(eltwise_node: IR.NNNode.EltwiseNode.EltwiseNode) -> str:
    retval = 'Eltwise Node<br/>'
    retval += 'Operation: {}<br/>'.format(ir_enums.EltwiseType.get_name(eltwise_node.Operation()))
    retval += 'Stable Prod Grad: {}'.format(eltwise_node.StableProdGrad())
    return list_table([retval])


def batch_norm_node_label(batch_norm_node: IR.NNNode.BatchNormNode.BatchNormNode) -> str:
    retval = 'BatchNorm Node<br/>'
    retval += 'Axis: {}<br/>'.format(batch_norm_node.Axis())
    retval += 'Use Global Stats: {}<br/>'.format(batch_norm_node.UseGlobalStats())
    retval += 'Scale: {}<br/>'.format(batch_norm_node.Scale())
    retval += 'Eps: {}<br/>'.format(batch_norm_node.Eps())
    return list_table([retval])


def scale_node_label(scale_node: IR.NNNode.ScaleNode.ScaleNode) -> str:
    retval = 'Scale Node<br/>'
    retval += 'BiasTerm: {}<br/>'.format(scale_node.BiasTerm())
    return list_table([retval])


def permute_node_label(permute_node: IR.NNNode.PermuteNode.PermuteNode) -> str:
    retval = 'Permute Node<br/>'
    retval += 'Order: {}<br/>'.format(dim4_label(permute_node.Order()))
    return list_table([retval])


def reshape_node_label(reshape_node: IR.NNNode.ReshapeNode.ReshapeNode) -> str:
    retval = 'ReshapeNode Node<br/>'
    retval += 'Shape: {}<br/>'.format(dim4_label(reshape_node.Shape()))
    return list_table([retval])


def slice_node_label(slice_node: IR.NNNode.SliceNode.SliceNode) -> str:
    retval = 'Slice Node<br/>'
    retval += 'Axis: {}<br/>'.format(slice_node.Axis())
    retval += 'Points: {}<br/>'.format(typed_array_label(slice_node.Points()))
    return list_table([retval])


def tile_node_label(tile_node: IR.NNNode.TileNode.TileNode) -> str:
    retval = 'Tile Node<br/>'
    retval += 'Axis: {}<br/>'.format(tile_node.Axis())
    retval += 'Tiles: {}<br/>'.format(tile_node.Tiles())
    return list_table([retval])


# Labels for OpNode
def shift_node_label(shift_node: IR.OPNode.ShiftNode.ShiftNode) -> str:
    quantization_shift = shift_node.QuantizationShiftAsNumpy()
    multiplication_shift = shift_node.MultiplicationShiftAsNumpy()
    activation_shift = shift_node.ActivationShiftAsNumpy()
    lut_scale = shift_node.LutScaleAsNumpy()
    lut_bias = shift_node.LutBiasAsNumpy()
    grelu_info = shift_node.GreluInfoAsNumpy()

    retval = 'Shift Node<br/>'
    if type(quantization_shift) is np.ndarray and quantization_shift.any():
        retval += 'Quantization Shift: {}<br/>'.format(quantization_shift.shape)
    if type(multiplication_shift) is np.ndarray and multiplication_shift.any():
        retval += 'Multiplication Shift: {}<br/>'.format(multiplication_shift)
    if type(activation_shift) is np.ndarray and activation_shift.any():
        retval += 'Activation Shift: {}<br/>'.format(activation_shift)
    if type(lut_scale) is np.ndarray and lut_scale.any():
        retval += 'Lut Scale: {}<br/>'.format(lut_scale)
    if type(lut_bias) is np.ndarray and lut_bias.any():
        retval += 'Lut Bias: {}<br/>'.format(lut_bias)
    if type(grelu_info) is np.ndarray and grelu_info.any():
        retval += 'Grelu Info: {}<br/>'.format(grelu_info)

    return list_table([retval])

# Torch Aten Ops
def aten_slice_node_label(aten_slice_node: IR.NNNode.AtenSliceNode.AtenSliceNode) -> str:
    dim = aten_slice_node.Dim()
    start = aten_slice_node.Start()
    end = aten_slice_node.End()
    step = aten_slice_node.Step()

    retval = 'AtenSlice Node<br/>'
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    if start is not None:
        retval += 'Start: {}<br/>'.format(start)
    if end is not None:
        retval += 'End: {}<br/>'.format(end)
    if step is not None:
        retval += 'Step: {}<br/>'.format(step)
    return list_table([retval])


def aten_unsqueeze_node_label(aten_unsqueeze_node: IR.NNNode.AtenUnsqueezeNode.AtenUnsqueezeNode) -> str:
    retval = 'AtenUnsqueeze Node<br/>'
    dim = aten_unsqueeze_node.Dim()
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    return list_table([retval])


def aten_to_node_label(aten_to_node: IR.NNNode.AtenToNode.AtenToNode) -> str:
    retval = 'AtenTo Node<br/>'
    dtype = aten_to_node.Dtype()
    non_blocking = aten_to_node.NonBlocking()
    copy = aten_to_node.Copy()

    optional_memory_format = aten_to_node.OptionalMemoryFormat()
    if dtype is not None:
        retval += 'Dtype: {}<br/>'.format(dtype)
    if non_blocking is not None:
        retval += 'NonBlocking: {}<br/>'.format(non_blocking)
    if copy is not None:
        retval += 'Copy: {}<br/>'.format(copy)
    if optional_memory_format is not None:
        retval += 'OptionalMemoryFormat: {}<br/>'.format(optional_memory_format)
    return list_table([retval])


def aten_embedding_node_label(aten_embedding_node: IR.NNNode.AtenEmbeddingNode.AtenEmbeddingNode) -> str:
    retval = 'AtenEmbedding Node<br/>'
    padding_idx = aten_embedding_node.PaddingIdx()
    scale_grad_by_freq = aten_embedding_node.ScaleGradByFreq()
    sparse = aten_embedding_node.Sparse()

    if padding_idx is not None:
        retval += 'PaddingIdx: {}<br/>'.format(padding_idx)
    if scale_grad_by_freq is not None:
        retval += 'ScaleGradByFreq: {}<br/>'.format(scale_grad_by_freq)
    if sparse is not None:
        retval += 'Sparse: {}<br/>'.format(sparse)
    return list_table([retval])


def aten_copy_node_label(aten_copy_node: IR.NNNode.AtenCopyNode.AtenCopyNode) -> str:
    retval = 'AtenCopy Node<br/>'
    non_blocking = aten_copy_node.NonBlocking()
    
    if non_blocking is not None:
        retval += 'NonBlocking: {}<br/>'.format(non_blocking)
    return list_table([retval])


def aten_derive_index_node_label(aten_derive_index_node: IR.NNNode.AtenDeriveIndexNode.AtenDeriveIndexNode) -> str:
    retval = 'AtenDeriveIndex Node<br/>'
    # start = aten_derive_index_node.Start()
    start = None
    step = aten_derive_index_node.Step()
    
    if start is not None:
        retval += 'Start: {}<br/>'.format(start)
    if step is not None:
        retval += 'Step: {}<br/>'.format(step)
    return list_table([retval])


def aten_format_node_label(aten_format_node: IR.NNNode.AtenFormatNode.AtenFormatNode) -> str:
    retval = 'AtenFormat Node<br/>'
    assembly_format = aten_format_node.AssemblyFormat()
    
    if assembly_format is not None:
        retval += 'AssemblyFormat: {}<br/>'.format(assembly_format)
    return list_table([retval])


def aten_transpose_node_label(aten_transpose_node: IR.NNNode.AtenTransposeNode.AtenTransposeNode) -> str:
    retval = 'AtenTranspose Node<br/>'
    dim_0 = aten_transpose_node.Dim0()
    dim_1 = aten_transpose_node.Dim1()

    if dim_0 is not None:
        retval += 'Dim0: {}<br/>'.format(dim_0)
    if dim_1 is not None:
        retval += 'Dim1: {}<br/>'.format(dim_1)

    return list_table([retval])


def aten_size_node_label(aten_size_node: IR.NNNode.AtenSizeNode.AtenSizeNode) -> str:
    retval = 'AtenSizeNode Node<br/>'
    dim = aten_size_node.Dim()
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    return list_table([retval])


def aten_select_node_label(aten_select_node: IR.NNNode.AtenSelectNode.AtenSelectNode) -> str:
    retval = 'AtenSelect Node Node<br/>'
    dim = aten_select_node.Dim()
    index = aten_select_node.Index()
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    if index is not None:
        retval += 'Index: {}<br/>'.format(index)
    return list_table([retval])


def aten_expand_node_label(aten_expand_node: IR.NNNode.AtenExpandNode.AtenExpandNode) -> str:
    retval = 'AtenExpandNode Node<br/>'
    implicit = aten_expand_node.Implicit()
    if implicit is not None:
        retval += 'Implicit: {}<br/>'.format(implicit)
    return list_table([retval])


def aten_dropout_node_label(aten_dropout_node: IR.NNNode.AtenDropoutNode.AtenDropoutNode) -> str:
    retval = 'AtenDropout Node Node<br/>'
    proportion = aten_dropout_node.Proportion()
    train = aten_dropout_node.Train()

    if proportion is not None:
        retval += 'Proportion: {}<br/>'.format(proportion)
    if train is not None:
        retval += 'Train: {}<br/>'.format(train)
    return list_table([retval])


def aten_cat_node_label(aten_cat_node: IR.NNNode.AtenCatNode.AtenCatNode) -> str:
    retval = 'AtenCat Node<br/>'
    dim = aten_cat_node.Dim()
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    return list_table([retval])


def aten_lstm_node_label(aten_lstm_node: IR.NNNode.AtenLSTMNode.AtenLSTMNode) -> str:
    retval = 'AtenLSTM Node<br/>'
    has_biases = aten_lstm_node.HasBiases()
    num_layers = aten_lstm_node.NumLayers()
    dropout = aten_lstm_node.Dropout()
    train = aten_lstm_node.Train()
    bidirectional = aten_lstm_node.Bidirectional()
    batch_first = aten_lstm_node.BatchFirst()

    if has_biases is not None:
        retval += 'HasBiases: {}<br/>'.format(has_biases)
    if num_layers is not None:
        retval += 'NumLayers: {}<br/>'.format(num_layers)
    if dropout is not None:
        retval += 'Dropout: {}<br/>'.format(dropout)
    if train is not None:
        retval += 'Train: {}<br/>'.format(train)
    if bidirectional is not None:
        retval += 'Bidirectional: {}<br/>'.format(bidirectional)
    if batch_first is not None:
        retval += 'BatchFirst: {}<br/>'.format(batch_first)
    return list_table([retval])


def aten_pack_padded_sequence_node_label(aten_pack_padded_sequence_node: IR.NNNode.AtenPackPaddedSequenceNode.AtenPackPaddedSequenceNode) -> str:
    retval = 'AtenPackPaddedSequence Node<br/>'
    batch_first = aten_pack_padded_sequence_node.BatchFirst()
    if batch_first is not None:
        retval += 'BatchFirst: {}<br/>'.format(batch_first)
    return list_table([retval])


def aten_pad_packed_sequence_node_label(aten_pad_packed_sequence_node: IR.NNNode.AtenPadPackedSequenceNode.AtenPadPackedSequenceNode) -> str:
    retval = 'AtenPadPackedSequence Node<br/>'
    batch_first = aten_pad_packed_sequence_node.BatchFirst()
    padding_value = aten_pad_packed_sequence_node.PaddingValue()
    if batch_first is not None:
        retval += 'BatchFirst: {}<br/>'.format(batch_first)
    if padding_value is not None:
        retval += 'PaddingValue: {}<br/>'.format(padding_value)
    return list_table([retval])


def aten_set_item_node_label(aten_set_item_node: IR.NNNode.AtenSetItemNode.AtenSetItemNode) -> str:
    retval = 'AtenSetItem Node<br/>'
    indices = aten_set_item_node.Indices()
    if indices is not None:
        retval += 'Indices: {}<br/>'.format(indices)
    return list_table([retval])


def aten_arange_node_label(aten_arange_node: IR.NNNode.AtenArangeNode.AtenArangeNode) -> str:
    retval = 'AtenArange Node<br/>'
    start = aten_arange_node.Start()
    step = aten_arange_node.Step()
    dtype = aten_arange_node.Dtype()
    layout = aten_arange_node.Layout()
    pin_memory = aten_arange_node.PinMemory()

    if start is not None:
        retval += 'Start: {}<br/>'.format(start)
    if step is not None:
        retval += 'Step: {}<br/>'.format(step)
    if dtype is not None:
        retval += 'Dtype: {}<br/>'.format(dtype)
    if layout is not None:
        retval += 'Layout: {}<br/>'.format(layout)
    if pin_memory is not None:
        retval += 'PinMemory: {}<br/>'.format(pin_memory)

    return list_table([retval])
    

def aten_as_tensor_node_label(aten_as_tensor_node: IR.NNNode.AtenAsTensorNode.AtenAsTensorNode) -> str:
    retval = 'AtenAsTensor Node<br/>'
    dtype = aten_as_tensor_node.Dtype()
    device = aten_as_tensor_node.Device()
    if dtype is not None:
        retval += 'Dtype: {}<br/>'.format(dtype)
    if device is not None:
        retval += 'Device: {}<br/>'.format(device)
    return list_table([retval])


def aten_clamp_node_label(aaten_clamp_node: IR.NNNode.AtenClampNode.AtenClampNode) -> str:
    retval = 'AtenClamp Node<br/>'
    min = aaten_clamp_node.Min()
    max = aaten_clamp_node.Max()
    if min is not None:
        retval += 'Min: {}<br/>'.format(min)
    if max is not None:
        retval += 'Max: {}<br/>'.format(max)
    return list_table([retval])


def aten_chunk_node_label(aten_chunk_node: IR.NNNode.AtenChunkNode.AtenChunkNode) -> str:
    retval = 'AtenChunk Node<br/>'
    chunks = aten_chunk_node.Chunks()
    dim = aten_chunk_node.Dim()
    if chunks is not None:
        retval += 'Chunks: {}<br/>'.format(chunks)
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    return list_table([retval])


def aten_contiguous_node_label(aten_contiguous_node: IR.NNNode.AtenContiguousNode.AtenContiguousNode) -> str:
    retval = 'AtenContiguous Node<br/>'
    memory_format = aten_contiguous_node.MemoryFormat()
    if memory_format is not None:
        retval += 'MemoryFormat: {}<br/>'.format(memory_format)
    return list_table([retval])


def aten_gather_node_label(aten_gather_node: IR.NNNode.AtenGatherNode.AtenGatherNode) -> str:
    retval = 'AtenGather Node<br/>'
    dim = aten_gather_node.Dim()
    sparse_grad = aten_gather_node.SparseGrad()
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    if sparse_grad is not None:
        retval += 'SparseGrad: {}<br/>'.format(sparse_grad)
    return list_table([retval])


def aten_index_put_node_label(aten_index_put_node: IR.NNNode.AtenIndexPutNode.AtenIndexPutNode) -> str:
    retval = 'AtenIndexPut Node<br/>'
    accumulate = aten_index_put_node.Accumulate()
    if accumulate is not None:
        retval += 'Accumulate: {}<br/>'.format(accumulate)
    return list_table([retval])


def aten_leaky_relu_node_label(aten_leaky_relu_node: IR.NNNode.AtenLeakyReluNode.AtenLeakyReluNode) -> str:
    retval = 'AtenLeakyRelu Node<br/>'
    scalar = aten_leaky_relu_node.Scalar()
    if scalar is not None:
        retval += 'Scalar: {}<br/>'.format(scalar)
    return list_table([retval])


def aten_log_softmax_node_label(aten_log_softmax_node: IR.NNNode.AtenLogSoftmaxNode.AtenLogSoftmaxNode) -> str:
    retval = 'AtenLogSoftmax Node<br/>'
    dim = aten_log_softmax_node.Dim()
    dtype = aten_log_softmax_node.Dtype()
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    if dtype is not None:
        retval += 'Dtype: {}<br/>'.format(dtype)
    return list_table([retval])


def aten_masked_fill_node_label(aten_masked_fill_node: IR.NNNode.AtenMaskedFillNode.AtenMaskedFillNode) -> str:
    retval = 'AtenMaskedFill Node<br/>'
    value = aten_masked_fill_node.Value()
    if value is not None:
        retval += 'Value: {}<br/>'.format(value)
    return list_table([retval])


def aten_min_node_label(aten_min_node: IR.NNNode.AtenMinNode.AtenMinNode) -> str:
    retval = 'AtenMin Node<br/>'
    dim_or_y = aten_min_node.DimOrY()
    keep_dim = aten_min_node.KeepDim()
    if dim_or_y is not None:
        retval += 'Dim_or_y: {}<br/>'.format(dim_or_y)
    if keep_dim is not None:
        retval += 'KeepDim: {}<br/>'.format(keep_dim)
    
    return list_table([retval])


def aten_mul_node_label(aten_mul_node: IR.NNNode.AtenMulNode.AtenMulNode) -> str:
    retval = 'AtenMul Node<br/>'
    other = aten_mul_node.Other()
    if other is not None:
        retval += 'Other: {}<br/>'.format(other)
    return list_table([retval])


def aten_ones_node_label(aten_ones_node: IR.NNNode.AtenOnesNode.AtenOnesNode) -> str:
    retval = 'AtenOnes Node<br/>'
    size = aten_ones_node.Size()
    dtype = aten_ones_node.Dtype()
    layout = aten_ones_node.Layout()
    device = aten_ones_node.Device()

    if size is not None:
        retval += 'Size: {}<br/>'.format(size)
    if dtype is not None:
        retval += 'Dtype: {}<br/>'.format(dtype)
    if layout is not None:
        retval += 'Layout: {}<br/>'.format(layout)
    if device is not None:
        retval += 'Device: {}<br/>'.format(device)
    return list_table([retval])


def aten_softmax_node_label(aten_softmax_node: IR.NNNode.AtenSoftmaxNode.AtenSoftmaxNode) -> str:
    retval = 'AtenSoftmax Node<br/>'
    dim = aten_softmax_node.Dim()
    dtype = aten_softmax_node.Dtype()
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    if dtype is not None:
        retval += 'Dtype: {}<br/>'.format(dtype)
    return list_table([retval])


def aten_squeeze_node_label(aten_squeeze_node: IR.NNNode.AtenSqueezeNode.AtenSqueezeNode) -> str:
    retval = 'AtenSqueeze Node<br/>'
    dim = aten_squeeze_node.Dim()
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    return list_table([retval])


def aten_sum_node_label(aten_sum_node: IR.NNNode.AtenSumNode.AtenSumNode) -> str:
    retval = 'AtenSum Node<br/>'
    dim = aten_sum_node.Dim()
    keepdim = aten_sum_node.KeepDim()
    dtype = aten_sum_node.KeepDim()
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    if keepdim is not None:
        retval += 'KeepDim: {}<br/>'.format(keepdim)
    if dtype is not None:
        retval += 'Dtype: {}<br/>'.format(dtype)
    return list_table([retval])
    

def aten_topk_node_label(aten_topk_node: IR.NNNode.AtenTopkNode.AtenTopkNode) -> str:
    retval = 'AtenTopk Node<br/>'
    k = aten_topk_node.K()
    dim = aten_topk_node.Dim()
    largest = aten_topk_node.Largest()
    sorted = aten_topk_node.Sorted()
    if k is not None:
        retval += 'K: {}<br/>'.format(k)
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    if largest is not None:
        retval += 'Largest: {}<br/>'.format(largest)
    if sorted is not None:
        retval += 'Sorted: {}<br/>'.format(sorted)
    return list_table([retval])


def aten_warn_node_label(aten_warn_node: IR.NNNode.AtenWarnNode.AtenWarnNode) -> str:
    retval = 'AtenWarn Node<br/>'
    value = aten_warn_node.Value()
    if value is not None:
        retval += 'Value: {}<br/>'.format(value)
    return list_table([retval])


def aten_index_select_node_label(aten_index_select_node: IR.NNNode.AtenIndexSelectNode.AtenIndexSelectNode) -> str:
    retval = 'AtenIndexSelect Node<br/>'
    dim = aten_index_select_node.Dim()
    if dim is not None:
        retval += 'Dim: {}<br/>'.format(dim)
    return list_table([retval])


def aten_conv2d_node_label(aten_conv2d_node: IR.NNNode.AtenConv2dNode.AtenConv2dNode) -> str:
    retval = 'AtenConv2d Node<br/>'
    stride = aten_conv2d_node.Stride()
    padding = aten_conv2d_node.Padding()
    dilation = aten_conv2d_node.Dilation()
    groups = aten_conv2d_node.Groups()
    if stride is not None:
       retval += 'Stride: {}<br/>'.format(dim2_label(stride))
    if padding is not None:
        retval += 'Padding: {}<br/>'.format(pad4_label(padding))
    if dilation is not None:
        retval += 'Dilation: {}<br/>'.format(dim2_label(dilation))
    if groups is not None:
        retval += 'Groups: {}<br/>'.format(groups)
    return list_table([retval])


def aten_maxpool2d_node_label(aten_maxpool2d_node: IR.NNNode.AtenMaxPool2dNode.AtenMaxPool2dNode) -> str:
    retval = 'AtenMaxPool2d Node<br/>'
    kernel_size = aten_maxpool2d_node.KernelSize()
    pad = aten_maxpool2d_node.Pad()
    stride = aten_maxpool2d_node.Stride()
    dilation = aten_maxpool2d_node.Dilation()
    return_indices = aten_maxpool2d_node.ReturnIndices()
    if kernel_size is not None:
        retval += 'KernelSize: {}<br/>'.format(dim2_label(kernel_size))
    if pad is not None:
        retval += 'Pad: {}<br/>'.format(pad4_label(pad))
    if stride is not None:
        retval += 'Stride: {}<br/>'.format(dim2_label(stride))
    if dilation is not None:
        retval += 'Dialation: {}<br/>'.format(dim2_label(dilation))
    if return_indices is not None:
        retval += 'ReturnIndices: {}<br/>'.format(return_indices)
    return list_table([retval])


def prim_constant_node_label(prim_constant_node: IR.CONTROLNode.PrimConstantNode.PrimConstantNode) -> str:
    # GDataType list comes from the defination of DTensor of GraphGen:
    # URL: https://github.sec.samsung.net/FIM/GraphGen/blob/amdgpu_pim/core/graphgen_network/graphgen_types.h
    GDataType = ['UNDEFINED','INT8','UINT8','INT16','UINT16','INT32','INT64','FLOAT16',
                'FLOAT32','FLOAT64','BOOL','STRING','DEVICE','TENSOR']
    named_datatype = {idx : symbol for idx, symbol in enumerate(GDataType)}
    retval = 'PrimConstant Node<br/>'
    tensor_shape = prim_constant_node.TensorShape()
    data_type = prim_constant_node.DataType()
    bit_width = prim_constant_node.BitWidth()
    if tensor_shape is not None:
        retval += 'TensorShape: {}<br/>'.format(dim4_label(tensor_shape))
    if data_type is not None:
        retval += 'DataType: {}<br/>'.format(named_datatype[data_type])
    if bit_width is not None:
        retval += 'BitWidth: {}<br/>'.format(bit_width)
    ScalarValueType = ['INT8','UINT8','INT16','UINT16','INT32','INT64','FLOAT16','FLOAT32','FLOAT64','BOOL']
    numpy_data = prim_constant_node.DataAsNumpy()
    if named_datatype[data_type] in ScalarValueType:
        retval += 'Value: {}<br/>'.format(numpy_data[0])
    return list_table([retval])


def prim_loop_index_node_label(prim_loop_index_node: IR.CONTROLNode.PrimLoopIndexNode.PrimLoopIndexNode) -> str:
    retval = 'PrimLoopIndex Node<br/>'
    index = prim_loop_index_node.Index()
    if index is not None:
        retval += 'LoopIndex: {}<br/>'.format(index)
    return list_table([retval])


def prim_loop_node_label(prim_loop_node: IR.CONTROLNode.PrimLoopNode.PrimLoopNode) -> str:
    retval = 'PrimLoop Node<br/>'
    trip_count = prim_loop_node.TripCount()
    cond = prim_loop_node.Cond()
    body_net = prim_loop_node.BodyNet()
    if trip_count is not None:
        retval += 'TripCount: {}<br/>'.format(trip_count)
    if cond is not None:
        retval += 'Cond: {}<br/>'.format(cond)
    if body_net is not None:
        retval += 'BodyNet: {}<br/>'.format(body_net)
    return list_table([retval])


def prim_tuple_index_node_label(prim_tuple_index_node: IR.CONTROLNode.PrimTupleIndexNode.PrimTupleIndexNode) -> str:
    retval = 'PrimTupleIndex Node<br/>'
    index = prim_tuple_index_node.Index()
    if index is not None:
        retval += 'Index: {}<br/>'.format(index)
    return list_table([retval])

def prim_if_node_label(prim_if_node: IR.CONTROLNode.PrimIfNode.PrimIfNode) -> str:
    retval = 'PrimIf Node<br/>'
    then_net = prim_if_node.ThenNet()
    else_net = prim_if_node.ElseNet()
    if then_net is not None:
        retval += 'ThenNet: {}<br/>'.format(then_net)
    if else_net is not None:
        retval += 'ElseNet: {}<br/>'.format(else_net)
    return list_table([retval])


def prim_call_method_node_label(prim_call_method_node: IR.CONTROLNode.PrimCallMethodNode.PrimCallMethodNode) -> str:
    retval = 'PrimCallMethod Node<br/>'
    target_network_name = prim_call_method_node.TargetNetworkName()
    if target_network_name is not None:
        retval += 'TargetNetworkName: {}<br/>'.format(target_network_name)
    return list_table([retval])

def prim_variable_node_label(prim_variable_node: IR.CONTROLNode.PrimVariableNode.PrimVariableNode) -> str:
    retval = 'PrimVariable Node<br/>'
    # data_type  = prim_variable_node.DataType()
    data_type = None
    if data_type  is not None:
        retval += 'DataType : {}<br/>'.format(data_type )
    return list_table([retval])


# Labels for Blob
def data_edge_label(ir_data_edge: ir_enums.EdgeType.DataEdge) -> str:
    return list_table(['Data Edge Blob'])


def kernel_blob_edge_label(conv_node: IR.NNNode.ConvNode.ConvNode) -> str:
    return list_table(['Kernel Blob'])


def bias_blob_edge_label(conv_node: IR.NNNode.ConvNode.ConvNode) -> str:
    return list_table(['Bias Blob'])


def weight_blob_edge_label(fc_node: IR.NNNode.FullyConnectedNode.FullyConnectedNode) -> str:
    return list_table(['Weight Blob'])


def aten_bias_blob_edge_label() -> str:
    return list_table(['Bias Blob'])


def aten_weight_blob_edge_label() -> str:
    return list_table(['Weight Blob'])


def blob_node_label(blob: IR.Blob.Blob) -> str:
    retval = ''
    # retval += 'QuantType: {}<br/>'.format(ir_enums.QuantType.get_name(blob.QuantType()))
    retval += 'ShapeType: {}<br/>'.format(ir_enums.ShapeType.get_name(blob.Shape()))
    retval += 'Dim: {}<br/>'.format(dim4_label(blob.Dim()))
    retval += 'DataType: {}<br/>'.format(ir_enums.DataType.get_name(blob.DataType()))
    retval += 'BitWidth: {}<br/>'.format(blob.BitWidth())
    if blob.HwInfo():
        hwinfo = blob.HwInfo()
        retval += 'LivenessNode: {} - {}<br/>'.format(hwinfo.LivenessStartNode(),
                                                      hwinfo.LivenessEndNode())
        retval += 'Alignment: {}<br/>'.format(dim4_label(hwinfo.AlignmentUnit()))
    # retval += 'ZeroPoint: {}<br/>'.format(blob.ZeroPoint())
    # FIXME: frac_len and quant_leval should be handled later
    # frac_len = []
    # if blob.FracLenLength() <= 5:
    #     for idx in range(0, blob.FracLenLength()):
    #         frac_len.append(blob.FracLen(idx))
    # else:
    #     for idx in range(0, 3):
    #         frac_len.append(blob.FracLen(idx))
    #     frac_len.append('...')
    #     frac_len.append(blob.FracLen(blob.FracLenLength() - 1))
    # retval += 'FracLen: {} size: {}<br/>'.format(frac_len, blob.FracLenLength())
    return list_table([blob_title(blob), retval])


def blob_title(blob: IR.Blob.Blob) -> str:
    return bold('[BLOB:{}]<br/>{}'.format(blob.Id(), blob.Name().decode('utf-8')))


def ir_node_title(ir_node: IR.Node.Node) -> str:
    return bold('[NODE:{}]<br/>{}'.format(ir_node.Id(), ir_node.Name().decode('utf-8')))


def ir_edge_title(ir_edge: IR.Edge.Edge, hide_name: bool) -> str:
    title = '[EDGE:{}]'.format(ir_edge.Id())
    if not hide_name:
        title += '<br/>{}'.format(ir_edge.Name().decode('utf-8'))
    return bold(title)


# for types
def dim2_label(dim2: IR.Type.Dim2.Dim2) -> str:
    return '(h:{} w:{})'.format(dim2.H(), dim2.W())


def dim4_label(dim4: IR.Type.Dim4.Dim4) -> str:
    if dim4 is None:
        return '(No dim)'
    return '(n:{} c:{} h:{} w:{})'.format(dim4.N(), dim4.C(), dim4.H(), dim4.W())


def pad4_label(pad4: IR.Type.Pad4.Pad4) -> str:
    return '(t:{} b:{} l:{} r:{})'.format(pad4.T(), pad4.B(), pad4.L(), pad4.R())


def typed_array_label(typed_array: IR.Type.TypedArray.TypedArray) -> str:
    as_f32_arr = typed_array.F32ArrAsNumpy()
    as_i16_arr = typed_array.I16ArrAsNumpy()
    as_i8_arr = typed_array.I8ArrAsNumpy()
    as_ui8_arr = typed_array.Ui8ArrAsNumpy()
    as_i64_arr = typed_array.I64ArrAsNumpy()

    if type(as_f32_arr) is np.ndarray:
        return '{}'.format(as_f32_arr)
    if type(as_i16_arr) is np.ndarray:
        return '{}'.format(as_i16_arr)
    if type(as_i8_arr) is np.ndarray:
        return '{}'.format(as_i8_arr)
    if type(as_ui8_arr) is np.ndarray:
        return '{}'.format(as_ui8_arr)
    if type(as_i64_arr) is np.ndarray:
        return '{}'.format(as_i64_arr)
    return 'Unknown type in TypedArray'


# for offsets and sizes
def offset_or_size_label(value: int) -> str:
    ''' Returns the integer value provided as formatted string with "decimal (hex)" format '''
    return '{} (0x{:x})'.format(value, value)


# for html elements
def html_str(text: str) -> str:
    return '<{}>'.format(text)


def table(tr) -> str:
    return '<table BORDER="0" cellspacing="0">{}</table>'.format(tr)


def list_table(rows: list) -> str:
    rows = [x for x in rows if x is not None]
    for num, row in enumerate(rows):
        if row is None:
            continue
        if str(row).startswith('<tr>') and str(row).endswith('</tr>'):
            rows[num] = str(row)
        elif '<hr/>' == str(row):
            rows[num] = '<tr><td></td></tr><hr/><tr><td></td></tr>'
        else:
            rows[num] = tr(str(row))
    return table(''.join(rows))


def tr(text: str, align='center') -> str:
    return '<tr><td align="{0}" balign="{0}">{1}</td></tr>'.format(align, text)


def bold(text: str) -> str:
    return '<b>{}</b>'.format(text)
