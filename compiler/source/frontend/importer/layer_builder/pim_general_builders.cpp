/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#include "frontend/importer/layer_builder/layer_builder.h"

#define DECLARE_TORCH_OP_BUILDER(op_name, type_name, layer_name)                                \
    namespace nn_compiler                                                                       \
    {                                                                                           \
    namespace frontend                                                                          \
    {                                                                                           \
    std::shared_ptr<ir::NNLayer> op_name##Builder::buildLayer(const torch::jit::Node* node_ref) \
    {                                                                                           \
        DLOG(INFO) << "build " << convertLayerTypeToString(type_name);                          \
        nn_compiler::ir::LayerType type = type_name;                                            \
        std::string name = "";                                                                  \
        layer_name = std::make_shared<ir::op_name##Layer>(name, type);                          \
        const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(layer_name);                 \
        return layer;                                                                           \
    }                                                                                           \
    }                                                                                           \
    }  // namespace nn_compiler

DECLARE_TORCH_OP_BUILDER(AtenAbs, nn_compiler::ir::LayerType::ATENABS, aten_abs_layer_)
DECLARE_TORCH_OP_BUILDER(AtenAdd, nn_compiler::ir::LayerType::ATENADD, aten_add_layer_)
DECLARE_TORCH_OP_BUILDER(AtenAddmm, nn_compiler::ir::LayerType::ATENADDMM, aten_addmm_layer_)
DECLARE_TORCH_OP_BUILDER(AtenAnd, nn_compiler::ir::LayerType::ATENAND, aten_and_layer_)
DECLARE_TORCH_OP_BUILDER(AtenAny, nn_compiler::ir::LayerType::ATENANY, aten_any_layer_)
DECLARE_TORCH_OP_BUILDER(AtenAppend, nn_compiler::ir::LayerType::ATENAPPEND, aten_append_layer_)
DECLARE_TORCH_OP_BUILDER(AtenArange1, nn_compiler::ir::LayerType::ATENARANGE1, aten_arange_layer_)
DECLARE_TORCH_OP_BUILDER(AtenArange2, nn_compiler::ir::LayerType::ATENARANGE2, aten_arange_layer_)
DECLARE_TORCH_OP_BUILDER(AtenArange3, nn_compiler::ir::LayerType::ATENARANGE3, aten_arange_layer_)
DECLARE_TORCH_OP_BUILDER(AtenArgmax, nn_compiler::ir::LayerType::ATENARGMAX, aten_argmax_layer_)
DECLARE_TORCH_OP_BUILDER(AtenAsTensor, nn_compiler::ir::LayerType::ATENASTENSOR, aten_as_tensor_layer_)
DECLARE_TORCH_OP_BUILDER(AtenBitwiseNot, nn_compiler::ir::LayerType::ATENBITWISENOT, aten_bitwise_not_layer_)
DECLARE_TORCH_OP_BUILDER(AtenBmm, nn_compiler::ir::LayerType::ATENBMM, aten_bmm_layer_)
DECLARE_TORCH_OP_BUILDER(AtenBool, nn_compiler::ir::LayerType::ATENBOOL, aten_bool_layer_)
DECLARE_TORCH_OP_BUILDER(AtenCat, nn_compiler::ir::LayerType::ATENCAT, aten_cat_layer_)
DECLARE_TORCH_OP_BUILDER(AtenCeil, nn_compiler::ir::LayerType::ATENCEIL, aten_ceil_layer_)
DECLARE_TORCH_OP_BUILDER(AtenChunk, nn_compiler::ir::LayerType::ATENCHUNK, aten_chunk_layer_)
DECLARE_TORCH_OP_BUILDER(AtenClamp, nn_compiler::ir::LayerType::ATENCLAMP, aten_clamp_layer_)
DECLARE_TORCH_OP_BUILDER(AtenClear, nn_compiler::ir::LayerType::ATENCLEAR, aten_clear_layer_)
DECLARE_TORCH_OP_BUILDER(AtenClone, nn_compiler::ir::LayerType::ATENCLONE, aten_clone_layer_)
DECLARE_TORCH_OP_BUILDER(AtenContiguous, nn_compiler::ir::LayerType::ATENCONTIGUOUS, aten_contiguous_layer_)
DECLARE_TORCH_OP_BUILDER(AtenCopy, nn_compiler::ir::LayerType::ATENCOPY, aten_copy_layer_)
DECLARE_TORCH_OP_BUILDER(AtenCpu, nn_compiler::ir::LayerType::ATENCPU, aten_cpu_layer_)
DECLARE_TORCH_OP_BUILDER(AtenCuda, nn_compiler::ir::LayerType::ATENCUDA, aten_cuda_layer_)
DECLARE_TORCH_OP_BUILDER(AtenCumsum, nn_compiler::ir::LayerType::ATENCUMSUM, aten_cumsum_layer_)
DECLARE_TORCH_OP_BUILDER(AtenDeriveIndex, nn_compiler::ir::LayerType::ATENDERIVEINDEX, aten_derive_index_layer_)
DECLARE_TORCH_OP_BUILDER(AtenDetach, nn_compiler::ir::LayerType::ATENDETACH, aten_detach_layer_)
DECLARE_TORCH_OP_BUILDER(AtenDim, nn_compiler::ir::LayerType::ATENDIM, aten_dim_layer_)
DECLARE_TORCH_OP_BUILDER(AtenDiv, nn_compiler::ir::LayerType::ATENDIV, aten_div_layer_)
DECLARE_TORCH_OP_BUILDER(AtenDropout, nn_compiler::ir::LayerType::ATENDROPOUT, aten_dropout_layer_)
DECLARE_TORCH_OP_BUILDER(AtenEmbedding, nn_compiler::ir::LayerType::ATENEMBEDDING, aten_embedding_layer_)
DECLARE_TORCH_OP_BUILDER(AtenEinsum, nn_compiler::ir::LayerType::ATENEINSUM, aten_einsum_layer_)
DECLARE_TORCH_OP_BUILDER(AtenEq, nn_compiler::ir::LayerType::ATENEQ, aten_eq_layer_)
DECLARE_TORCH_OP_BUILDER(AtenEqual, nn_compiler::ir::LayerType::ATENEQUAL, aten_equal_layer_)
DECLARE_TORCH_OP_BUILDER(AtenExpand, nn_compiler::ir::LayerType::ATENEXPAND, aten_expand_layer_)
DECLARE_TORCH_OP_BUILDER(AtenFill, nn_compiler::ir::LayerType::ATENFILL, aten_fill_layer_)
DECLARE_TORCH_OP_BUILDER(AtenFloorDivide, nn_compiler::ir::LayerType::ATENFLOORDIVIDE, aten_floor_divide_layer_)
DECLARE_TORCH_OP_BUILDER(AtenFormat, nn_compiler::ir::LayerType::ATENFORMAT, aten_format_layer_)
DECLARE_TORCH_OP_BUILDER(AtenFullLike, nn_compiler::ir::LayerType::ATENFULLLIKE, aten_full_like_layer_)
DECLARE_TORCH_OP_BUILDER(AtenGather, nn_compiler::ir::LayerType::ATENGATHER, aten_gather_layer_)
DECLARE_TORCH_OP_BUILDER(AtenGe, nn_compiler::ir::LayerType::ATENGE, aten_ge_layer_)
DECLARE_TORCH_OP_BUILDER(AtenGetItem, nn_compiler::ir::LayerType::ATENGETITEM, aten_get_item_layer_)
DECLARE_TORCH_OP_BUILDER(AtenGt, nn_compiler::ir::LayerType::ATENGT, aten_gt_layer_)
DECLARE_TORCH_OP_BUILDER(AtenIndex, nn_compiler::ir::LayerType::ATENINDEX, aten_index_layer_)
DECLARE_TORCH_OP_BUILDER(AtenIndexPut, nn_compiler::ir::LayerType::ATENINDEXPUT, aten_index_put_layer_)
DECLARE_TORCH_OP_BUILDER(AtenIndexSelect, nn_compiler::ir::LayerType::ATENINDEXSELECT, aten_index_select_layer_)
DECLARE_TORCH_OP_BUILDER(AtenInt, nn_compiler::ir::LayerType::ATENINT, aten_int_layer_)
DECLARE_TORCH_OP_BUILDER(AtenIntImplicit, nn_compiler::ir::LayerType::ATENINTIMPLICIT, aten_int_implicit_layer_)
DECLARE_TORCH_OP_BUILDER(AtenIs, nn_compiler::ir::LayerType::ATENIS, aten_is_layer_)
DECLARE_TORCH_OP_BUILDER(AtenIsInf, nn_compiler::ir::LayerType::ATENISINF, aten_is_inf_layer_)
DECLARE_TORCH_OP_BUILDER(AtenIsNot, nn_compiler::ir::LayerType::ATENISNOT, aten_is_not_layer_)
DECLARE_TORCH_OP_BUILDER(AtenItem, nn_compiler::ir::LayerType::ATENITEM, aten_item_layer_)
DECLARE_TORCH_OP_BUILDER(AtenLeakyRelu, nn_compiler::ir::LayerType::ATENLEAKYRELU, aten_leaky_relu_layer_)
DECLARE_TORCH_OP_BUILDER(AtenLe, nn_compiler::ir::LayerType::ATENLE, aten_le_layer_)
DECLARE_TORCH_OP_BUILDER(AtenLen, nn_compiler::ir::LayerType::ATENLEN, aten_len_layer_)
DECLARE_TORCH_OP_BUILDER(AtenList, nn_compiler::ir::LayerType::ATENLIST, aten_list_layer_)
DECLARE_TORCH_OP_BUILDER(AtenLog, nn_compiler::ir::LayerType::ATENLOG, aten_log_layer_)
DECLARE_TORCH_OP_BUILDER(AtenLogSoftmax, nn_compiler::ir::LayerType::ATENLOGSOFTMAX, aten_log_softmax_layer_)
DECLARE_TORCH_OP_BUILDER(AtenLt, nn_compiler::ir::LayerType::ATENLT, aten_lt_layer_)
DECLARE_TORCH_OP_BUILDER(AtenMaskedFill, nn_compiler::ir::LayerType::ATENMASKEDFILL, aten_masked_fill_layer_)
DECLARE_TORCH_OP_BUILDER(AtenMaskedSelect, nn_compiler::ir::LayerType::ATENMASKEDSELECT, aten_masked_select_layer_)
DECLARE_TORCH_OP_BUILDER(AtenMatmul, nn_compiler::ir::LayerType::ATENMATMUL, aten_matmul_layer_)
DECLARE_TORCH_OP_BUILDER(AtenMax, nn_compiler::ir::LayerType::ATENMAX, aten_max_layer_)
DECLARE_TORCH_OP_BUILDER(AtenMaxPool2d, nn_compiler::ir::LayerType::ATENMAXPOOL2D, aten_max_pool2d_layer_)
DECLARE_TORCH_OP_BUILDER(AtenMean, nn_compiler::ir::LayerType::ATENMEAN, aten_mean_layer_)
DECLARE_TORCH_OP_BUILDER(AtenMin, nn_compiler::ir::LayerType::ATENMIN, aten_min_layer_)
DECLARE_TORCH_OP_BUILDER(AtenMul, nn_compiler::ir::LayerType::ATENMUL, aten_mul_layer_)
DECLARE_TORCH_OP_BUILDER(AtenNe, nn_compiler::ir::LayerType::ATENNE, aten_ne_layer_)
DECLARE_TORCH_OP_BUILDER(AtenNeg, nn_compiler::ir::LayerType::ATENNEG, aten_neg_layer_)
DECLARE_TORCH_OP_BUILDER(AtenNorm, nn_compiler::ir::LayerType::ATENNORM, aten_norm_layer_)
DECLARE_TORCH_OP_BUILDER(AtenNot, nn_compiler::ir::LayerType::ATENNOT, aten_not_layer_)
DECLARE_TORCH_OP_BUILDER(AtenOneHot, nn_compiler::ir::LayerType::ATENONEHOT, aten_one_hot_layer_)
DECLARE_TORCH_OP_BUILDER(AtenOnes, nn_compiler::ir::LayerType::ATENONES, aten_ones_layer_)
DECLARE_TORCH_OP_BUILDER(AtenPackPaddedSequence, nn_compiler::ir::LayerType::ATENPACKPADDEDSEQUENCE,
                         aten_pack_padded_sequence_layer_)
DECLARE_TORCH_OP_BUILDER(AtenPadPackedSequence, nn_compiler::ir::LayerType::ATENPADPACKEDSEQUENCE,
                         aten_pad_packed_sequence_layer_)
DECLARE_TORCH_OP_BUILDER(AtenPermute, nn_compiler::ir::LayerType::ATENPERMUTE, aten_permute_layer_)
DECLARE_TORCH_OP_BUILDER(AtenPow, nn_compiler::ir::LayerType::ATENPOW, aten_pow_layer_)
DECLARE_TORCH_OP_BUILDER(AtenRelu, nn_compiler::ir::LayerType::ATENRELU, aten_relu_layer_)
DECLARE_TORCH_OP_BUILDER(AtenReshape, nn_compiler::ir::LayerType::ATENRESHAPE, aten_reshape_layer_)
DECLARE_TORCH_OP_BUILDER(AtenRemainder, nn_compiler::ir::LayerType::ATENREMAINDER, aten_remainder_layer_)
DECLARE_TORCH_OP_BUILDER(AtenRepeat, nn_compiler::ir::LayerType::ATENREPEAT, aten_repeat_layer_)
DECLARE_TORCH_OP_BUILDER(AtenRsqrt, nn_compiler::ir::LayerType::ATENRSQRT, aten_rsqrt_layer_)
DECLARE_TORCH_OP_BUILDER(AtenSelect, nn_compiler::ir::LayerType::ATENSELECT, aten_select_layer_)
DECLARE_TORCH_OP_BUILDER(AtenSetItem, nn_compiler::ir::LayerType::ATENSETITEM, aten_set_item_layer_)
DECLARE_TORCH_OP_BUILDER(AtenSize, nn_compiler::ir::LayerType::ATENSIZE, aten_size_layer_)
DECLARE_TORCH_OP_BUILDER(AtenSlice, nn_compiler::ir::LayerType::ATENSLICE, aten_slice_layer_)
DECLARE_TORCH_OP_BUILDER(AtenSoftmax, nn_compiler::ir::LayerType::ATENSOFTMAX, aten_softmax_layer_)
DECLARE_TORCH_OP_BUILDER(AtenSqueeze, nn_compiler::ir::LayerType::ATENSQUEEZE, aten_squeeze_layer_)
DECLARE_TORCH_OP_BUILDER(AtenSub, nn_compiler::ir::LayerType::ATENSUB, aten_sub_layer_)
DECLARE_TORCH_OP_BUILDER(AtenSum, nn_compiler::ir::LayerType::ATENSUM, aten_sum_layer_)
DECLARE_TORCH_OP_BUILDER(AtenTanh, nn_compiler::ir::LayerType::ATENTANH, aten_tanh_layer_)
DECLARE_TORCH_OP_BUILDER(AtenTensor, nn_compiler::ir::LayerType::ATENTENSOR, aten_tensor_layer_)
DECLARE_TORCH_OP_BUILDER(AtenTo1, nn_compiler::ir::LayerType::ATENTO1, aten_to_layer_)
DECLARE_TORCH_OP_BUILDER(AtenTo2, nn_compiler::ir::LayerType::ATENTO2, aten_to_layer_)
DECLARE_TORCH_OP_BUILDER(AtenTo3, nn_compiler::ir::LayerType::ATENTO3, aten_to_layer_)
DECLARE_TORCH_OP_BUILDER(AtenTopk, nn_compiler::ir::LayerType::ATENTOPK, aten_topk_layer_)
DECLARE_TORCH_OP_BUILDER(AtenTranspose, nn_compiler::ir::LayerType::ATENTRANSPOSE, aten_transpose_layer_)
DECLARE_TORCH_OP_BUILDER(AtenTriu, nn_compiler::ir::LayerType::ATENTRIU, aten_triu_layer_)
DECLARE_TORCH_OP_BUILDER(AtenTypeAs, nn_compiler::ir::LayerType::ATENTYPEAS, aten_type_as_layer_)
DECLARE_TORCH_OP_BUILDER(AtenUnsqueeze, nn_compiler::ir::LayerType::ATENUNSQUEEZE, aten_unsqueeze_layer_)
DECLARE_TORCH_OP_BUILDER(AtenView, nn_compiler::ir::LayerType::ATENVIEW, aten_view_layer_)
DECLARE_TORCH_OP_BUILDER(AtenWarn, nn_compiler::ir::LayerType::ATENWARN, aten_warn_layer_)
DECLARE_TORCH_OP_BUILDER(AtenWhere, nn_compiler::ir::LayerType::ATENWHERE, aten_where_layer_)
DECLARE_TORCH_OP_BUILDER(AtenZeros, nn_compiler::ir::LayerType::ATENZEROS, aten_zeros_layer_)
DECLARE_TORCH_OP_BUILDER(AtenZerosLike, nn_compiler::ir::LayerType::ATENZEROSLIKE, aten_zeros_like_layer_)

DECLARE_TORCH_OP_BUILDER(PrimBlock, nn_compiler::ir::LayerType::PRIMBLOCK, prim_block_layer_)
DECLARE_TORCH_OP_BUILDER(PrimData, nn_compiler::ir::LayerType::PRIMDATA, prim_data_layer_)
DECLARE_TORCH_OP_BUILDER(PrimDevice, nn_compiler::ir::LayerType::PRIMDEVICE, prim_device_layer_)
DECLARE_TORCH_OP_BUILDER(PrimDtype, nn_compiler::ir::LayerType::PRIMDTYPE, prim_dtype_layer_)
DECLARE_TORCH_OP_BUILDER(PrimEndIf, nn_compiler::ir::LayerType::PRIMENDIF, prim_end_if_layer_)
DECLARE_TORCH_OP_BUILDER(PrimEndLoop, nn_compiler::ir::LayerType::PRIMENDLOOP, prim_end_loop_layer_)
DECLARE_TORCH_OP_BUILDER(PrimGetAttr, nn_compiler::ir::LayerType::PRIMGETATTR, prim_get_attr_layer_)
DECLARE_TORCH_OP_BUILDER(PrimIf, nn_compiler::ir::LayerType::PRIMIF, prim_if_layer_)
DECLARE_TORCH_OP_BUILDER(PrimInput, nn_compiler::ir::LayerType::PRIMINPUT, prim_input_layer_)
DECLARE_TORCH_OP_BUILDER(PrimListConstruct, nn_compiler::ir::LayerType::PRIMLISTCONSTRUCT, prim_list_construct_layer_)
DECLARE_TORCH_OP_BUILDER(PrimListUnpack, nn_compiler::ir::LayerType::PRIMLISTUNPACK, prim_list_unpack_layer_)
DECLARE_TORCH_OP_BUILDER(PrimLoop, nn_compiler::ir::LayerType::PRIMLOOP, prim_loop_layer_)
DECLARE_TORCH_OP_BUILDER(PrimLoopIndex, nn_compiler::ir::LayerType::PRIMLOOPINDEX, prim_loop_index_layer_)
DECLARE_TORCH_OP_BUILDER(PrimOutput, nn_compiler::ir::LayerType::PRIMOUTPUT, prim_output_layer_)
DECLARE_TORCH_OP_BUILDER(PrimRaiseException, nn_compiler::ir::LayerType::PRIMRAISEEXCEPTION,
                         prim_raise_exception_layer_)
DECLARE_TORCH_OP_BUILDER(PrimSetAttr, nn_compiler::ir::LayerType::PRIMSETATTR, prim_set_attr_layer_)
DECLARE_TORCH_OP_BUILDER(PrimToList, nn_compiler::ir::LayerType::PRIMTOLIST, prim_to_list_layer_)
DECLARE_TORCH_OP_BUILDER(PrimTupleConstruct, nn_compiler::ir::LayerType::PRIMTUPLECONSTRUCT,
                         prim_tuple_construct_layer_)
DECLARE_TORCH_OP_BUILDER(PrimTupleIndex, nn_compiler::ir::LayerType::PRIMTUPLEINDEX, prim_tuple_index_layer_)
DECLARE_TORCH_OP_BUILDER(PrimTupleUnpack, nn_compiler::ir::LayerType::PRIMTUPLEUNPACK, prim_tuple_unpack_layer_)
DECLARE_TORCH_OP_BUILDER(PrimType, nn_compiler::ir::LayerType::PRIMTYPE, prim_type_layer_)
DECLARE_TORCH_OP_BUILDER(PrimUncheckedCast, nn_compiler::ir::LayerType::PRIMUNCHECKEDCAST, prim_unchecked_cast_layer_)
DECLARE_TORCH_OP_BUILDER(PrimUninitialized, nn_compiler::ir::LayerType::PRIMUNINITIALIZED, prim_uninitialized_layer_)
DECLARE_TORCH_OP_BUILDER(PrimVariable, nn_compiler::ir::LayerType::PRIMVARIABLE, prim_variable_layer_)
