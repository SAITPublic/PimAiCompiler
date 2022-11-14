/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#pragma once

#include <torch/script.h>

#include "frontend/importer/utils/attr_parser.h"
#include "ir/include/layers/all_layers.h"
#include "ir/include/nn_model.h"

namespace nn_compiler
{
namespace frontend
{
class LayerBuilder
{
   public:
    LayerBuilder() { attr_parser_ = std::make_shared<AttrParser>(); }

    virtual std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref) = 0;

    std::shared_ptr<AttrParser> parser() { return attr_parser_; }

    virtual ~LayerBuilder() = default;

   private:
    std::shared_ptr<AttrParser> attr_parser_ = nullptr;
};

class AtenAbsBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAbsLayer> aten_abs_layer_;
};

class AtenAddBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAddLayer> aten_add_layer_;
};

class AtenAddmmBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAddmmLayer> aten_addmm_layer_;
};

class AtenAndBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAndLayer> aten_and_layer_;
};

class AtenAnyBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAnyLayer> aten_any_layer_;
};

class AtenAppendBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAppendLayer> aten_append_layer_;
};

class AtenArange1Builder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenArange1Layer> aten_arange_layer_;
};

class AtenArange2Builder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenArange2Layer> aten_arange_layer_;
};

class AtenArange3Builder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenArange3Layer> aten_arange_layer_;
};

class AtenArgmaxBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenArgmaxLayer> aten_argmax_layer_;
};

class AtenAsTensorBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAsTensorLayer> aten_as_tensor_layer_;
};

class AtenBatchNorm2dBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenBatchNorm2dLayer> aten_batch_norm2d_layer_;
};

class AtenBitwiseNotBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenBitwiseNotLayer> aten_bitwise_not_layer_;
};

class AtenBmmBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenBmmLayer> aten_bmm_layer_;
};

class AtenBoolBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenBoolLayer> aten_bool_layer_;
};

class AtenCatBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenCatLayer> aten_cat_layer_;
};

class AtenCeilBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenCeilLayer> aten_ceil_layer_;
};

class AtenChunkBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenChunkLayer> aten_chunk_layer_;
};

class AtenClampBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenClampLayer> aten_clamp_layer_;
};

class AtenClearBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenClearLayer> aten_clear_layer_;
};

class AtenCloneBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenCloneLayer> aten_clone_layer_;
};

class AtenContiguousBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenContiguousLayer> aten_contiguous_layer_;
};

class AtenConv2dBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenConv2dLayer> aten_conv2d_layer_;
};

class AtenCopyBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenCopyLayer> aten_copy_layer_;
};

class AtenCpuBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenCpuLayer> aten_cpu_layer_;
};

class AtenCudaBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenCudaLayer> aten_cuda_layer_;
};

class AtenDeriveIndexBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenDeriveIndexLayer> aten_derive_index_layer_;
};

class AtenCumsumBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenCumsumLayer> aten_cumsum_layer_;
};

class AtenDetachBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenDetachLayer> aten_detach_layer_;
};

class AtenDimBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenDimLayer> aten_dim_layer_;
};

class AtenDivBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenDivLayer> aten_div_layer_;
};

class AtenDropoutBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenDropoutLayer> aten_dropout_layer_;
};

class AtenEmbeddingBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenEmbeddingLayer> aten_embedding_layer_;
};

class AtenEinsumBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenEinsumLayer> aten_einsum_layer_;
};

class AtenEqBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenEqLayer> aten_eq_layer_;
};

class AtenEqualBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenEqualLayer> aten_equal_layer_;
};

class AtenExpandBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenExpandLayer> aten_expand_layer_;
};

class AtenFillBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenFillLayer> aten_fill_layer_;
};

class AtenFloorDivideBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenFloorDivideLayer> aten_floor_divide_layer_;
};

class AtenFormatBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenFormatLayer> aten_format_layer_;
};

class AtenFullLikeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenFullLikeLayer> aten_full_like_layer_;
};

class AtenGatherBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenGatherLayer> aten_gather_layer_;
};

class AtenGeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenGeLayer> aten_ge_layer_;
};

class AtenGtBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenGtLayer> aten_gt_layer_;
};

class AtenIndexBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIndexLayer> aten_index_layer_;
};

class AtenGetItemBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenGetItemLayer> aten_get_item_layer_;
};

class AtenIndexPutBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIndexPutLayer> aten_index_put_layer_;
};

class AtenIndexSelectBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIndexSelectLayer> aten_index_select_layer_;
};

class AtenIntBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIntLayer> aten_int_layer_;
};

class AtenIntImplicitBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIntImplicitLayer> aten_int_implicit_layer_;
};

class AtenIsBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIsLayer> aten_is_layer_;
};

class AtenIsInfBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIsInfLayer> aten_is_inf_layer_;
};

class AtenIsNotBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIsNotLayer> aten_is_not_layer_;
};

class AtenItemBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenItemLayer> aten_item_layer_;
};

class AtenLayerNormBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLayerNormLayer> aten_layer_norm_layer_;
};

class AtenLeakyReluBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLeakyReluLayer> aten_leaky_relu_layer_;
};

class AtenLeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLeLayer> aten_le_layer_;
};

class AtenLenBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLenLayer> aten_len_layer_;
};

class AtenLinearBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLinearLayer> aten_linear_layer_;
};

class AtenListBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenListLayer> aten_list_layer_;
};

class AtenLogBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLogLayer> aten_log_layer_;
};

class AtenLogSoftmaxBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLogSoftmaxLayer> aten_log_softmax_layer_;
};

class AtenLSTM1Builder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLSTM1Layer> aten_lstm1_layer_;
};

class AtenLSTM2Builder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLSTM2Layer> aten_lstm2_layer_;
};

class AtenLtBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLtLayer> aten_lt_layer_;
};

class AtenMaskedFillBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenMaskedFillLayer> aten_masked_fill_layer_;
};

class AtenMaskedSelectBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenMaskedSelectLayer> aten_masked_select_layer_;
};

class AtenMatmulBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenMatmulLayer> aten_matmul_layer_;
};

class AtenMaxBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenMaxLayer> aten_max_layer_;
};

class AtenMaxPool2dBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenMaxPool2dLayer> aten_max_pool2d_layer_;
};

class AtenMeanBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenMeanLayer> aten_mean_layer_;
};

class AtenMinBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenMinLayer> aten_min_layer_;
};

class AtenMulBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenMulLayer> aten_mul_layer_;
};

class AtenNeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenNeLayer> aten_ne_layer_;
};

class AtenNegBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenNegLayer> aten_neg_layer_;
};

class AtenNormBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenNormLayer> aten_norm_layer_;
};

class AtenNotBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenNotLayer> aten_not_layer_;
};

class AtenOneHotBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenOneHotLayer> aten_one_hot_layer_;
};

class AtenOnesBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenOnesLayer> aten_ones_layer_;
};

class AtenPackPaddedSequenceBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenPackPaddedSequenceLayer> aten_pack_padded_sequence_layer_;
};

class AtenPadPackedSequenceBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenPadPackedSequenceLayer> aten_pad_packed_sequence_layer_;
};

class AtenPermuteBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenPermuteLayer> aten_permute_layer_;
};

class AtenPowBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenPowLayer> aten_pow_layer_;
};

class AtenReluBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenReluLayer> aten_relu_layer_;
};

class AtenReshapeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenReshapeLayer> aten_reshape_layer_;
};

class AtenRemainderBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenRemainderLayer> aten_remainder_layer_;
};

class AtenRepeatBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenRepeatLayer> aten_repeat_layer_;
};

class AtenRsqrtBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenRsqrtLayer> aten_rsqrt_layer_;
};

class AtenSelectBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenSelectLayer> aten_select_layer_;
};

class AtenSqueezeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenSqueezeLayer> aten_squeeze_layer_;
};

class AtenSetItemBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenSetItemLayer> aten_set_item_layer_;
};

class AtenSizeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenSizeLayer> aten_size_layer_;
};

class AtenSliceBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenSliceLayer> aten_slice_layer_;
};

class AtenSoftmaxBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenSoftmaxLayer> aten_softmax_layer_;
};

class AtenSubBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenSubLayer> aten_sub_layer_;
};

class AtenSumBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenSumLayer> aten_sum_layer_;
};

class AtenTanhBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenTanhLayer> aten_tanh_layer_;
};

class AtenTriuBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenTriuLayer> aten_triu_layer_;
};

class AtenTensorBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenTensorLayer> aten_tensor_layer_;
};

class AtenTypeAsBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenTypeAsLayer> aten_type_as_layer_;
};

class AtenTo1Builder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenTo1Layer> aten_to_layer_;
};

class AtenTo2Builder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenTo2Layer> aten_to_layer_;
};

class AtenTo3Builder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenTo3Layer> aten_to_layer_;
};

class AtenTopkBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenTopkLayer> aten_topk_layer_;
};

class AtenTransposeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenTransposeLayer> aten_transpose_layer_;
};

class AtenUnsqueezeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenUnsqueezeLayer> aten_unsqueeze_layer_;
};

class AtenViewBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenViewLayer> aten_view_layer_;
};

class AtenWarnBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenWarnLayer> aten_warn_layer_;
};

class AtenWhereBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenWhereLayer> aten_where_layer_;
};

class AtenZerosBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenZerosLayer> aten_zeros_layer_;
};

class AtenZerosLikeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenZerosLikeLayer> aten_zeros_like_layer_;
};

class PrimBlockBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimBlockLayer> prim_block_layer_;
};

class PrimCallMethodBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

    std::shared_ptr<ir::NNLayer> buildLayerCustom(const std::string target_network_name);

   private:
    std::shared_ptr<ir::PrimCallMethodLayer> prim_callmethod_layer_;
};

class PrimConstantBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimConstantLayer> prim_constant_layer_;
};

class PrimDataBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimDataLayer> prim_data_layer_;
};

class PrimDeviceBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimDeviceLayer> prim_device_layer_;
};

class PrimDtypeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimDtypeLayer> prim_dtype_layer_;
};

class PrimEndIfBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimEndIfLayer> prim_end_if_layer_;
};

class PrimEndLoopBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimEndLoopLayer> prim_end_loop_layer_;
};

class PrimGetAttrBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimGetAttrLayer> prim_get_attr_layer_;
};

class PrimIfBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimIfLayer> prim_if_layer_;
};

class PrimInputBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimInputLayer> prim_input_layer_;
};

class PrimListConstructBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimListConstructLayer> prim_list_construct_layer_;
};

class PrimListUnpackBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimListUnpackLayer> prim_list_unpack_layer_;
};

class PrimLoopBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimLoopLayer> prim_loop_layer_;
};

class PrimLoopIndexBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimLoopIndexLayer> prim_loop_index_layer_;
};

class PrimOutputBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimOutputLayer> prim_output_layer_;
};

class PrimRaiseExceptionBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimRaiseExceptionLayer> prim_raise_exception_layer_;
};

class PrimSetAttrBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimSetAttrLayer> prim_set_attr_layer_;
};

class PrimToListBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimToListLayer> prim_to_list_layer_;
};

class PrimTupleConstructBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimTupleConstructLayer> prim_tuple_construct_layer_;
};

class PrimTupleIndexBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimTupleIndexLayer> prim_tuple_index_layer_;
};

class PrimTupleUnpackBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimTupleUnpackLayer> prim_tuple_unpack_layer_;
};

class PrimTypeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimTypeLayer> prim_type_layer_;
};

class PrimUncheckedCastBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimUncheckedCastLayer> prim_unchecked_cast_layer_;
};

class PrimUninitializedBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimUninitializedLayer> prim_uninitialized_layer_;
};

class PrimVariableBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::PrimVariableLayer> prim_variable_layer_;
};

class LayerBuilders
{
   public:
    LayerBuilders()
    {
        // Register layer builder.
        layer_builders_["aten::abs"] = std::make_shared<AtenAbsBuilder>();
        layer_builders_["aten::add"] = std::make_shared<AtenAddBuilder>();
        layer_builders_["aten::add_"] = std::make_shared<AtenAddBuilder>();
        layer_builders_["aten::addmm"] = std::make_shared<AtenAddmmBuilder>();
        layer_builders_["aten::__and__"] = std::make_shared<AtenAndBuilder>();
        layer_builders_["aten::any"] = std::make_shared<AtenAnyBuilder>();
        layer_builders_["aten::append"] = std::make_shared<AtenAppendBuilder>();
        layer_builders_["aten::arange1"] = std::make_shared<AtenArange1Builder>();
        layer_builders_["aten::arange2"] = std::make_shared<AtenArange2Builder>();
        layer_builders_["aten::arange3"] = std::make_shared<AtenArange3Builder>();
        layer_builders_["aten::argmax"] = std::make_shared<AtenArgmaxBuilder>();
        layer_builders_["aten::as_tensor"] = std::make_shared<AtenAsTensorBuilder>();
        layer_builders_["aten::batch_norm"] = std::make_shared<AtenBatchNorm2dBuilder>();
        layer_builders_["aten::bitwise_not"] = std::make_shared<AtenBitwiseNotBuilder>();
        layer_builders_["aten::bmm"] = std::make_shared<AtenBmmBuilder>();
        layer_builders_["aten::Bool"] = std::make_shared<AtenBoolBuilder>();
        layer_builders_["aten::cat"] = std::make_shared<AtenCatBuilder>();
        layer_builders_["aten::ceil"] = std::make_shared<AtenCeilBuilder>();
        layer_builders_["aten::chunk"] = std::make_shared<AtenChunkBuilder>();
        layer_builders_["aten::clamp"] = std::make_shared<AtenClampBuilder>();
        layer_builders_["aten::clear"] = std::make_shared<AtenClearBuilder>();
        layer_builders_["aten::clone"] = std::make_shared<AtenCloneBuilder>();
        layer_builders_["aten::contiguous"] = std::make_shared<AtenContiguousBuilder>();
        layer_builders_["aten::conv2d"] = std::make_shared<AtenConv2dBuilder>();
        layer_builders_["aten::copy_"] = std::make_shared<AtenCopyBuilder>();
        layer_builders_["aten::cpu"] = std::make_shared<AtenCpuBuilder>();
        layer_builders_["aten::cuda"] = std::make_shared<AtenCudaBuilder>();
        layer_builders_["aten::cumsum"] = std::make_shared<AtenCumsumBuilder>();
        layer_builders_["aten::detach"] = std::make_shared<AtenDetachBuilder>();
        layer_builders_["aten::dim"] = std::make_shared<AtenDimBuilder>();
        layer_builders_["aten::div"] = std::make_shared<AtenDivBuilder>();
        layer_builders_["aten::div_"] = std::make_shared<AtenDivBuilder>();
        layer_builders_["aten::dropout"] = std::make_shared<AtenDropoutBuilder>();
        layer_builders_["aten::dropout_"] = std::make_shared<AtenDropoutBuilder>();
        layer_builders_["aten::embedding"] = std::make_shared<AtenEmbeddingBuilder>();
        layer_builders_["aten::einsum"] = std::make_shared<AtenEinsumBuilder>();
        layer_builders_["aten::eq"] = std::make_shared<AtenEqBuilder>();
        layer_builders_["aten::equal"] = std::make_shared<AtenEqualBuilder>();
        layer_builders_["aten::expand"] = std::make_shared<AtenExpandBuilder>();
        layer_builders_["aten::fill_"] = std::make_shared<AtenFillBuilder>();
        layer_builders_["aten::floor_divide"] = std::make_shared<AtenFloorDivideBuilder>();
        layer_builders_["aten::format"] = std::make_shared<AtenFormatBuilder>();
        layer_builders_["aten::full_like"] = std::make_shared<AtenFullLikeBuilder>();
        layer_builders_["aten::gather"] = std::make_shared<AtenGatherBuilder>();
        layer_builders_["aten::ge"] = std::make_shared<AtenGeBuilder>();
        layer_builders_["aten::__getitem__"] = std::make_shared<AtenGetItemBuilder>();
        layer_builders_["aten::gt"] = std::make_shared<AtenGtBuilder>();
        layer_builders_["aten::index"] = std::make_shared<AtenIndexBuilder>();
        layer_builders_["aten::index_put_"] = std::make_shared<AtenIndexPutBuilder>();
        layer_builders_["aten::index_select"] = std::make_shared<AtenIndexSelectBuilder>();
        layer_builders_["aten::Int"] = std::make_shared<AtenIntBuilder>();
        layer_builders_["aten::IntImplicit"] = std::make_shared<AtenIntImplicitBuilder>();
        layer_builders_["aten::isinf"] = std::make_shared<AtenIsInfBuilder>();
        layer_builders_["aten::item"] = std::make_shared<AtenItemBuilder>();
        layer_builders_["aten::layer_norm"] = std::make_shared<AtenLayerNormBuilder>();
        layer_builders_["aten::leaky_relu"] = std::make_shared<AtenLeakyReluBuilder>();
        layer_builders_["aten::le"] = std::make_shared<AtenLeBuilder>();
        layer_builders_["aten::len"] = std::make_shared<AtenLenBuilder>();
        layer_builders_["aten::linear"] = std::make_shared<AtenLinearBuilder>();
        layer_builders_["aten::list"] = std::make_shared<AtenListBuilder>();
        layer_builders_["aten::log"] = std::make_shared<AtenLogBuilder>();
        layer_builders_["aten::log_softmax"] = std::make_shared<AtenLogSoftmaxBuilder>();
        layer_builders_["aten::lstm1"] = std::make_shared<AtenLSTM1Builder>();
        layer_builders_["aten::lstm2"] = std::make_shared<AtenLSTM2Builder>();
        layer_builders_["aten::lt"] = std::make_shared<AtenLtBuilder>();
        layer_builders_["aten::masked_fill"] = std::make_shared<AtenMaskedFillBuilder>();
        layer_builders_["aten::masked_fill_"] = std::make_shared<AtenMaskedFillBuilder>();
        layer_builders_["aten::masked_select"] = std::make_shared<AtenMaskedSelectBuilder>();
        layer_builders_["aten::matmul"] = std::make_shared<AtenMatmulBuilder>();
        layer_builders_["aten::max"] = std::make_shared<AtenMaxBuilder>();
        layer_builders_["aten::max_pool2d"] = std::make_shared<AtenMaxPool2dBuilder>();
        layer_builders_["aten::mean"] = std::make_shared<AtenMeanBuilder>();
        layer_builders_["aten::min"] = std::make_shared<AtenMinBuilder>();
        layer_builders_["aten::mul"] = std::make_shared<AtenMulBuilder>();
        layer_builders_["aten::ne"] = std::make_shared<AtenNeBuilder>();
        layer_builders_["aten::neg"] = std::make_shared<AtenNegBuilder>();
        layer_builders_["aten::norm"] = std::make_shared<AtenNormBuilder>();
        layer_builders_["aten::one_hot"] = std::make_shared<AtenOneHotBuilder>();
        layer_builders_["aten::ones"] = std::make_shared<AtenOnesBuilder>();

        layer_builders_["aten::_pack_padded_sequence"] = std::make_shared<AtenPackPaddedSequenceBuilder>();
        layer_builders_["aten::_pad_packed_sequence"] = std::make_shared<AtenPadPackedSequenceBuilder>();

        layer_builders_["aten::permute"] = std::make_shared<AtenPermuteBuilder>();
        layer_builders_["aten::pow"] = std::make_shared<AtenPowBuilder>();
        layer_builders_["aten::relu"] = std::make_shared<AtenReluBuilder>();
        layer_builders_["aten::reshape"] = std::make_shared<AtenReshapeBuilder>();
        layer_builders_["aten::remainder"] = std::make_shared<AtenRemainderBuilder>();
        layer_builders_["aten::repeat"] = std::make_shared<AtenRepeatBuilder>();
        layer_builders_["aten::rsqrt"] = std::make_shared<AtenRsqrtBuilder>();
        layer_builders_["aten::select"] = std::make_shared<AtenSelectBuilder>();
        layer_builders_["aten::squeeze"] = std::make_shared<AtenSqueezeBuilder>();
        layer_builders_["aten::_set_item"] = std::make_shared<AtenSetItemBuilder>();
        layer_builders_["aten::size"] = std::make_shared<AtenSizeBuilder>();
        layer_builders_["aten::slice"] = std::make_shared<AtenSliceBuilder>();
        layer_builders_["aten::softmax"] = std::make_shared<AtenSoftmaxBuilder>();
        layer_builders_["aten::sub"] = std::make_shared<AtenSubBuilder>();
        layer_builders_["aten::sum"] = std::make_shared<AtenSumBuilder>();
        layer_builders_["aten::tanh"] = std::make_shared<AtenTanhBuilder>();
        layer_builders_["aten::tensor"] = std::make_shared<AtenTensorBuilder>();
        layer_builders_["aten::to1"] = std::make_shared<AtenTo1Builder>();
        layer_builders_["aten::to2"] = std::make_shared<AtenTo2Builder>();
        layer_builders_["aten::to3"] = std::make_shared<AtenTo3Builder>();
        layer_builders_["aten::topk"] = std::make_shared<AtenTopkBuilder>();
        layer_builders_["aten::transpose"] = std::make_shared<AtenTransposeBuilder>();
        layer_builders_["aten::triu"] = std::make_shared<AtenTriuBuilder>();
        layer_builders_["aten::type_as"] = std::make_shared<AtenTypeAsBuilder>();
        layer_builders_["aten::unsqueeze"] = std::make_shared<AtenUnsqueezeBuilder>();
        layer_builders_["aten::unsqueeze_"] = std::make_shared<AtenUnsqueezeBuilder>();
        layer_builders_["aten::view"] = std::make_shared<AtenViewBuilder>();
        layer_builders_["aten::warn"] = std::make_shared<AtenWarnBuilder>();
        layer_builders_["aten::where"] = std::make_shared<AtenWhereBuilder>();
        layer_builders_["aten::zeros"] = std::make_shared<AtenZerosBuilder>();
        layer_builders_["aten::zeros_like"] = std::make_shared<AtenZerosLikeBuilder>();
        layer_builders_["aten::__derive_index"] = std::make_shared<AtenDeriveIndexBuilder>();
        layer_builders_["aten::__is__"] = std::make_shared<AtenIsBuilder>();
        layer_builders_["aten::__isnot__"] = std::make_shared<AtenIsNotBuilder>();
        layer_builders_["aten::__not__"] = std::make_shared<AtenNotBuilder>();

        layer_builders_["prim::Block"] = std::make_shared<PrimBlockBuilder>();
        layer_builders_["prim::CallMethod"] = std::make_shared<PrimCallMethodBuilder>();
        layer_builders_["prim::Constant"] = std::make_shared<PrimConstantBuilder>();
        layer_builders_["prim::data"] = std::make_shared<PrimDataBuilder>();
        layer_builders_["prim::device"] = std::make_shared<PrimDeviceBuilder>();
        layer_builders_["prim::dtype"] = std::make_shared<PrimDtypeBuilder>();
        layer_builders_["prim::EndIf"] = std::make_shared<PrimEndIfBuilder>();
        layer_builders_["prim::EndLoop"] = std::make_shared<PrimEndLoopBuilder>();
        layer_builders_["prim::GetAttr"] = std::make_shared<PrimGetAttrBuilder>();
        layer_builders_["prim::If"] = std::make_shared<PrimIfBuilder>();
        layer_builders_["prim::Input"] = std::make_shared<PrimInputBuilder>();
        layer_builders_["prim::ListConstruct"] = std::make_shared<PrimListConstructBuilder>();
        layer_builders_["prim::ListUnpack"] = std::make_shared<PrimListUnpackBuilder>();
        layer_builders_["prim::Loop"] = std::make_shared<PrimLoopBuilder>();
        layer_builders_["prim::LoopIndex"] = std::make_shared<PrimLoopIndexBuilder>();
        layer_builders_["prim::Output"] = std::make_shared<PrimOutputBuilder>();
        layer_builders_["prim::RaiseException"] = std::make_shared<PrimRaiseExceptionBuilder>();
        layer_builders_["prim::SetAttr"] = std::make_shared<PrimSetAttrBuilder>();
        layer_builders_["prim::tolist"] = std::make_shared<PrimToListBuilder>();
        layer_builders_["prim::TupleConstruct"] = std::make_shared<PrimTupleConstructBuilder>();
        layer_builders_["prim::TupleIndex"] = std::make_shared<PrimTupleIndexBuilder>();
        layer_builders_["prim::TupleUnpack"] = std::make_shared<PrimTupleUnpackBuilder>();
        layer_builders_["prim::type"] = std::make_shared<PrimTypeBuilder>();
        layer_builders_["prim::unchecked_cast"] = std::make_shared<PrimUncheckedCastBuilder>();
        layer_builders_["prim::Uninitialized"] = std::make_shared<PrimUninitializedBuilder>();
        layer_builders_["prim::Variable"] = std::make_shared<PrimVariableBuilder>();
    }

    std::shared_ptr<LayerBuilder> get(std::string layer_type)
    {
        if (layer_builders_[layer_type] == nullptr) {
            DLOG(INFO) << "layer type " << layer_type << " is unsupport.";
        }
        return layer_builders_[layer_type];
    }

    ~LayerBuilders() { this->layer_builders_.clear(); }

   private:
    std::map<std::string, std::shared_ptr<LayerBuilder>> layer_builders_;
};

}  // namespace frontend
}  // namespace nn_compiler
