#pragma once

#include "ir/include/layers/pim_general_layers.h"

#include "ir/include/layers/aten_add_layer.h"
#include "ir/include/layers/aten_addmm_layer.h"
#include "ir/include/layers/aten_arange1_layer.h"
#include "ir/include/layers/aten_arange2_layer.h"
#include "ir/include/layers/aten_arange3_layer.h"
#include "ir/include/layers/aten_as_tensor_layer.h"
#include "ir/include/layers/aten_batch_norm2d_layer.h"
#include "ir/include/layers/aten_cat_layer.h"
#include "ir/include/layers/aten_chunk_layer.h"
#include "ir/include/layers/aten_clamp_layer.h"
#include "ir/include/layers/aten_clone_layer.h"
#include "ir/include/layers/aten_contiguous_layer.h"
#include "ir/include/layers/aten_conv2d_layer.h"
#include "ir/include/layers/aten_copy_layer.h"
#include "ir/include/layers/aten_derive_index_layer.h"
#include "ir/include/layers/aten_dropout_layer.h"
#include "ir/include/layers/aten_embedding_layer.h"
#include "ir/include/layers/aten_expand_layer.h"
#include "ir/include/layers/aten_format_layer.h"
#include "ir/include/layers/aten_gather_layer.h"
#include "ir/include/layers/aten_get_item_layer.h"
#include "ir/include/layers/aten_index_put_layer.h"
#include "ir/include/layers/aten_index_select_layer.h"
#include "ir/include/layers/aten_layer_norm_layer.h"
#include "ir/include/layers/aten_leaky_relu_layer.h"
#include "ir/include/layers/aten_linear_layer.h"
#include "ir/include/layers/aten_log_softmax_layer.h"
#include "ir/include/layers/aten_lstm1_layer.h"
#include "ir/include/layers/aten_lstm2_layer.h"
#include "ir/include/layers/aten_masked_fill_layer.h"
#include "ir/include/layers/aten_max_layer.h"
#include "ir/include/layers/aten_max_pool2d_layer.h"
#include "ir/include/layers/aten_min_layer.h"
#include "ir/include/layers/aten_norm_layer.h"
#include "ir/include/layers/aten_ones_layer.h"
#include "ir/include/layers/aten_pack_padded_sequence_layer.h"
#include "ir/include/layers/aten_pad_packed_sequence_layer.h"
#include "ir/include/layers/aten_select_layer.h"
#include "ir/include/layers/aten_set_item_layer.h"
#include "ir/include/layers/aten_size_layer.h"
#include "ir/include/layers/aten_slice_layer.h"
#include "ir/include/layers/aten_softmax_layer.h"
#include "ir/include/layers/aten_squeeze_layer.h"
#include "ir/include/layers/aten_sub_layer.h"
#include "ir/include/layers/aten_sum_layer.h"
#include "ir/include/layers/aten_to1_layer.h"
#include "ir/include/layers/aten_to2_layer.h"
#include "ir/include/layers/aten_topk_layer.h"
#include "ir/include/layers/aten_transpose_layer.h"
#include "ir/include/layers/aten_unsqueeze_layer.h"
#include "ir/include/layers/aten_warn_layer.h"

#include "ir/include/layers/prim_callmethod_layer.h"
#include "ir/include/layers/prim_constant_layer.h"
#include "ir/include/layers/prim_end_if_layer.h"
#include "ir/include/layers/prim_end_loop_layer.h"
#include "ir/include/layers/prim_get_attr_layer.h"
#include "ir/include/layers/prim_if_layer.h"
#include "ir/include/layers/prim_loop_index_layer.h"
#include "ir/include/layers/prim_loop_layer.h"
#include "ir/include/layers/prim_tuple_index_layer.h"
#include "ir/include/layers/prim_variable_layer.h"
