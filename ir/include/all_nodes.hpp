/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "ir/include/node.hpp"

#include "ir/include/global_node.hpp"
#include "ir/include/hw_node.hpp"
#include "ir/include/nn_node.hpp"
#include "ir/include/op_node.hpp"
#include "ir/include/q_node.hpp"
#include "ir/include/v_node.hpp"

#include "ir/include/nn_nodes/activation_node.hpp"
#include "ir/include/nn_nodes/aten_add_node.hpp"
#include "ir/include/nn_nodes/aten_addmm_node.hpp"
#include "ir/include/nn_nodes/aten_and_node.hpp"
#include "ir/include/nn_nodes/aten_any_node.hpp"
#include "ir/include/nn_nodes/aten_append_node.hpp"
#include "ir/include/nn_nodes/aten_arange_node.hpp"
#include "ir/include/nn_nodes/aten_as_tensor_node.hpp"
#include "ir/include/nn_nodes/aten_bitwise_not_node.hpp"
#include "ir/include/nn_nodes/aten_bmm_node.hpp"
#include "ir/include/nn_nodes/aten_bool_node.hpp"
#include "ir/include/nn_nodes/aten_cat_node.hpp"
#include "ir/include/nn_nodes/aten_ceil_node.hpp"
#include "ir/include/nn_nodes/aten_chunk_node.hpp"
#include "ir/include/nn_nodes/aten_clamp_node.hpp"
#include "ir/include/nn_nodes/aten_copy_node.hpp"
#include "ir/include/nn_nodes/aten_derive_index_node.hpp"
#include "ir/include/nn_nodes/aten_dim_node.hpp"
#include "ir/include/nn_nodes/aten_div_node.hpp"
#include "ir/include/nn_nodes/aten_dropout_node.hpp"
#include "ir/include/nn_nodes/aten_embedding_node.hpp"
#include "ir/include/nn_nodes/aten_eq_node.hpp"
#include "ir/include/nn_nodes/aten_expand_node.hpp"
#include "ir/include/nn_nodes/aten_format_node.hpp"
#include "ir/include/nn_nodes/aten_gather_node.hpp"
#include "ir/include/nn_nodes/aten_ge_node.hpp"
#include "ir/include/nn_nodes/aten_get_item_node.hpp"
#include "ir/include/nn_nodes/aten_gt_node.hpp"
#include "ir/include/nn_nodes/aten_index_node.hpp"
#include "ir/include/nn_nodes/aten_index_put_node.hpp"
#include "ir/include/nn_nodes/aten_index_select_node.hpp"
#include "ir/include/nn_nodes/aten_int_node.hpp"
#include "ir/include/nn_nodes/aten_is_node.hpp"
#include "ir/include/nn_nodes/aten_item_node.hpp"
#include "ir/include/nn_nodes/aten_leaky_relu_node.hpp"
#include "ir/include/nn_nodes/aten_len_node.hpp"
#include "ir/include/nn_nodes/aten_linear_node.hpp"
#include "ir/include/nn_nodes/aten_list_node.hpp"
#include "ir/include/nn_nodes/aten_log_node.hpp"
#include "ir/include/nn_nodes/aten_log_softmax_node.hpp"
#include "ir/include/nn_nodes/aten_lstm_node.hpp"
#include "ir/include/nn_nodes/aten_lt_node.hpp"
#include "ir/include/nn_nodes/aten_masked_fill_node.hpp"
#include "ir/include/nn_nodes/aten_masked_select_node.hpp"
#include "ir/include/nn_nodes/aten_matmul_node.hpp"
#include "ir/include/nn_nodes/aten_max_node.hpp"
#include "ir/include/nn_nodes/aten_max_pool2d_node.hpp"
#include "ir/include/nn_nodes/aten_min_node.hpp"
#include "ir/include/nn_nodes/aten_mul_node.hpp"
#include "ir/include/nn_nodes/aten_ne_node.hpp"
#include "ir/include/nn_nodes/aten_neg_node.hpp"
#include "ir/include/nn_nodes/aten_not_node.hpp"
#include "ir/include/nn_nodes/aten_ones_node.hpp"
#include "ir/include/nn_nodes/aten_pack_padded_sequence_node.hpp"
#include "ir/include/nn_nodes/aten_pad_packed_sequence_node.hpp"
#include "ir/include/nn_nodes/aten_pow_node.hpp"
#include "ir/include/nn_nodes/aten_relu_node.hpp"
#include "ir/include/nn_nodes/aten_select_node.hpp"
#include "ir/include/nn_nodes/aten_set_item_node.hpp"
#include "ir/include/nn_nodes/aten_size_node.hpp"
#include "ir/include/nn_nodes/aten_slice_node.hpp"
#include "ir/include/nn_nodes/aten_softmax_node.hpp"
#include "ir/include/nn_nodes/aten_squeeze_node.hpp"
#include "ir/include/nn_nodes/aten_sub_node.hpp"
#include "ir/include/nn_nodes/aten_sum_node.hpp"
#include "ir/include/nn_nodes/aten_tanh_node.hpp"
#include "ir/include/nn_nodes/aten_tensor_node.hpp"
#include "ir/include/nn_nodes/aten_to_node.hpp"
#include "ir/include/nn_nodes/aten_topk_node.hpp"
#include "ir/include/nn_nodes/aten_transpose_node.hpp"
#include "ir/include/nn_nodes/aten_unsqueeze_node.hpp"
#include "ir/include/nn_nodes/aten_view_node.hpp"
#include "ir/include/nn_nodes/aten_warn_node.hpp"
#include "ir/include/nn_nodes/aten_zeros_like_node.hpp"
#include "ir/include/nn_nodes/aten_zeros_node.hpp"
#include "ir/include/nn_nodes/batchnorm_node.hpp"
#include "ir/include/nn_nodes/concat_node.hpp"
#include "ir/include/nn_nodes/convolution_node.hpp"
#include "ir/include/nn_nodes/copy_node.hpp"
#include "ir/include/nn_nodes/data_format_node.hpp"
#include "ir/include/nn_nodes/deconvolution_node.hpp"
#include "ir/include/nn_nodes/depth_to_space_node.hpp"
#include "ir/include/nn_nodes/dummy_node.hpp"
#include "ir/include/nn_nodes/eltwise_node.hpp"
#include "ir/include/nn_nodes/fullyconnected_node.hpp"
#include "ir/include/nn_nodes/input_node.hpp"
#include "ir/include/nn_nodes/matmul_node.hpp"
#include "ir/include/nn_nodes/permute_node.hpp"
#include "ir/include/nn_nodes/pool_node.hpp"
#include "ir/include/nn_nodes/priorbox_node.hpp"
#include "ir/include/nn_nodes/reshape_node.hpp"
#include "ir/include/nn_nodes/scale_node.hpp"
#include "ir/include/nn_nodes/slice_node.hpp"
#include "ir/include/nn_nodes/softmax_node.hpp"
#include "ir/include/nn_nodes/space_to_depth_node.hpp"
#include "ir/include/nn_nodes/tile_node.hpp"

#include "ir/include/hw_nodes/maa_eltwise_node.hpp"

#include "ir/include/op_nodes/shift_node.hpp"

#include "ir/include/q_nodes/dequant_node.hpp"
#include "ir/include/q_nodes/quant_node.hpp"

#include "ir/include/global_nodes/global_concat_node.hpp"
#include "ir/include/global_nodes/global_split_node.hpp"
#include "ir/include/global_nodes/global_sync_node.hpp"

#include "ir/include/v_nodes/vconcat_node.hpp"
#include "ir/include/v_nodes/vsplit_node.hpp"

#include "ir/include/control_nodes/prim_block_node.hpp"
#include "ir/include/control_nodes/prim_constant_node.hpp"
#include "ir/include/control_nodes/prim_data_node.hpp"
#include "ir/include/control_nodes/prim_device_node.hpp"
#include "ir/include/control_nodes/prim_dtype_node.hpp"
#include "ir/include/control_nodes/prim_end_if_node.hpp"
#include "ir/include/control_nodes/prim_end_loop_node.hpp"
#include "ir/include/control_nodes/prim_if_node.hpp"
#include "ir/include/control_nodes/prim_input_node.hpp"
#include "ir/include/control_nodes/prim_list_construct_node.hpp"
#include "ir/include/control_nodes/prim_list_unpack_node.hpp"
#include "ir/include/control_nodes/prim_loop_index_node.hpp"
#include "ir/include/control_nodes/prim_loop_node.hpp"
#include "ir/include/control_nodes/prim_output_node.hpp"
#include "ir/include/control_nodes/prim_raise_exception_node.hpp"
#include "ir/include/control_nodes/prim_tuple_construct_node.hpp"
#include "ir/include/control_nodes/prim_tuple_index_node.hpp"
#include "ir/include/control_nodes/prim_tuple_unpack_node.hpp"
#include "ir/include/control_nodes/prim_unchecked_cast_node.hpp"
#include "ir/include/control_nodes/prim_uninitialized_node.hpp"
#include "ir/include/control_nodes/prim_variable_node.hpp"
