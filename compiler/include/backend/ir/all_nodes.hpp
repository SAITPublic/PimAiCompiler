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

#include "ir/node.hpp"

#include "ir/global_node.hpp"
#include "ir/hw_node.hpp"
#include "ir/nn_node.hpp"
#include "ir/op_node.hpp"
#include "ir/q_node.hpp"
#include "ir/v_node.hpp"

#include "ir/nn_nodes/activation_node.hpp"
#include "ir/nn_nodes/batchnorm_node.hpp"
#include "ir/nn_nodes/concat_node.hpp"
#include "ir/nn_nodes/convolution_node.hpp"
#include "ir/nn_nodes/copy_node.hpp"
#include "ir/nn_nodes/data_format_node.hpp"
#include "ir/nn_nodes/deconvolution_node.hpp"
#include "ir/nn_nodes/depth_to_space_node.hpp"
#include "ir/nn_nodes/dummy_node.hpp"
#include "ir/nn_nodes/eltwise_node.hpp"
#include "ir/nn_nodes/fullyconnected_node.hpp"
#include "ir/nn_nodes/input_node.hpp"
#include "ir/nn_nodes/matmul_node.hpp"
#include "ir/nn_nodes/permute_node.hpp"
#include "ir/nn_nodes/pool_node.hpp"
#include "ir/nn_nodes/priorbox_node.hpp"
#include "ir/nn_nodes/reshape_node.hpp"
#include "ir/nn_nodes/scale_node.hpp"
#include "ir/nn_nodes/slice_node.hpp"
#include "ir/nn_nodes/softmax_node.hpp"
#include "ir/nn_nodes/space_to_depth_node.hpp"
#include "ir/nn_nodes/tile_node.hpp"

#include "ir/hw_nodes/maa_eltwise_node.hpp"

#include "ir/op_nodes/shift_node.hpp"

#include "ir/q_nodes/dequant_node.hpp"
#include "ir/q_nodes/quant_node.hpp"

#include "ir/global_nodes/global_concat_node.hpp"
#include "ir/global_nodes/global_split_node.hpp"
#include "ir/global_nodes/global_sync_node.hpp"

#include "ir/v_nodes/vconcat_node.hpp"
#include "ir/v_nodes/vsplit_node.hpp"
