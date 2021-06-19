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