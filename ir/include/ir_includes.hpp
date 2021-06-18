/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file.    ir_includes.hpp
 * @brief.
 * @details. This header includes necessary header files for ir
 * @version. 0.1.
 */

#pragma once

#include "ir/include/blob.hpp"
#include "ir/include/compute_instr.hpp"
#include "ir/include/control_edge.hpp"
#include "ir/include/data_blob.hpp"
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/edge_execution_step.hpp"
#include "ir/include/execution_step.hpp"
#include "ir/include/featuremap_blob.hpp"
#include "ir/include/global_node.hpp"
#include "ir/include/instruction.hpp"
#include "ir/include/ir_types.hpp"
#include "ir/include/kernel_node_parameters.hpp"
#include "ir/include/memory_instr.hpp"
#include "ir/include/misc_instr.hpp"

#include "ir/include/all_nodes.hpp" // NOLINT
#include "ir/include/nn_node_type_traits.hpp"
#include "ir/include/node_execution_step.hpp"

#include "ir/include/instructions/dma_start_instr.hpp"
#include "ir/include/instructions/dma_sync_instr.hpp"
#include "ir/include/instructions/execute_start_instr.hpp"
#include "ir/include/instructions/execute_sync_instr.hpp"
#include "ir/include/instructions/signal_send_instr.hpp"
#include "ir/include/instructions/signal_wait_instr.hpp"
#include "ir/include/instructions/vsync_instr.hpp"
