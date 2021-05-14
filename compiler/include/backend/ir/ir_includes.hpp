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

#include "ir/blob.hpp"
#include "ir/compute_instr.hpp"
#include "ir/control_edge.hpp"
#include "ir/data_blob.hpp"
#include "ir/data_edge.hpp"
#include "ir/edge.hpp"
#include "ir/edge_execution_step.hpp"
#include "ir/execution_step.hpp"
#include "ir/featuremap_blob.hpp"
#include "ir/global_node.hpp"
#include "ir/instruction.hpp"
#include "ir/ir_types.hpp"
#include "ir/kernel_node_parameters.hpp"
#include "ir/memory_instr.hpp"
#include "ir/misc_instr.hpp"

#include "ir/all_nodes.hpp" // NOLINT
#include "ir/nn_node_type_traits.hpp"
#include "ir/node_execution_step.hpp"

#include "ir/instructions/dma_start_instr.hpp"
#include "ir/instructions/dma_sync_instr.hpp"
#include "ir/instructions/execute_start_instr.hpp"
#include "ir/instructions/execute_sync_instr.hpp"
#include "ir/instructions/signal_send_instr.hpp"
#include "ir/instructions/signal_wait_instr.hpp"
#include "ir/instructions/vsync_instr.hpp"
