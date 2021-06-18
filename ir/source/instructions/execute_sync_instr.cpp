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
 * @file    execute_sync_instr.cpp
 * @brief   This is ExecuteSyncInstruction class
 * @details This source defines ExecuteSyncInstruction class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */

#include "ir/instructions/execute_sync_instr.hpp"
#include "ir/common/log.hpp"

namespace nn_compiler {
namespace nn_ir {

void ExecuteSyncInstruction::dump(std::ostream& s) const {
    ComputeInstruction::dump(s);
    s << ", " << start_id_;
}

} // namespace nn_ir
} // namespace nn_compiler
