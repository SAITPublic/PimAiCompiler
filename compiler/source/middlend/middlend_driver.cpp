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

#include "middlend/middlend_driver.hpp"
#include "middlend/optimizer/pass_manager.h"

namespace nn_compiler
{
namespace middlend
{
RetVal MiddlendDriver::initialize() { return RetVal::SUCCESS; }

RetVal MiddlendDriver::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "NNCompiler MiddlendDriver::run() is called";

    optimizer(model);

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::optimizer(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "NNCompiler MiddlendDriver::optimizer() is called";

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->runPasses(model);

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::finalize() { return RetVal::SUCCESS; }

}  // namespace middlend
}  // namespace nn_compiler
