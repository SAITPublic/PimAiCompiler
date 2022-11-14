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

#include <experimental/filesystem>

#include "frontend/frontend_driver.hpp"

namespace nn_compiler
{
namespace frontend
{
RetVal FrontendDriver::initialize(const std::string& in_file_path, const std::string& model_name)
{
    DLOG(INFO) << "NNCompiler FrontendDriver::initialize() is called";
    in_file_path_ = in_file_path;
    model_name_ = model_name;

    return RetVal::SUCCESS;
}

RetVal FrontendDriver::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "NNCompiler FrontendDriver::run() is called";

    importer(model);

    optimizer(model);

    return RetVal::SUCCESS;
}

RetVal FrontendDriver::finalize()
{
    DLOG(INFO) << "NNCompiler FrontendDriver::finalize() is called";

    return RetVal::SUCCESS;
}

RetVal FrontendDriver::importer(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    std::shared_ptr<ModelBuilder> builder = std::make_shared<ModelBuilder>();
    builder->build(model, in_file_path_);

    return RetVal::SUCCESS;
}

RetVal FrontendDriver::optimizer(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto pass_manager = std::make_shared<PassManager>(model_name_);
    pass_manager->runPasses(model);

    return RetVal::SUCCESS;
}

}  // namespace frontend
}  // namespace nn_compiler
