/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#pragma once

#include "common/include/types.hpp"
#include "frontend/importer/model_builder.h"
#include "frontend/optimizer/pass_manager.h"

namespace nn_compiler
{
namespace frontend
{
class FrontendDriver
{
   public:
    FrontendDriver() = default;

    FrontendDriver(const FrontendDriver&) = delete;
    FrontendDriver(FrontendDriver&&) = delete;

    FrontendDriver& operator=(const FrontendDriver&) = delete;
    FrontendDriver& operator=(FrontendDriver&&) = delete;

    /**
     * @brief     initialize a frontend pipeline
     * @details   This function initialize a frontend pipeline
     * @inputs    const std::string& in_file_path, const std::string& model_name
     * @returns   return code
     */
    RetVal initialize(const std::string& in_file_path, const std::string& model_name);

    /**
     * @brief   Call and run frontend
     * @details This function calls the pipeline of frontend
     * @returns return code
     */
    RetVal run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    /**
     * @brief   Destroy all data and terminate the program
     * @details This function destroies all remained data and releases allocated memories
     * @returns return code
     */
    RetVal finalize();

   private:
    std::string in_file_path_ = "";

    std::string model_name_ = "";

    /**
     * @brief   Import model
     * @details This function reads in mode file and build model in NNModel
     * @returns return code
     */
    RetVal importer(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    /**
     * @brief   Apply Graph optimizations
     * @details This function runs graph optimization passes onto model graph
     * @returns return code
     */
    RetVal optimizer(std::unique_ptr<nn_compiler::ir::NNModel>& model);
};  // class FrontendDriver

}  // namespace frontend
}  // namespace nn_compiler
