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

#pragma once

#include "common/include/types.hpp"
#include "ir/include/nn_model.h"

namespace nn_compiler
{
namespace middlend
{
class MiddlendDriver
{
   public:
    MiddlendDriver() = default;

    MiddlendDriver(const MiddlendDriver&) = delete;
    MiddlendDriver(MiddlendDriver&&) = delete;

    MiddlendDriver& operator=(const MiddlendDriver&) = delete;
    MiddlendDriver& operator=(MiddlendDriver&&) = delete;

    /**
     * @brief     initialize a compilation pipeline
     * @details   This function initialize a compilation pipeline
     * @inputs    std::vector<std::unique_ptr<nn_compiler::nn_ir::NNIR>> NNIR_graphs
     * @returns   return code
     */
    RetVal initialize();

    /**
     * @brief   Perform optimization/transformation on NNIR graphs
     * @details This function calls the pipeline to perform optimization/transformation
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
    /**
     * @brief   Apply optimizations
     * @details This function runs optimization passes onto model graph
     * @returns return code
     */
    RetVal optimizer(std::unique_ptr<nn_compiler::ir::NNModel>& model);

};  // class MiddlendDriver

}  // namespace middlend
}  // namespace nn_compiler
