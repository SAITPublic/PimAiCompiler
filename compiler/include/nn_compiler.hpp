/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "compiler/include/frontend/frontend_driver.hpp"
#include "compiler/include/middlend/middlend_driver.hpp"

namespace nn_compiler
{
namespace compiler
{
class NNCompiler
{
   public:
    NNCompiler() = default;

    NNCompiler(const NNCompiler&) = delete;
    NNCompiler(NNCompiler&&) = delete;

    NNCompiler& operator=(const NNCompiler&) = delete;
    NNCompiler& operator=(NNCompiler&&) = delete;

    /**
     * @brief   initialize a compiler.
     * @details This function initializes a compiler,
     *          and build a compilation pipeline with frontend, middlend and backend.
     * @inputs  int compile_level, std::string file_path, std::string model_name
     * @returns return code: success or failure.
     */
    RetVal initialize(const std::string& file_path, const std::string& model_name);

    /**
     * @brief   run compiler.
     * @details This function runs compilation pipeline.
     * @inputs  int compile_level, std::string file_path
     * @returns return code: success or failure.
     */
    RetVal compile();

    /**
     * @brief   run compiler.
     * @details This function runs compilation pipeline.
     * @inputs  model graph container.
     * @returns return code: success or failure.
     */
    RetVal compile(std::unique_ptr<ir::NNModel>& model);

    /**
     * @brief   Destroy all data of class.
     * @details This function destroies all remained data and releases allocated memories of the compiler.
     * @returns return code: success or failure.
     */
    RetVal finalize();

   private:
    /**
     * @brief   Frontend compilation pipeline.
     * @details This function runs frontend compilation pipeline with torchscript IR input.
     * @inputs  torchscript IR file path, model type[RNNT, GNMT, HWR] and model graph container.
     * @returns return code: success or failure.
     */
    RetVal frontend(const std::string& file_path, const std::string& model_name, std::unique_ptr<ir::NNModel>& model);

    /**
     * @brief   Milddlend compilation pipeline.
     * @details This function runs middlend compilation pipeline with NNIR input.
     * @inputs  NNIR file path
     * @returns return code: success or failure.
     */
    RetVal middlend(std::unique_ptr<ir::NNModel>& model);

    /**
     * @brief   Backend compilation pipeline.
     * @details This function runs backend compilation pipeline with no input,
     *          NNIR graphs have been parsed and get prepared by frontend or middlend.
     * @inputs  .
     * @returns return code: success or failure.
     */
    RetVal backend();

    std::string input_file_path_ = "";

    std::unique_ptr<frontend::FrontendDriver> frontend_driver_ = nullptr;
    std::unique_ptr<middlend::MiddlendDriver> middlend_driver_ = nullptr;
    // TODO: Add members of backend drivers

    std::string model_name_ = "";

};  // class NNCompiler

}  // namespace compiler
}  // namespace nn_compiler
