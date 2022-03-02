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

#include "common/include/log.hpp"
#include "compiler/include/frontend/frontend_driver.hpp"
#include "compiler/include/middlend/middlend_driver.hpp"

namespace nn_compiler {
namespace compiler {

class NNCompiler {
 public:
    NNCompiler() = default;

    NNCompiler(const NNCompiler&) = delete;
    NNCompiler(NNCompiler&&)      = delete;

    NNCompiler& operator=(const NNCompiler&) = delete;
    NNCompiler& operator=(NNCompiler&&) = delete;

    /**
     * @brief   initialize a compiler.
     * @details This function initializes a compiler,
     *          and build a compilation pipeline with frontend, middlend and backend.
     * @inputs  int compile_level, std::string file_path, std::string model_name
     * @returns return code: success or failure.
     */
    RetVal initialize(const int& compile_level, const std::string& file_path,
                      const std::string& model_name);

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

    // TODO(SRCX): remove this interface after refactor.
    /**
     * @brief   run compiler.
     * @details This function runs compilation pipeline.
     * @inputs  an empty vector ready to put NNIR(LLO) graphs
     * @returns const NNIR vector
     */
    RetVal compile(std::vector<std::shared_ptr<nn_compiler::nn_ir::NNIR>>& NNIR_graphs);

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

    // TODO(SRCX): remove this interface after refactor.
    /**
     * @brief   Milddlend compilation pipeline.
     * @details This function runs middlend compilation pipeline with NNIR input.
     * @inputs  NNIR file path
     * @returns return code: success or failure.
     */
    RetVal middlend(const std::string& file_path);

    /**
     * @brief   Milddlend compilation pipeline.
     * @details This function runs middlend compilation pipeline.
     * @inputs  Model graph container.
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

    // TODO(SRCX): remove compile_level option after refactor.
    /**
     * @details Compile level, possible values(0, 1, 2), which stands for different components of compilation pipeline.
     *          0 : frontend->middlend->backend;
     *          1 : middlend->backend;
     *          2 : backend.
     */
    int compile_level_ = 0;

    std::string input_file_path_  = "";

    std::unique_ptr<frontend::FrontendDriver> frontend_driver_ = nullptr;
    std::unique_ptr<middlend::MiddlendDriver> middlend_driver_ = nullptr;
    // TODO: Add members of backend drivers

    // TODO(SRCX): remove NNIR graphs when refactor successfully
    std::vector<std::shared_ptr<nn_compiler::nn_ir::NNIR>> NNIR_graphs_;

    std::string model_name_ = "";

}; // class NNCompiler

} // namespace compiler
} // namespace nn_compiler
