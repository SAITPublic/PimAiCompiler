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

namespace examples
{
class PipelineManager
{
   public:
    enum class ModelType { NO_MODEL = 0, RNNT, GNMT, HWR, Transformer, SwitchTransformer };

    PipelineManager() = default;

    PipelineManager(const PipelineManager&) = delete;
    PipelineManager(PipelineManager&&) = delete;

    PipelineManager& operator=(const PipelineManager&) = delete;
    PipelineManager& operator=(PipelineManager&&) = delete;

    /**
     * @brief     initialize a pipeline manager
     * @details   This function initialize a pipeline manager
     * @inputs    const std::string& input_file, const int& compile_level, const std::string& model_type,
     *            const bool& profiling
     * @returns   return code
     */
    RetVal initialize(const std::string& input_file, const std::string& model_type, const int& gpu_num,
                      const bool& profiling);

    /**
     * @brief   Call and run pipeline
     * @details This function runs the pipeline
     * @returns return code
     */
    RetVal run();

    /**
     * @brief   Destroy all data and terminate the program
     * @details This function destroies all remained data and releases allocated memories
     * @returns return code
     */
    RetVal finalize();

   private:
    std::string input_file_path_ = "";

    ModelType model_type_ = ModelType::NO_MODEL;

    bool is_profiling_ = false;

    int gpu_num_ = 1;

    void load_and_run_rnnt();

    void load_and_run_gnmt();

    void load_and_run_hwr();

    void load_and_run_transformer();

    void load_and_run_switchtransformer();

};  // class PipelineManager

}  // namespace examples
