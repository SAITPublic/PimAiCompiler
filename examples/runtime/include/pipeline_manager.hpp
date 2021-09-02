#pragma once

#include "common/include/log.hpp"

namespace examples
{
class PipelineManager
{
 public:
    enum class ModelType { NO_MODEL = 0, RNNT, GNMT, HWR };

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
    RetVal initialize(const std::string& input_file, const int& compile_level, const std::string& model_type,
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

    int compile_level_ = 1;

    bool is_profiling_ = false;

    void load_and_run_rnnt();

    void load_and_run_gnmt();

    void load_and_run_hwr();

};  // class PipelineManager

}  // namespace examples
