#pragma once

#include "compiler/include/common/log.hpp"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {
namespace frontend {

class FrontendDriver {
 public:
    FrontendDriver() = default;

    FrontendDriver(const FrontendDriver&) = delete;
    FrontendDriver(FrontendDriver&&) = delete;

    FrontendDriver& operator=(const FrontendDriver&) = delete;
    FrontendDriver& operator=(FrontendDriver&&) = delete;

    /**
     * @brief     initialize a frontend pipeline
     * @details   This function initialize a frontend pipeline
     * @inputs    std::string& in_file_path
     * @returns   return code
     */
    RetVal initialize(const std::string& in_file_path);

    /**
     * @brief   Call and run frontend
     * @details This function calls the pipeline of frontend
     * @returns return code
     */
    RetVal run();

    /**
     * @brief    Do post-compile actions
     * @details  This function reads in the result IR geneated by frontend
     * @inputs   A vector of NNIR graphs
     * @returns  return code
     */
    RetVal wrapup(std::vector<std::shared_ptr<nn_compiler::nn_ir::NNIR>>& NNIR_graphs);

    /**
     * @brief   Destroy all data and terminate the program
     * @details This function destroies all remained data and releases allocated memories
     * @returns return code
     */
    RetVal finalize();

 private:
    std::string in_file_path_ = "";

    std:: string graphgen_path_ = "";

    std::vector<std::shared_ptr<nn_compiler::nn_ir::NNIR>> graphs_;

    std::string createRunningCommand();

    void importFrontendIR();

}; // class FrontendDriver

} // namespace frontend
} // namespace nn_compiler
