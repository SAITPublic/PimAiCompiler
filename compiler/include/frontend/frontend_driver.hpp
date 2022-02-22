#pragma once

#include <memory>
#include <string>
#include <vector>

#include "compiler/include/common/log.hpp"
#include "ir/include/nn_ir.hpp"
#include "new_ir/include/nn_model.h"

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

    // TODO(SRCX): remove NNIR graphs when refactor successfully
    std::vector<std::shared_ptr<nn_compiler::nn_ir::NNIR>> graphs_;

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
}; // class FrontendDriver

} // namespace frontend
} // namespace nn_compiler
