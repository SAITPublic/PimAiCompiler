#pragma once

#include <vector>

#include "compiler/include/common/log.hpp"
#include "new_ir/include/nn_model.h"


namespace nn_compiler {
namespace middlend {

class MiddlendDriver {
 public:
    MiddlendDriver() = default;

    MiddlendDriver(const MiddlendDriver&) = delete;
    MiddlendDriver(MiddlendDriver&&)      = delete;

    MiddlendDriver& operator=(const MiddlendDriver&) = delete;
    MiddlendDriver& operator=(MiddlendDriver&&)      = delete;


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
     * @brief   Apply Graph optimizations
     * @details This function runs graph optimization passes onto model graph
     * @returns return code
     */
    RetVal optimizer(std::unique_ptr<nn_compiler::ir::NNModel>& model);

}; // class MiddlendDriver

} // namespace middlend
} // namespace nn_compiler
