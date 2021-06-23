#pragma once

#include <vector>

#include "ir/include/nn_ir.hpp"

#include "compiler/include/middlend/context/compilation_context.hpp"
#include "compiler/include/middlend/context/compilation_context_builder.hpp"
#include "compiler/include/middlend/passes/pass_manager.hpp"
#include "compiler/include/middlend/trait/trait_manager.hpp"
#include "compiler/include/middlend/utils/util_builder.hpp"
#include "compiler/include/middlend/utils/util_manager.hpp"

#include "compiler/include/middlend/passes/pass_service/conf_json/conf_json.hpp"

namespace nn_compiler {
namespace middlend {

class MiddlendDriver {
 public:
    MiddlendDriver() = default;

    MiddlendDriver(const MiddlendDriver&) = delete;
    MiddlendDriver(MiddlendDriver&&)      = delete;

    MiddlendDriver& operator=(const MiddlendDriver&) = delete;
    MiddlendDriver& operator=(MiddlendDriver&&) = delete;

    /**
     * @brief     initialize a compilation pipeline
     * @details   This function initialize a compilation pipeline
     * @returns   return code
     */
    RetVal initialize();

    /**
     * @brief   Import IR file and compose a pass pipeline
     * @details This function import NNIR graphs and initialize passes
     * @returns return code
     */
    RetVal build();

    /**
     * @brief   Perform optimization/transformation on NNIR graphs
     * @details This function calls the pipeline to perform optimization/transformation
     * @returns return code
     */
    RetVal run();

    /**
     * @brief   Do post-compile actions
     * @details This function writes the result and verifies the result IR (optional)
     * @returns return code
     */
    RetVal wrapup();

    /**
     * @brief   Destroy all data and terminate the program
     * @details This function destroies all remained data and releases allocated memories
     * @returns return code
     */
    RetVal finalize();

 private:
    void importIR();

    void buildPasses();

    void exportIR();

    // number of registered passes
    uint32_t pass_counter_ = 0;

    void bumpPassCounter() { pass_counter_++; }

    std::vector<std::unique_ptr<nn_compiler::nn_ir::NNIR>> graphs_;

    // manager of compiler passes and utils
    PassManager<>      base_pass_manager_;
    UtilManager        base_util_manager_;
    CompilationContext base_context_;
    TraitManager       base_trait_manager_;

}; // class MiddlendDriver

} // namespace middlend
} // namespace nn_compiler
