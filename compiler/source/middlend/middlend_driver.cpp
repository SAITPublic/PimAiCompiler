#include "common/include/command_line_parser.hpp"

#include "compiler/include/middlend/common/log.hpp"
#include "ir/include/ir_importer.hpp"

#include "compiler/include/middlend/middlend_driver.hpp"

/// Command-line options
// File path
static cl_opt::Option<std::string>
    in_ir_file_path(std::vector<std::string>{"-i", "--input"}, "<file>", "IR file path", cl_opt::Required::YES);

static cl_opt::Option<int> verify_level("-v",
                                        "<level>",
                                        "verification level. Possible values: 0, 1, 2 (default: 0)",
                                        cl_opt::Required::NO,
                                        cl_opt::Hidden::NO,
                                        static_cast<int>(nn_compiler::PassManager<>::VerificationLevelType::NONE),
                                        std::vector<std::string>{"0", "1", "2"});

static cl_opt::Option<std::string> pass_config_file_path(std::vector<std::string>{"-c", "--configuration"},
                                                         "<file>",
                                                         "passes configuration file path",
                                                         cl_opt::Required::YES);

namespace nn_compiler {
namespace middlend {

RetVal MiddlendDriver::initialize() {
    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::build() {
    Log::ME::I() << "NNCompiler MiddlendDriver::build() is called";
    importIR();

    buildPasses();

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::run() {
    Log::ME::I() << "NNCompiler MiddlendDriver::run() is called";

    // #1. Call the passes pipeline
    base_pass_manager_.initialize(base_util_manager_, base_trait_manager_);

    for (const auto& graph : graphs_) {
        base_pass_manager_.capability_check(*graph);
    }

    for (const auto& graph : graphs_) {
        base_context_.resetLocalData();

        base_pass_manager_.run(*graph, base_context_, verify_level);
    }

    base_pass_manager_.finalize();

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::wrapup() {
    Log::ME::I() << "NNCompiler MiddlendDriver::wrapup() is called";
    exportIR();

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::finalize() {
    return RetVal::SUCCESS;
}

void MiddlendDriver::importIR() {
    IRImporter ir_importer;
    ir_importer.getNNIRFromFile(in_ir_file_path, graphs_);
}

void MiddlendDriver::buildPasses() {
}

void MiddlendDriver::exportIR() {
}

} // namespace middlend
} //namespace nn_compiler
