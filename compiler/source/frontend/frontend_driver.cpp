#include <experimental/filesystem>

#include "common/include/command_line_parser.hpp"

#include "compiler/include/frontend/frontend_driver.hpp"

#include "ir/include/ir_importer.hpp"

static cl_opt::Option<std::string>
        graphgen_path(std::vector<std::string>{"-g", "--graphgen_path"},
                              "<path>", "GraphGen real path",
                              cl_opt::Required::NO, cl_opt::Hidden::NO,
                              "None");

namespace nn_compiler {
namespace frontend {

RetVal FrontendDriver::initialize(const std::string& in_file_path) {
    Log::FE::I() << "NNCompiler FrontendDriver::initialize() is called";
    graphgen_path_ = static_cast<std::string>(graphgen_path);
    if (graphgen_path_.compare("None") == 0 ||
            graphgen_path_.find("GraphGen") == std::string::npos || graphgen_path_.find("../") != std::string::npos) {
        Log::FE::E() << "Please specify the real path of GraphGen (with option -g).";
    }

    in_file_path_ = in_file_path;

    return RetVal::SUCCESS;
}

RetVal FrontendDriver::run() {
    Log::FE::I() << "NNCompiler FrontendDriver::run() is called";

    std::string command = createRunningCommand();

    auto return_code = std::system(command.c_str());
    if (return_code != 0) {
        Log::FE::E() << "Failed with return value: " << return_code;
    }

    return RetVal::SUCCESS;
}

RetVal FrontendDriver::wrapup(std::vector<std::shared_ptr<nn_compiler::nn_ir::NNIR>>& NNIR_graphs) {
    Log::FE::I() << "NNCompiler FrontendDriver::wrapup() is called";

    importFrontendIR();

    NNIR_graphs.clear();
    for (auto graph : graphs_) {
        NNIR_graphs.push_back(graph);
    }
    return RetVal::SUCCESS;
}

RetVal FrontendDriver::finalize() {
    Log::FE::I() << "NNCompiler FrontendDriver::finalize() is called";

    return RetVal::SUCCESS;
}

std::string FrontendDriver::createRunningCommand() {
    std::string set_path_command = "PATH=$PATH:" + graphgen_path_ + "/build/tools/pim_ir_generator ";
    std::string command = set_path_command + "pim_ir_generator ";
    command.append(in_file_path_);
    return command;
}

void FrontendDriver::importFrontendIR() {
    IRImporter ir_importer;
    std::string ir_file_path = "output/pim/frontend/frontend.ir"; // based on GraphGen setting
    ir_importer.getNNIRFromFile(ir_file_path, graphs_);
}

} // namespace frontend
} //namespace nn_compiler
