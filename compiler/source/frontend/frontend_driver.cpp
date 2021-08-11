#include <experimental/filesystem>

#include "common/include/command_line_parser.hpp"

#include "compiler/include/frontend/frontend_driver.hpp"

#include "ir/include/ir_importer.hpp"

namespace nn_compiler {
namespace frontend {

RetVal FrontendDriver::initialize(const std::string& in_file_path) {
    Log::FE::I() << "NNCompiler FrontendDriver::initialize() is called";
    in_file_path_ = in_file_path;

    graphgen_core_ = std::make_shared<graphgen::GraphGenCore>();

    return RetVal::SUCCESS;
}

RetVal FrontendDriver::run() {
    Log::FE::I() << "NNCompiler FrontendDriver::run() is called";

    try {
      graphgen_core_->model_import(in_file_path_);
    }
    catch(const std::exception& e) {
      Log::FE::E() << e.what();
    }

    graphgen_core_->intermediate_process();

    graphgen_core_->model_export("");

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

void FrontendDriver::importFrontendIR() {
    IRImporter ir_importer;
    std::string ir_file_path = "output/pim/frontend/frontend.ir"; // based on GraphGen setting
    ir_importer.getNNIRFromFile(ir_file_path, graphs_);
}

} // namespace frontend
} //namespace nn_compiler
