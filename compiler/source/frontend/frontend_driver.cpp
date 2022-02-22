#include <experimental/filesystem>

#include "common/include/command_line_parser.hpp"

#include "compiler/include/frontend/frontend_driver.hpp"

#include "ir/include/ir_importer.hpp"

namespace nn_compiler {
namespace frontend {

RetVal FrontendDriver::initialize(const std::string& in_file_path, const std::string& model_name) {
    Log::FE::I() << "NNCompiler FrontendDriver::initialize() is called";
    in_file_path_ = in_file_path;
    model_name_   = model_name;

    return RetVal::SUCCESS;
}

RetVal FrontendDriver::run(std::unique_ptr<nn_compiler::ir::NNModel> model) {
    Log::FE::I() << "NNCompiler FrontendDriver::run() is called";

    importer(model);

    optimizer(model);

    return RetVal::SUCCESS;
}

RetVal FrontendDriver::finalize() {
    Log::FE::I() << "NNCompiler FrontendDriver::finalize() is called";

    return RetVal::SUCCESS;
}

RetVal FrontendDriver::importer(std::unique_ptr<nn_compiler::ir::NNModel> model) {
    // TODO(SRCX): model importer

    return RetVal::SUCCESS;
}

RetVal FrontendDriver::optimizer(std::unique_ptr<nn_compiler::ir::NNModel> model) {
    // TODO(SRCX): graph optimizer

    return RetVal::SUCCESS;
}

} // namespace frontend
} //namespace nn_compiler
