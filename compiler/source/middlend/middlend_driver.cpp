#include "common/include/command_line_parser.hpp"

#include "compiler/include/middlend/common/log.hpp"
#include "ir/include/ir_importer.hpp"

#include "compiler/include/middlend/middlend_driver.hpp"

/// Command-line options
// File path
static cl_opt::Option<std::string>
    in_ir_file_path(std::vector<std::string>{"-i", "--input"}, "<file>", "IR file path", cl_opt::Required::YES);

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
