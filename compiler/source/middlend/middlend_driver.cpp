#include "common/include/command_line_parser.hpp"
#include "common/include/file_parser.hpp"

#include "compiler/include/middlend/common/log.hpp"
#include "ir/include/ir_importer.hpp"

#include "compiler/include/middlend/middlend_driver.hpp"

/// Command-line options
// File path
static cl_opt::Option<std::string>
    in_ir_file_path(std::vector<std::string>{"-i", "--input"}, "<file>", "IR file path", cl_opt::Required::YES);

namespace nn_compiler {
namespace middlend {

void MiddlendDriver::initialize() {
}

void MiddlendDriver::importIR() {
    Log::ME::I() << "[importer] NNCompiler MiddlendDriver::importIR() is called";

    IRImporter ir_importer;
    ir_importer.getNNIRFromFile(in_ir_file_path, graphs_);
}

void MiddlendDriver::runPasses() {
}

void MiddlendDriver::exportIR() {
}

void MiddlendDriver::finalize() {
}

} // namespace middlend
} //namespace nn_compiler
