#include "compiler/include/common/log.hpp"
#include "ir/include/ir_importer.hpp"

#include "compiler/include/middlend/middlend_driver.hpp"

#include <cstdio>
#include <cstdlib> 

namespace nn_compiler {
namespace middlend {

RetVal MiddlendDriver::initialize() {
    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::run(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
    Log::ME::I() << "NNCompiler MiddlendDriver::run() is called";

    optimizer(model);

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::optimizer(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    Log::ME::I() << "NNCompiler MiddlendDriver::optimizer() is called";
    // TODO(SRCX): implement this code.

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::finalize() {
    return RetVal::SUCCESS;
}

} // namespace middlend
} //namespace nn_compiler
