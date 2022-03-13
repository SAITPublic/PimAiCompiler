#include <cstdio>
#include <cstdlib> 

#include "compiler/include/common/log.hpp"
#include "compiler/include/middlend/middlend_driver.hpp"
#include "compiler/include/middlend/optimizer/pass_manager.h"

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
    
    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->runPasses(model);

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::finalize() {
    return RetVal::SUCCESS;
}

} // namespace middlend
} //namespace nn_compiler
