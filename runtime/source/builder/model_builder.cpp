#include "builder/model_builder.h"
#include "compiler/include/nn_compiler.hpp"
#include "ir/include/nn_ir.hpp"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
RetVal ModelBuilder::compileModel(const int compile_level)
{
    nn_compiler::NNCompiler compiler;
    std::vector<std::shared_ptr<nncir::NNIR>> NNIR_graphs;
    compiler.initialize(compile_level, this->model_path_);
    compiler.compile(NNIR_graphs);
    compiler.finalize();
    this->runnable_ir_ = NNIR_graphs.front();

    return RetVal::SUCCESS;
}

RetVal ModelBuilder::preloadModel() { return RetVal::SUCCESS; }

}  // namespace nnrt
