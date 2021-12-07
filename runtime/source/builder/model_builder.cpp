#include "builder/model_builder.h"
#include "compiler/include/nn_compiler.hpp"
#include "ir/include/nn_ir.hpp"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
RetVal ModelBuilder::compileModel(const int compile_level, const std::string model_type)
{
    nn_compiler::NNCompiler compiler;
    std::vector<std::shared_ptr<nncir::NNIR>> NNIR_graphs;
    compiler.initialize(compile_level, this->model_path_, model_type);
    compiler.compile(NNIR_graphs);
    compiler.finalize();
    this->runnable_ir_ = NNIR_graphs.front();

    return RetVal::SUCCESS;
}

RetVal ModelBuilder::preloadModel() { return RetVal::SUCCESS; }

std::pair<std::shared_ptr<nncir::NNIR>, ModelBuilder::blob_store_type> ModelBuilder::getModel() {
    return std::make_pair(this->runnable_ir_, this->preloaded_blobs_container_);
}

}  // namespace nnrt
