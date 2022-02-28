#include "builder/model_builder.h"
#include "compiler/include/nn_compiler.hpp"

namespace nn_compiler {
namespace runtime {

RetVal ModelBuilder::preProcess() {
    return RetVal::SUCCESS;
}

RetVal ModelBuilder::compileModel(const int compile_level, const std::string model_type) {
    nn_compiler::compiler::NNCompiler compiler;

    return RetVal::SUCCESS;
}

RetVal ModelBuilder::preloadModel() {

    return RetVal::SUCCESS;
}

std::pair<std::unique_ptr<nn_compiler::ir::NNModel>, ModelBuilder::blob_store_type>
ModelBuilder::getModel() {
    return std::make_pair(this->model_, this->preloaded_blobs_container_);
}

RetVal ModelBuilder::loadWeightAndBias(){
    return RetVal::SUCCESS;
}

}  // namespace runtime
}  // namespace nn_compiler
