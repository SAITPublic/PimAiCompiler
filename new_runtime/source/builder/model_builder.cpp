#include "common/log.hpp"
#include "new_runtime/include/builder/model_builder.h"

namespace nn_compiler {
namespace runtime {

RetVal ModelBuilder::preProcess(std::unique_ptr<nn_compiler::ir::NNModel> &model) {
    return RetVal::SUCCESS;
}

RetVal ModelBuilder::preloadModel(std::unique_ptr<nn_compiler::ir::NNModel> &model) {
    return RetVal::SUCCESS;
}

RetVal ModelBuilder::loadWeightAndBias(){
    return RetVal::SUCCESS;
}

}  // namespace runtime
}  // namespace nn_compiler
