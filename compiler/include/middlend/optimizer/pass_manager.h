#pragma once

#include "compiler/include/common/pass.hpp"
#include "ir/include/nn_model.h"

namespace nn_compiler {
namespace middlend {

class PassManager {
 public:
    explicit PassManager();

    void runPasses(std::unique_ptr<nn_compiler::ir::NNModel>& model);
};

}  // namespace middlend
}  // namespace nn_compiler
