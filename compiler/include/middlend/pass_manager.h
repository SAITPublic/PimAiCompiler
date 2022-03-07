#pragma once

#include "compiler/include/frontend/optimizer/pass.h"
#include "new_ir/include/nn_model.h"

namespace nn_compiler {
namespace middlend {

class PassManager {
 public:
    explicit PassManager();

    void runPasses(std::unique_ptr<nn_compiler::ir::NNModel>& model);

 private:
};

}  // namespace middlend
}  // namespace nn_compiler
