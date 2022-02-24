#pragma once

#include "compiler/include/frontend/optimizer/pass.h"
#include "new_ir/include/nn_model.h"

namespace nn_compiler {
namespace frontend {

class PassManager {
 public:
    explicit PassManager(const std::string& model_name);

    void runPasses(std::unique_ptr<nn_compiler::ir::NNModel>& model);

 private:
    std::string model_name_ = "";
};

}  // namespace frontend
}  // namespace nn_compiler
