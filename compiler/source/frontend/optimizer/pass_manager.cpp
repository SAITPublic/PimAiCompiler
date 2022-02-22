#include "compiler/include/common/log.hpp"
#include "compiler/include/frontend/optimizer/pass_manager.h"

namespace nn_compiler {
namespace frontend {

PassManager::PassManager(const std::string& model_name) {
    model_name_ = model_name;
}

void PassManager::runPasses(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
    Log::FE::I() << "PassManager::runPasses is called.";
    auto base_pass = std::make_shared<Pass>();

    // 1. TODO(SRCX): declare optimization passes, like: auto fuse_act = std::make_shared<FuseActivation>();

    // 2. TODO(SRCX): add optimization passes, like: base_pass->add(fuse_act);

    while (base_pass->getSuccessor() != nullptr) {
        base_pass = base_pass->getSuccessor();
        if (base_pass->fitCondition(model)) {
            base_pass->run(model);
        }
    }
}

} // namespace frontend
} // namespace nn_compiler
