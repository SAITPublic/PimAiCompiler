#pragma once

#include "compiler/include/common/pass.hpp"

namespace nn_compiler {

namespace frontend {

class FuseActivation : public Pass {
 public:
    FuseActivation();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~FuseActivation() = default;

 private:
    std::vector<std::string> supportd_host_types_ = {"aten::transpose"};
    bool feasibleHostType(const std::string &type);

    std::vector<std::string> supported_parasite_types_ = {"aten::relu", "aten::max"};

    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers_;

    bool feasibleParasiteType(const std::string &type);
};

}  // namespace frontend
}  // namespace nn_compiler
