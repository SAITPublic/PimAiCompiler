#pragma once

#include "compiler/include/frontend/optimizer/pass.h"

namespace nn_compiler {

namespace frontend {

class FuseActivation : public Pass {
 public:
    FuseActivation();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~FuseActivation() = default;

 private:
    std::vector<std::string> supported_host_types_  = {"Convolution", "InnerProduct"};
    std::vector<std::string> supportd_torch_host_types_ = {"aten::transpose"};
    bool feasibleHostType(const std::string &type);

    std::vector<std::string> supported_torch_parasite_types_ = {"aten::relu", "aten::max"};
    std::vector<std::string> supported_parasite_types_ = {"ReLU",
                                                          "Clip",
                                                          "ApproxSigmoid",
                                                          "PieceWiseLinear",
                                                          "ReLU6"};
    bool feasibleParasiteType(const std::string &type);
    void doFuseActivation(std::shared_ptr<nn_compiler::ir::NNNetwork>& graph);
};

}  // namespace frontend
}  // namespace nn_compiler
