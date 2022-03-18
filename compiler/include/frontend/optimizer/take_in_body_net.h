#pragma once

#include "compiler/include/common/pass.hpp"

namespace nn_compiler
{
namespace frontend
{
class TakeInBodyNet : public Pass
{
   public:
    TakeInBodyNet();

    void fitIfCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void fitLoopCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~TakeInBodyNet() = default;

   private:
    std::vector<std::pair<std::shared_ptr<nn_compiler::ir::NNLayer>, std::shared_ptr<nn_compiler::ir::NNNetwork>>>
        prim_if_layers_;

    std::vector<std::pair<std::shared_ptr<nn_compiler::ir::NNLayer>, std::shared_ptr<nn_compiler::ir::NNNetwork>>>
        prim_loop_layers_;

    void take_in_if_body(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void take_in_loop_body(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    uint32_t getUniqueTensorId(std::unique_ptr<nn_compiler::ir::NNModel>& model);
};

}  // namespace frontend
}  // namespace nn_compiler
