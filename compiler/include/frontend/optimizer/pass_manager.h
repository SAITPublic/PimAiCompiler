#pragma once

#include "common/pass.hpp"

namespace nn_compiler
{
namespace frontend
{
class PassManager
{
   public:
    explicit PassManager(const std::string& model_name);

    void runPasses(std::unique_ptr<nn_compiler::ir::NNModel>& model);

   private:
    std::string model_name_ = "";
};

}  // namespace frontend
}  // namespace nn_compiler
