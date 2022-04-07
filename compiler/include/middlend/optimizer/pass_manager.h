#pragma once

#include "common/pass.hpp"

namespace nn_compiler
{
namespace middlend
{
class PassManager
{
   public:
    explicit PassManager();

    void runPasses(std::unique_ptr<nn_compiler::ir::NNModel>& model);
};

}  // namespace middlend
}  // namespace nn_compiler
