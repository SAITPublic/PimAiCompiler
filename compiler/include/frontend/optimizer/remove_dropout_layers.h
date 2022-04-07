#pragma once

#include "common/pass.hpp"

namespace nn_compiler
{
namespace frontend
{
/** @Details:
 *  Remove unnecessary dropout layers as they do nothing at inference time.
 **/
class RemoveDropoutLayers : public Pass
{
   public:
    RemoveDropoutLayers();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~RemoveDropoutLayers() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> remove_layers_;
};

}  // namespace frontend
}  // namespace nn_compiler
