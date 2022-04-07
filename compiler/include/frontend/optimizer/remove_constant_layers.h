#pragma once

#include "common/pass.hpp"

namespace nn_compiler
{
namespace frontend
{
/** @Details:
 *  Remove unnecessary constant layers to simplify the graph.
 **/
class RemoveConstantLayers : public Pass
{
   public:
    RemoveConstantLayers();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~RemoveConstantLayers() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> remove_layers_;
};

}  // namespace frontend
}  // namespace nn_compiler
