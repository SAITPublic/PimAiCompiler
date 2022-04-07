#pragma once

#include "common/pass.hpp"

namespace nn_compiler
{
namespace middlend
{
/** @Details:
 *  Process and help prim::If/prim::EndIf/prim::Loop/PrimEndLoop to find their next layer in execution time.
 *  (1) For Prim::If layer, set where is the start layer of its then-net as well as else-net.
 *  (2) For Prim::EndIf layer, set which is the next layer after this sub-net (then-net or else-net).
 *  (3) For Prim::Loop layer, set where is the next layer after this loop.
 *  (4) For Prim::EndLoop layer, set where is the next layer after loop body runs, i.e., its corresponding Prim::Loop
 *      layer.
 **/
class ControlLayerExecution : public Pass
{
   public:
    ControlLayerExecution();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~ControlLayerExecution() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> control_layers_;

};  // class ControlLayerExecution

}  // namespace middlend
}  // namespace nn_compiler
