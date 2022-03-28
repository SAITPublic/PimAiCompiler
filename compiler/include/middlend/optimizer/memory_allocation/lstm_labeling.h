#pragma once

#include "compiler/include/common/pass.hpp"

#include "ir/include/nn_model.h"
#include "ir/include/nn_network.h"
#include "ir/include/types.h"

namespace nn_compiler
{
namespace middlend
{
/** @Details:
 *  Set relative attributes of aten::LSTM1 layers for custom optmization pattern:
 *
 *                 |
 *            aten::lstm1
 *                 |
 *         prim::tuple_construct
 *                 |
 *            aten::append
 *                 |
 **/
class LstmLabeling : public Pass
{
   public:
    LstmLabeling();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~LstmLabeling() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> aten_lstm1_layers_;
};  // class LstmLabeling

}  // namespace middlend
}  // namespace nn_compiler
