#pragma once

#include "common/pass.hpp"

namespace nn_compiler
{
namespace middlend
{
/** @Details:
 *
 **/
class MultiStreamExecution : public Pass
{
   public:
    MultiStreamExecution();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~MultiStreamExecution() = default;

   private:
    bool backwardToCheckAndFindStartLayer(
        std::unique_ptr<nn_compiler::ir::NNModel>& model,
        std::shared_ptr<nn_compiler::ir::NNLayer>& multi_stream_start_layer,
        std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>& multi_stream_apply_layers);

    void process(std::unique_ptr<nn_compiler::ir::NNModel>& model, int start_idx);

    std::map<std::shared_ptr<nn_compiler::ir::NNLayer>, std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>>
        start_to_stream_layers_;

    std::map<std::shared_ptr<nn_compiler::ir::NNLayer>, std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>>
        pre_layers_map_;

    std::vector<std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>> branches_;

    int max_search_level_ = 10;

};  // class MultiStreamExecution

}  // namespace middlend
}  // namespace nn_compiler
