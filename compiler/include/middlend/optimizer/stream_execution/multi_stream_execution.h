#pragma once

#include "common/pass.hpp"

namespace nn_compiler
{
namespace middlend
{
/** @Details:
 *
 **/
class MutiStreamExecution : public Pass
{
   public:
    MutiStreamExecution();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~MutiStreamExecution() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> muti_stream_layers_;

    bool isSameLayerType(std::vector<std::shared_ptr<ir::NNLayer>>& predecessors);

    int reorganizeLayerOrders(std::shared_ptr<ir::NNGraph>& graph, int start_idx);

    void insertMultiStreamLayer(std::shared_ptr<ir::NNGraph>& graph, int idx);

};  // class MutiStreamExecution

}  // namespace middlend
}  // namespace nn_compiler
