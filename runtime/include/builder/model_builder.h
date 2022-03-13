#pragma once

#include <torch/script.h>

#include "ir/include/nn_model.h"
#include "ir/include/types.h"

namespace nn_compiler
{
namespace runtime
{

using namespace nn_compiler::ir;
class ModelBuilder
{
   public:
    typedef std::unordered_map<int64_t, std::pair<DataType, torch::jit::IValue>> data_store_type;

    RetVal preProcess(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    RetVal preloadModel(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    RetVal loadWeightAndBias(nn_compiler::ir::DTensor &data);

    data_store_type getPreLoadedData() { return preloaded_data_container_; }

   private:
    data_store_type preloaded_data_container_;

    int preload_start_id_ = 0;
};

}  // namespace runtime
}  // namespace nn_compiler
