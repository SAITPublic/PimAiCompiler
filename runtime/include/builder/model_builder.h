#pragma once

#include <torch/script.h>

#include "common/include/types.hpp"
#include "half.hpp"
#include "ir/include/layers/all_layers.h"
#include "ir/include/nn_model.h"
#include "ir/include/tensors/data_tensor.h"
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

    std::pair<std::vector<int64_t>, std::vector<int64_t>> loadWeightAndBias(std::vector<at::Tensor> weight_data,
                                                                            std::vector<at::Tensor> bias_data);

    data_store_type getPreLoadedData() { return preloaded_data_container_; }

   private:
    data_store_type preloaded_data_container_;

    int64_t preload_id_ = 0;
};

}  // namespace runtime
}  // namespace nn_compiler
