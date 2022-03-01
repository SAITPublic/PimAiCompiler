#pragma once

#include <torch/script.h>

#include "new_ir/include/nn_model.h"
#include "types.h"

namespace nn_compiler {
namespace runtime {

class ModelBuilder
{
   public:
    typedef std::unordered_map<int64_t, std::pair<DataType, torch::jit::IValue>> blob_store_type;

    RetVal preProcess(std::unique_ptr<nn_compiler::ir::NNModel> &model);

    RetVal preloadModel(std::unique_ptr<nn_compiler::ir::NNModel> &model);

    RetVal loadWeightAndBias();

    blob_store_type getPreLoadedData() { return preloaded_blobs_container_; }

   private:
    blob_store_type preloaded_blobs_container_;

    std::string model_path_;
};

}  // namespace runtime
}  // namespace nn_compiler
