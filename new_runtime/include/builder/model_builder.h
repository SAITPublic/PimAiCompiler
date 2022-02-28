#pragma once

#include <torch/script.h>

#include "new_ir/include/nn_model.h"

namespace nn_compiler {
namespace runtime {

class ModelBuilder
{
   public:
    typedef std::unordered_map<int64_t, std::pair<nnrt::DataType, torch::jit::IValue>> blob_store_type;

    ModelBuilder(std::string model_path) {
        this->model_path_ = model_path;
    }

    RetVal preProcess();

    RetVal compileModel(int compile_level, const std::string model_type);

    RetVal preloadModel();

    RetVal loadWeightAndBias();

    std::pair<std::unique_ptr<nn_compiler::ir::NNModel>, blob_store_type> getModel();

   private:
    std::unique_ptr<nn_compiler::ir::NNModel> model_ = nullptr;

    blob_store_type preloaded_blobs_container_;

    std::string model_path_;
};

}  // namespace runtime
}  // namespace nn_compiler
