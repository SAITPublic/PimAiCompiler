#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include "builder/model_builder.h"
#include "executor/stream_executor.h"
#include "new_ir/include/nn_model.h"
#include "types.h"

namespace nn_compiler {
namespace runtime {

class NNRuntime
{
   public:
    NNRuntime() {}

    NNRuntime(std::unique_ptr<nn_compiler::ir::NNModel> &model, std::string model_type = "");

    std::vector<torch::Tensor> inferenceModel(std::unique_ptr<nn_compiler::ir::NNModel> &model,
                                              const std::vector<torch::Tensor>& input_tensors,
                                              bool profiling = false);

    int rocblas_init(void);
    
    int test(void);

    ~NNRuntime();

   private:
    std::shared_ptr<ModelBuilder> mbuilder_ = nullptr;

    std::shared_ptr<StreamExecutor> executor_ = nullptr;

    std::string model_type_ = "";
};

}  // namespace runtime
}  // namespace nn_compiler
