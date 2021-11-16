#pragma once

#include <memory>
#include <tuple>
#include <vector>
#include "builder/model_builder.h"
#include "executor/stream_executor.h"
#include "executor/aten_ops.h"
#include "nnrt_types.h"

namespace nnrt
{
class NNRuntime
{
   public:
    NNRuntime() {}

    NNRuntime(const std::string torch_model_path, int compile_level = 1, std::string model_type = "");

    std::vector<torch::Tensor> inferenceModel(const std::vector<torch::Tensor>& input_tensors,
                                              bool profiling = false);

    int rocblas_init(void);
    
    int test(void);

    ~NNRuntime();

   private:
    // Runnable NNIR in ModelBuilder
    std::shared_ptr<ModelBuilder> mbuilder_;

    std::shared_ptr<StreamExecutor> executor_;
};

}  // namespace nnrt
