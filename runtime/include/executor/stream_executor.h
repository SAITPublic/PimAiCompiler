#pragma once
#include <torch/script.h>
#include <string>
#include "model_builder.h"
#include "nnrt_types.h"
#include "ir/include/nn_ir.hpp"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
class StreamExecutor
{
   public:
    StreamExecutor() {}

     RetVal inferenceModel(const std::shared_ptr<nncir::NNIR> runnable_ir, const std::vector<torch::Tensor>& input_tensors, std::vector<torch::Tensor>& output_tensors);

   private:
};

}  // namespace nnrt
