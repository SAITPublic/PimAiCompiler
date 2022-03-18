#include "compiler/include/nn_compiler.hpp"
#include "runtime/include/nn_runtime.h"

#include "pipeline_manager.h"

using namespace nn_compiler::compiler;
using namespace nn_compiler::runtime;
namespace fs = std::experimental::filesystem;

namespace NNRuntimeInterface
{
PipelineManager::PipelineManager(const std::string& input_file_path, std::string model_type, bool profiling)
{
    this->input_file_path_ = input_file_path;
    this->model_type_ = model_type;
    this->is_profiling_ = profiling;

    model = std::make_unique<nn_compiler::ir::NNModel>();

    compiler = std::make_shared<NNCompiler>();
    compiler->initialize(input_file_path_, model_type_);
    compiler->compile(model);

    runtime = std::make_shared<NNRuntime>(model, model_type_);
}

std::vector<torch::Tensor> PipelineManager::inferenceModel(const std::vector<torch::Tensor>& input_tensors)
{
    outputs.clear();
    runtime->inferenceModel(input_tensors, outputs, is_profiling_);
    return outputs;
}

}  // namespace NNRuntimeInterface
