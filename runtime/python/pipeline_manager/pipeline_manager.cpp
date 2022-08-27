#include "compiler/include/nn_compiler.hpp"
#include "runtime/include/nn_runtime.h"

#include "pipeline_manager.h"
#include <thread>

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

    // runtime = std::make_shared<NNRuntime>(model, model_type_);
}

// static std::vector<torch::Tensor> output_tensors;

std::vector<torch::Tensor> launchInference(std::shared_ptr<nn_compiler::ir::NNModel> model_,
                            const std::vector<torch::Tensor>& input_tensor,std::string model_type_, bool profiling,int i){
    std::unique_ptr<nn_compiler::ir::NNModel> model = std::make_unique<nn_compiler::ir::NNModel>(*model_);
    std::vector<torch::Tensor> output_tensors;
    hipSetDevice(i);
    std::vector<torch::Tensor> input_tensors;
    for(auto item:input_tensor){
        input_tensors.emplace_back(item.cuda());
    }
    NNRuntime runtime_(model, model_type_);
    runtime_.inferenceModel(input_tensors, output_tensors, profiling);

}

std::vector<torch::Tensor> PipelineManager::inferenceModel(const std::vector<torch::Tensor>& input_tensors)
{
    outputs.clear();
    std::shared_ptr<nn_compiler::ir::NNModel> model_ = std::move(model);
    std::vector<std::thread> thread_pool;
    
    for(int i =0; i<8;i++){
        thread_pool.emplace_back(std::thread (&launchInference, model_, input_tensors, model_type_, is_profiling_,i));
    }
    for(auto &t:thread_pool){
        t.join();
    }
    
    return outputs;
}

}  // namespace NNRuntimeInterface
