#include "compiler/include/nn_compiler.hpp"
#include "runtime/include/nn_runtime.h"

#include <thread>
#include "pipeline_manager.h"

using namespace nn_compiler::compiler;
using namespace nn_compiler::runtime;
namespace fs = std::experimental::filesystem;

namespace NNRuntimeInterface
{
int PipelineManager::finally_nums = 0;
std::vector<torch::Tensor> PipelineManager::output_tensors_;

void PipelineManager::launchInference(std::shared_ptr<nn_compiler::ir::NNModel> model_,
                                      PipelineManager::threadParam* thread_param, std::string model_type_,
                                      bool profiling, int gpu_id)
{
    std::unique_ptr<nn_compiler::ir::NNModel> model = std::make_unique<nn_compiler::ir::NNModel>(*model_);
    bool created_runtime = false;
    std::shared_ptr<NNRuntime> runtime_ = NULL;
    hipSetDevice(gpu_id);
    while (thread_param->is_running) {
        sleep(0);
        if (!created_runtime) {
            runtime_ = std::make_shared<NNRuntime>(model, model_type_);
            created_runtime = true;
        }
        if (thread_param->is_enable) {
            std::vector<torch::Tensor> input_tensors;
            for (auto item : thread_param->input_tensor) {
                input_tensors.emplace_back(item.cuda());
            }

            runtime_->inferenceModel(input_tensors, output_tensors_, profiling);
            thread_param->is_enable = false;
            finally_nums++;
        }
    }
}

PipelineManager::PipelineManager(const std::string& input_file_path, std::string model_type, int gpu_num,
                                 bool profiling)
{
    this->input_file_path_ = input_file_path;
    this->model_type_ = model_type;
    this->is_profiling_ = profiling;

    model = std::make_unique<nn_compiler::ir::NNModel>();

    compiler = std::make_shared<NNCompiler>();
    compiler->initialize(input_file_path_, model_type_);
    compiler->compile(model);
    if (model_type_ == "SwitchTransformer") {
        finally_nums = 0;
        std::shared_ptr<nn_compiler::ir::NNModel> model_ = std::move(model);
        for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
            PipelineManager::threadParam* thread_param = new PipelineManager::threadParam();
            thread_param->is_running = true;
            thread_param->is_enable = false;
            thread_pool_.emplace_back(std::make_pair(
                std::thread(&launchInference, model_, thread_param, model_type_, is_profiling_, gpu_id), thread_param));
        }
    }else{
        runtime = std::make_shared<NNRuntime>(model, model_type_);
    }
}

std::vector<torch::Tensor> PipelineManager::inferenceModel(const std::vector<torch::Tensor>& input_tensors)
{
    outputs.clear();
    output_tensors_.clear();
    if (model_type_ == "SwitchTransformer") {
        int input_num = 0;
        if (input_tensors[0].size(0) > thread_pool_.size()) {
            input_num = thread_pool_.size();
        } else {
            input_num = input_tensors[0].size(0);
        }

        std::vector<torch::Tensor> input_chunks = input_tensors[0].chunk(input_num, 0);
        std::vector<torch::Tensor> attention_chunks = input_tensors[1].chunk(input_num, 0);

        std::vector<std::vector<torch::Tensor>> total_input_tensors;

        for (int idx = 0; idx < input_chunks.size(); idx++) {
            auto input = input_chunks[idx];
            auto attention_mask = attention_chunks[idx];
            std::vector<torch::Tensor> input_tensors;
            input_tensors.push_back(input);
            input_tensors.push_back(attention_mask);
            total_input_tensors.push_back(input_tensors);
        }

        finally_nums = 0;
        for (int i = 0; i < input_chunks.size(); i++) {
            thread_pool_[i].second->input_tensor = total_input_tensors[i];
            thread_pool_[i].second->is_enable = true;
        }

        while (true) {
            sleep(0);
            if (finally_nums == input_chunks.size()) break;
        }
        torch::Tensor total_tensor = at::cat(output_tensors_, 0);
        outputs.push_back(total_tensor);

        return outputs;
    } else {
        runtime->inferenceModel(input_tensors, outputs, is_profiling_);
        return outputs;
    }
}

}  // namespace NNRuntimeInterface
