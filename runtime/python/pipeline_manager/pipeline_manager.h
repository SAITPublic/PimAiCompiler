#pragma once

#include <torch/script.h>

#include <experimental/filesystem>

#include "compiler/include/nn_compiler.hpp"
#include "ir/include/nn_model.h"
#include "runtime/include/nn_runtime.h"

using namespace nn_compiler::compiler;
using namespace nn_compiler::runtime;
namespace fs = std::experimental::filesystem;

namespace NNRuntimeInterface
{
class PipelineManager
{
   public:
    public:
    struct threadParam {
        bool is_enable;
        std::vector<torch::Tensor> input_tensor;
    };
    PipelineManager() = default;
    PipelineManager(const std::string& input_file_path, std::string model_type, bool profiling = false,
                    int gpu_num = 1);
    std::vector<torch::Tensor> inferenceModel(const std::vector<torch::Tensor>& input_tensors);
    // void* _launchRuntime(std::shared_ptr<nn_compiler::ir::NNModel> model_,
    //                         const std::vector<torch::Tensor>& input_tensor,  bool profiling);

    /**
     * @brief   Destroy all data and terminate the program
     * @details This function destroies all remained data and releases allocated memories
     * @returns return code
     */

   private:
    std::string input_file_path_ = "";

    std::string model_type_;

    bool is_profiling_ = false;

    std::unique_ptr<nn_compiler::ir::NNModel> model = nullptr;
    std::shared_ptr<NNCompiler> compiler = nullptr;
    std::shared_ptr<NNRuntime> runtime = nullptr;
    std::vector<torch::Tensor> outputs;

    std::vector<std::pair<std::thread, threadParam*>> thread_pool_;
    threadParam param_;
    static int finally_nums;
    static std::vector<torch::Tensor> output_tensors_;
    static void launchInference(std::shared_ptr<nn_compiler::ir::NNModel> model_,
                                PipelineManager::threadParam* thread_param, std::string model_type_, bool profiling,
                                int gpu_id);

};  // class PipelineManager

}  // namespace NNRuntimeInterface
