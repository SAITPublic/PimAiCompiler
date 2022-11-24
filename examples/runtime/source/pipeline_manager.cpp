/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#include "compiler/include/nn_compiler.hpp"
#include "ir/include/nn_model.h"
#include "runtime/include/nn_runtime.h"
#include "runtime/include/utils/tv_tools.h"
#include "runtime/include/utils/utils.h"

#include "pipeline_manager.hpp"

using namespace nn_compiler::compiler;
using namespace nn_compiler::runtime;
using namespace nn_compiler::runtime::utils;
namespace fs = std::experimental::filesystem;

namespace examples
{
RetVal PipelineManager::initialize(const std::string& input_file, const std::string& model_type, const int& gpu_num,
                                   const bool& profiling)
{
    if (model_type == "RNNT") {
        model_type_ = ModelType::RNNT;
    } else if (model_type == "GNMT") {
        model_type_ = ModelType::GNMT;
    } else if (model_type == "HWR") {
        model_type_ = ModelType::HWR;
    } else if (model_type == "Transformer") {
        model_type_ = ModelType::Transformer;
    } else if (model_type == "SwitchTransformer") {
        model_type_ = ModelType::SwitchTransformer;
    } else if (model_type == "PRMOE") {
        model_type_ = ModelType::PRMOE;
    } else {
        DLOG(FATAL) << "Unsupported model type.";
    }
    str_model_type_ = model_type;
    input_file_path_ = input_file;
    is_profiling_ = profiling;
    gpu_num_ = gpu_num;

    return RetVal::SUCCESS;
}

RetVal PipelineManager::run()
{
    switch (model_type_) {
        case ModelType::RNNT:
            load_and_run_rnnt();
            break;
        case ModelType::GNMT:
            load_and_run_gnmt();
            break;
        case ModelType::HWR:
            load_and_run_hwr();
            break;
        case ModelType::Transformer:
            load_and_run_transformer();
            break;
        case ModelType::PRMOE:
        case ModelType::SwitchTransformer:
            load_and_run_switchtransformer();
            break;
        default:
            break;
    }

    return RetVal::SUCCESS;
}

RetVal PipelineManager::finalize() { return RetVal::SUCCESS; }

void PipelineManager::load_and_run_rnnt()
{
    std::string feature_len_file = "examples/runtime/resource/rnnt/inputs/feature_len.bin";
    std::string feature_file = "examples/runtime/resource/rnnt/inputs/feature.bin";
    std::string current_path = fs::current_path();
    if (current_path.find("/build") != std::string::npos) {
        feature_len_file = "../" + feature_len_file;
        feature_file = "../" + feature_file;
    }
    if (access(feature_len_file.c_str(), F_OK) == -1 || access(feature_file.c_str(), F_OK) == -1) {
        DLOG(FATAL) << "Please run at base or build directory.";
    }

    std::unique_ptr<nn_compiler::ir::NNModel> model = std::make_unique<nn_compiler::ir::NNModel>();

    NNCompiler compiler;
    compiler.initialize(input_file_path_, str_model_type_);
    compiler.compile(model);

    NNRuntime runtime(model, str_model_type_);

    std::vector<torch::Tensor> input_tensors;
    std::vector<torch::Tensor> output_tensors;
    // load inputs from files
    auto tensor_feature = loadTensor(feature_file, {341, 1, 240}, DataType::FLOAT16).cuda();
    auto tensor_feature_len = loadTensor(feature_len_file, {1}, DataType::INT64);
    input_tensors.push_back(tensor_feature);
    input_tensors.push_back(tensor_feature_len);

    for (auto item : input_tensors) {
        DLOG(INFO) << "Input: "
                   << "size: " << item.sizes() << " dtype:" << item.dtype() << " device:" << item.device();
    }
    // Inference
    runtime.inferenceModel(input_tensors, output_tensors, is_profiling_);
}

void PipelineManager::load_and_run_gnmt()
{
    std::string src_file = "examples/runtime/resource/gnmt/inputs/src_1_12_torch.cuda.LongTensor.bin";
    std::string src_length_file = "examples/runtime/resource/gnmt/inputs/src_length_1_torch.cuda.LongTensor.bin";
    std::string bos_file = "examples/runtime/resource/gnmt/inputs/bos_1_1_torch.cuda.LongTensor.bin";
    std::string current_path = fs::current_path();
    if (current_path.find("/build") != std::string::npos) {
        src_file = "../" + src_file;
        src_length_file = "../" + src_length_file;
        bos_file = "../" + bos_file;
    }
    if (access(src_file.c_str(), F_OK) == -1 || access(src_length_file.c_str(), F_OK) == -1 ||
        access(bos_file.c_str(), F_OK) == -1) {
        DLOG(FATAL) << "Please run at base or build directory.";
    }

    std::unique_ptr<nn_compiler::ir::NNModel> model = std::make_unique<nn_compiler::ir::NNModel>();

    NNCompiler compiler;
    compiler.initialize(input_file_path_, str_model_type_);
    compiler.compile(model);

    NNRuntime runtime(model, str_model_type_);

    std::vector<torch::Tensor> input_tensors;
    std::vector<torch::Tensor> output_tensors;
    // load inputs from files
    auto tensor_src = loadTensor(src_file, {1, 12}, DataType::INT64).cuda();
    auto tensor_src_length = loadTensor(src_length_file, {1}, DataType::INT64).cuda();
    auto tensor_bos = loadTensor(bos_file, {1, 1}, DataType::INT64).cuda();

    input_tensors.push_back(tensor_src);
    input_tensors.push_back(tensor_src_length);
    input_tensors.push_back(tensor_bos);

    for (auto item : input_tensors) {
        DLOG(INFO) << "Input: "
                   << "size: " << item.sizes() << " dtype:" << item.dtype() << " device:" << item.device();
    }
    // Inference
    runtime.inferenceModel(input_tensors, output_tensors, is_profiling_);
}

void PipelineManager::load_and_run_hwr()
{
    std::string input_file = "./examples/runtime/resource/hwr/inputs/input_hwr_1_1_1024_128.bin";
    std::string current_path = fs::current_path();
    if (current_path.find("/build") != std::string::npos) {
        input_file = "../" + input_file;
    }
    if (access(input_file.c_str(), F_OK) == -1) {
        DLOG(FATAL) << "Please run at base or build directory.";
    }

    std::unique_ptr<nn_compiler::ir::NNModel> model = std::make_unique<nn_compiler::ir::NNModel>();

    NNCompiler compiler;
    compiler.initialize(input_file_path_, str_model_type_);
    compiler.compile(model);

    NNRuntime runtime(model, str_model_type_);

    std::vector<torch::Tensor> input_tensors;
    std::vector<torch::Tensor> output_tensors;
    // load inputs from files
    auto tensor = loadTensor(input_file, {1, 1, 1024, 128}, DataType::FLOAT16).cuda();
    input_tensors.push_back(tensor);

    for (auto item : input_tensors) {
        DLOG(INFO) << "Input: "
                   << "size: " << item.sizes() << " dtype:" << item.dtype() << " device:" << item.device();
    }
    // Inference
    runtime.inferenceModel(input_tensors, output_tensors, is_profiling_);
}

void PipelineManager::load_and_run_transformer()
{
    std::string input_file = "./examples/runtime/resource/transformer/inputs/transformer.bin";
    std::string current_path = fs::current_path();
    if (current_path.find("/build") != std::string::npos) {
        input_file = "../" + input_file;
    }
    if (access(input_file.c_str(), F_OK) == -1) {
        DLOG(FATAL) << "Please run at base or build directory.";
    }

    std::unique_ptr<nn_compiler::ir::NNModel> model = std::make_unique<nn_compiler::ir::NNModel>();

    NNCompiler compiler;
    compiler.initialize(input_file_path_, str_model_type_);
    compiler.compile(model);

    NNRuntime runtime(model, str_model_type_);

    std::vector<torch::Tensor> input_tensors;
    std::vector<torch::Tensor> output_tensors;
    // load inputs from files
    auto tensor = loadTensor(input_file, {1, 11}, DataType::INT64).cuda();
    input_tensors.push_back(tensor);

    for (auto item : input_tensors) {
        DLOG(INFO) << "Input: "
                   << "size: " << item.sizes() << " dtype:" << item.dtype() << " device:" << item.device();
    }

    // Inference
    runtime.inferenceModel(input_tensors, output_tensors, is_profiling_);
}

static std::vector<torch::Tensor> output_tensors_;

static void launchInference(std::shared_ptr<nn_compiler::ir::NNModel> model_,
                            const std::vector<torch::Tensor>& input_tensors, std::string model_type_, bool profiling,
                            int gpu_id)
{
    hipSetDevice(gpu_id);
    std::unique_ptr<nn_compiler::ir::NNModel> model = std::make_unique<nn_compiler::ir::NNModel>(*model_);
    std::shared_ptr<NNRuntime> runtime_ = std::make_shared<NNRuntime>(model, model_type_);
    std::vector<torch::Tensor> input_tensors_;
    for (auto item : input_tensors) {
        input_tensors_.emplace_back(item.cuda());
    }

    runtime_->inferenceModel(input_tensors_, output_tensors_, profiling);
}

void PipelineManager::load_and_run_switchtransformer()
{
    std::string input_file = "examples/runtime/resource/switch_transformer/inputs/src_2_13_input.bin";
    std::string src_attention_mask = "examples/runtime/resource/switch_transformer/inputs/src_2_13_attention_mask.bin";
    std::string current_path = fs::current_path();
    if (current_path.find("/build") != std::string::npos) {
        input_file = "../" + input_file;
        src_attention_mask = "../" + src_attention_mask;
    }
    if (access(input_file.c_str(), F_OK) == -1) {
        DLOG(FATAL) << "Please run at base or build directory.";
    }

    std::vector<std::thread> thread_pool_;
    // load inputs from files
    auto input_tensor = loadTensor(input_file, {2, 13}, DataType::INT64);
    auto attention_mask = loadTensor(src_attention_mask, {2, 13}, DataType::INT64);

    std::unique_ptr<nn_compiler::ir::NNModel> model = std::make_unique<nn_compiler::ir::NNModel>();

    NNCompiler compiler;
    compiler.initialize(input_file_path_, str_model_type_);
    compiler.compile(model);
    std::shared_ptr<nn_compiler::ir::NNModel> model_ = std::move(model);

    int input_num = 0;
    if (input_tensor.size(0) > gpu_num_) {
        input_num = gpu_num_;
    } else {
        input_num = input_tensor.size(0);
    }

    std::vector<torch::Tensor> input_chunks = input_tensor.chunk(input_num, 0);
    std::vector<torch::Tensor> attention_chunks = attention_mask.chunk(input_num, 0);

    for (int idx = 0; idx < input_chunks.size(); idx++) {
        auto input = input_chunks[idx];
        auto attention_mask = attention_chunks[idx];
        std::vector<torch::Tensor> input_tensors;
        input_tensors.push_back(input);
        input_tensors.push_back(attention_mask);
        thread_pool_.emplace_back(
            std::thread(&launchInference, model_, input_tensors, str_model_type_, is_profiling_, idx));
    }
    for (auto& t : thread_pool_) {
        t.join();
    }

    torch::Tensor total_tensor = at::cat(output_tensors_, 0);
    DLOG(INFO) << total_tensor;
}

}  // namespace examples
