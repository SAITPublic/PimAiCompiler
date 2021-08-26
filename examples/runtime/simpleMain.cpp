#include <glog/logging.h>
#include <cstdio>
#include <cstring>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>

#include "common/include/command_line_parser.hpp"
#include "nn_runtime.h"
#include "runtime/include/executor/prim_utils.h"
#include "runtime/include/tv_tools.h"

using namespace nnrt;
namespace fs = std::experimental::filesystem;

static cl_opt::Option<std::string>
input_file_path_option(std::vector<std::string>{"-i", "--input"}, "<file>", "Input file path", cl_opt::Required::YES);

static cl_opt::Option<std::string>
model_type_option(std::vector<std::string>{"-m", "--model"}, "<model type>", "Possible model type: RNNT/HWR/GNMT", cl_opt::Required::YES);

static cl_opt::Option<int> compile_level_option("-l",
                                                "<compile level>",
                                                "compile level. Possible values: 0 (frontend->middlend->backend),\n\
                                                 1 (middlend->backend), 2 (backend)",
                                                cl_opt::Required::YES);

void run_rnnt_from_file(std::string ir_file, int compile_level, std::string model_type)
{
    std::string feature_len_file = "examples/runtime/resource/rnnt/inputs/feature_len.bin";
    std::string feature_file = "examples/runtime/resource/rnnt/inputs/feature.bin";
    std::string current_path = fs::current_path();
    if (current_path.find("/build") != std::string::npos) {
        feature_len_file = "../" + feature_len_file;
        feature_file = "../" + feature_file;
    }
    if (access(feature_len_file.c_str(), F_OK) == -1 || access(feature_file.c_str(), F_OK) == -1) {
        DLOG(ERROR) << "Please run at base or build directory.";
    }

    NNRuntime runtime(ir_file, compile_level, model_type);
    std::vector<torch::Tensor> input_tensors;
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
    auto output_tensors = runtime.inferenceModel(input_tensors);
}

void run_hwr_from_file(std::string ir_file, int compile_level, std::string model_type)
{
    std::string input_file = "./examples/runtime/resource/hwr/inputs/input_hwr_1_1_1024_128.bin";
    std::string current_path = fs::current_path();
    if (current_path.find("/build") != std::string::npos) {
        input_file = "../" + input_file;
    }
    if (access(input_file.c_str(), F_OK) == -1) {
        DLOG(ERROR) << "Please run at base or build directory.";
    }

    NNRuntime runtime(ir_file, compile_level, model_type);
    std::vector<torch::Tensor> input_tensors;
    // load inputs from files
    auto tensor_ = loadTensor(input_file, {1, 1, 1024, 128}, DataType::FLOAT16).cuda();
    input_tensors.push_back(tensor_);

    for (auto item : input_tensors) {
        DLOG(INFO) << "Input: "
                   << "size: " << item.sizes() << " dtype:" << item.dtype() << " device:" << item.device();
    }
    // Inference
    auto output_tensors = runtime.inferenceModel(input_tensors);
    // check outputs
    TVComparator& tv_comp = TVComparator::getInstance();
    tv_comp.loadTV("./examples/runtime/resource/hwr/inputs/output_hwr_y_hat_128_1_98.bin", {128, 1, 98},
                   DataType::FLOAT16, "hwr_final_output");
    if (tv_comp.compare(output_tensors.at(0).cpu(), "hwr_final_output")) {
        LOG(INFO) << "HWR run successfully, output is correct !";
    } else {
        LOG(INFO) << "HWR run failed, output is incorrect !";
    }
}

void run_gnmt_from_file(std::string ir_file, int compile_level, std::string model_type)
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
    if (access(src_file.c_str(), F_OK) == -1 || access(src_length_file.c_str(), F_OK) == -1 || access(bos_file.c_str(), F_OK) == -1) {
        DLOG(ERROR) << "Please run at base or build directory.";
    }

    NNRuntime runtime(ir_file, compile_level, model_type);
    std::vector<torch::Tensor> input_tensors;
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
    auto output_tensors = runtime.inferenceModel(input_tensors);
}

int main(int argc, char* argv[])
{
    cl_opt::CommandLineParser::getInstance().parseCommandLine(argc, argv);

    auto input_file_path = static_cast<std::string>(input_file_path_option);
    auto model_type = static_cast<std::string>(model_type_option);
    auto compile_level = static_cast<int>(compile_level_option);

    if (model_type == "RNNT") {
        run_rnnt_from_file(input_file_path, compile_level, model_type);
    } else if (model_type == "HWR") {
        run_hwr_from_file(input_file_path, compile_level, model_type);
    } else if (model_type == "GNMT") {
        run_gnmt_from_file(input_file_path, compile_level, model_type);
    } else {
        DLOG(FATAL) << "Choice must be one of [RNNT, HWR, GNMT]";
    }

    return 0;
}
