#include <glog/logging.h>
#include <cstdio>
#include <cstring>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include "nn_runtime.h"
#include "runtime/include/executor/prim_utils.h"
#include "runtime/include/tv_tools.h"

using namespace nnrt;
namespace fs = std::experimental::filesystem;

void run_rnnt_from_file(std::string ir_file)
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

    NNRuntime runtime(ir_file);
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

void run_hwr_from_file(std::string ir_file)
{
    std::string input_file = "./examples/runtime/resource/hwr/inputs/input_hwr_1_1_1024_128.bin";
    std::string current_path = fs::current_path();
    if (current_path.find("/build") != std::string::npos) {
        input_file = "../" + input_file;
    }
    if (access(input_file.c_str(), F_OK) == -1) {
        DLOG(ERROR) << "Please run at base or build directory.";
    }

    NNRuntime runtime(ir_file);
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

void run_gnmt_from_file(std::string ir_file)
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

    NNRuntime runtime(ir_file);
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
int main(int argc, const char* argv[])
{
    LOG(INFO) << "start runtime! ";
    auto print_error = []() {
        LOG(ERROR) << "Usage: ./simpleMian model_path! or set ENV (export GRAPH_IR_FILE=path/to/your/graph_ir; export "
                      "MODEL_KIND=RNNT/HWR/GNMT)";
        return -1;
    };

    char* ir_file = nullptr;
    char* model_kind = nullptr;
    if (argc == 1) {
        ir_file = getenv("GRAPH_IR_FILE");
        model_kind = getenv("MODEL_KIND");
        if (ir_file == nullptr || model_kind == nullptr) {
            return print_error();
        }
    } else if (argc == 3) {
        ir_file = const_cast<char*>(argv[1]);
        model_kind = const_cast<char*>(argv[2]);
    } else {
        return print_error();
    }

    if (std::string(model_kind) == "RNNT") {
        run_rnnt_from_file(ir_file);
    } else if (std::string(model_kind) == "HWR") {
        run_hwr_from_file(ir_file);
    } else if (std::string(model_kind) == "GNMT") {
        run_gnmt_from_file(ir_file);
    } else {
        DLOG(FATAL) << "Choice must be one of [RNNT, HWR, GNMT]";
    }

    return 0;
}
