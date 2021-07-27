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

using namespace nnrt;

void run_rnnt_from_file(std::string ir_file)
{
    std::string feature_len_file = "examples/runtime/resource/rnnt/inputs/feature_len.bin";
    std::string feature_file = "examples/runtime/resource/rnnt/inputs/feature.bin";
    std::string current_path = std::experimental::filesystem::current_path();
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

int main(int argc, const char* argv[])
{
    LOG(INFO) << "start runtime! ";
    auto print_error = []() {
        LOG(ERROR) << "Usage: ./simpleMian model_path! or set ENV (export GRAPH_IR_FILE=path/to/your/graph_ir)";
        return -1;
    };

    char* ir_file = nullptr;
    if (argc == 1) {
        ir_file = getenv("GRAPH_IR_FILE");
        if (ir_file == nullptr) {
            return print_error();
        }
    } else if (argc == 2) {
        ir_file = const_cast<char*>(argv[1]);
    } else {
        return print_error();
    }
    run_rnnt_from_file(ir_file);
    return 0;
}
