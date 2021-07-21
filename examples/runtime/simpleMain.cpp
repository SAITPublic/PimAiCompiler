#include <cstdio>
#include <cstring>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <set>

#include "nn_runtime.h"
using namespace nnrt;

int main(int argc, const char* argv[]) {
    LOG(INFO) << "start runtime! ";

    auto print_error = [](){
        LOG(ERROR) << "Usage: ./simpleMian model_path! or set ENV (export GRAPH_IR_FILE=path/to/your/graph_ir)";
        return -1;
    };

    char* ir_file = nullptr;
    if(argc == 1) {
        ir_file = getenv("GRAPH_IR_FILE");
        if(ir_file == nullptr) {
            return print_error();
        }
    } else if (argc == 2) {
        ir_file = const_cast<char*>(argv[1]);
    } else {
         return print_error();
    }

    NNRuntime runtime(ir_file);
    runtime.test(); 
    
    std::vector<torch::Tensor> input_tensors;
    // Set test inputs
    // input_tensors.push_back(torch::tensor({{1, 2, 3}, {4, 5, 6}}));
    // input_tensors.push_back(torch::tensor({{20, 2, 30}, {-4, 50, 70}}));

    // For test RNNT with one sample
    input_tensors.push_back(torch::ones({341, 1, 240}, torch::kHalf));
    input_tensors.push_back(torch::tensor({341}, torch::kInt64));

    // Inference
    auto output_tensors = runtime.inferenceModel(input_tensors);
    return 0;
}


