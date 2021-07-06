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
    input_tensors.push_back(torch::tensor({{1, 3, 5}, {9, 7, 5}}));
    input_tensors.push_back(torch::tensor({{10, 30, 50}, {90, -20, 10}}));
    input_tensors.push_back(torch::tensor({18}));
    input_tensors.push_back(torch::tensor({7}));

    // Inference
    auto output_tensors = runtime.inferenceModel(input_tensors);
    return 0;
}


