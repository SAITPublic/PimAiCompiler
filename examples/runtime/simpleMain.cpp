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
    if (argc <2 ) {
        LOG(ERROR) << "Usage: ./simpleMian model_path!";
        return -1;

    }

    NNRuntime runtime(argv[1]);
    runtime.test(); 
    
    std::vector<torch::Tensor> input_tensors;
    // Set test inputs
    input_tensors.push_back(torch::tensor({{1, 2, 3}, {4, 5, 6}}));
    input_tensors.push_back(torch::tensor({{10, 20, 20}, {40, 50, 60}}));

    // Inference
    auto output_tensors = runtime.inferenceModel(input_tensors);
    return 0;
}


