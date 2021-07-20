#include <cstdio>
#include <cstring>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <set>
#include "runtime/include/executor/prim_utils.h"
#include "nn_runtime.h"

using namespace nnrt;


torch::Tensor load_tensor(const std::string& bin_file, const std::vector<int64_t>& shape, DataType dtype) {
    // ref: // https://www.cnblogs.com/yaos/p/12105108.html
    // Get num-elements
    int num_elements = 1;
    for(auto item : shape) {
        num_elements *= item;
    }

    int total_size = -1;
    if(dtype == DataType::INT64){
        total_size = num_elements * sizeof(int64_t);
    } else if(dtype == DataType::FLOAT32) {
        total_size = num_elements * sizeof(float);
    } else if(dtype == DataType::FLOAT16) {
        total_size = num_elements * sizeof(float) / 2;
    } else {
        DLOG(ERROR) <<"Unsupport data type!";
    }
    char* buffer = new char[total_size];
    // read binary file
    std::ifstream ifs(bin_file, std::ios::binary | std::ios::in);
    ifs.read(buffer, total_size);
    ifs.close();

    auto tensor = createPtTensor((void*)buffer, shape, dtype);
    return tensor; 
}


void run_rnnt_from_file(std::string ir_file) {

    std::string feature_len_file = "path/to/feature_len.bin";
    std::string feature_file = "path/to/feature.bin";
    NNRuntime runtime(ir_file);
    // runtime.test(); 
    
    std::vector<torch::Tensor> input_tensors;

    // load inputs from files
    auto tensor_feature = load_tensor(feature_file, {341, 1, 240}, DataType::FLOAT16);
    auto tensor_feature_len = load_tensor(feature_len_file, {1}, DataType::INT64);
    input_tensors.push_back(tensor_feature);
    input_tensors.push_back(tensor_feature_len);

    for(auto item : input_tensors) {
        DLOG(INFO) <<"Input: " << "size: " <<item.sizes() <<" dtype:" <<item.dtype() <<" device:" <<item.device();
    }

    std::cout<< tensor_feature_len <<std::endl;
    std::cout<< tensor_feature[0] <<std::endl;

    // Inference
    auto output_tensors = runtime.inferenceModel(input_tensors);
}

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
    // For test RNNT with one sample
    input_tensors.push_back(torch::ones({341, 1, 240}, torch::kHalf));
    input_tensors.push_back(torch::tensor({341}, torch::kInt64));

    // Inference
    auto output_tensors = runtime.inferenceModel(input_tensors);
    return 0;
}
