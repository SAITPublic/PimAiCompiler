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
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr =true;
    FLAGS_colorlogtostderr =true; 
    LOG(INFO) << "Hello,info! ";
    NNRuntime runtime("/home/rnnt.torchscript");

    runtime.test(); 
    NnrtBuffer *inputBuffer = nullptr;
    NnrtBuffer *outputBuffer = nullptr;
    
    runtime.inferenceModel(inputBuffer, outputBuffer);

    google::ShutdownGoogleLogging();
    return 0;
}


