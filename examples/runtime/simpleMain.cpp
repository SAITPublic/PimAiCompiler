#include <cstdio>
#include <cstring>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <set>

#include "nn_runtime.h"
using namespace nnr;

int main(int argc, const char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr =true;
    FLAGS_colorlogtostderr =true; 
    LOG(INFO) << "Hello,info! ";
    NNRuntime rnnt_runtime("/home/rnnt.torchscript");

    rnnt_runtime.test(); 
    NnrBuffer *inputBuffer = nullptr;
    NnrBuffer *outputBuffer = nullptr;
    
    rnnt_runtime.inferenceModel(inputBuffer, outputBuffer);

    google::ShutdownGoogleLogging();
    return 0;
}


