#include "compiler/include/nn_compiler.hpp"

namespace nn_compiler {

RetVal NNCompiler::initialize(const int& compile_level, const std::string& file_path) {
    Log::NC::I() << "NNCompiler::initialize(compile_level, file_path) is called";

    compile_level_ = compile_level;
    input_file_path_ = file_path;

    //TODO: frontend initialize

    // middlend initialize
    middlend_driver_ = std::make_unique<middlend::MiddlendDriver>();

    //TODO: backend initialize

    return RetVal::SUCCESS;
}

RetVal NNCompiler::compile() {
    Log::NC::I() << "NNCompiler::compile(void) is called";

    switch(compile_level_) {
        case 0:
            frontend(input_file_path_);
            middlend();
            backend();
            break;
        case 1:
            middlend(input_file_path_);
            backend();
            break;
        case 2:
            backend(input_file_path_);
            break;
        default:
            return RetVal::FAILURE;
    }

    return RetVal::SUCCESS;
}

RetVal NNCompiler::compile(std::vector<std::shared_ptr<nn_compiler::nn_ir::NNIR>>& NNIR_graphs) {
    Log::NC::I() << "NNCompiler::compile() is called";

    switch(compile_level_) {
        case 0:
            frontend(input_file_path_);
            middlend();
            backend();
            break;
        case 1:
            middlend(input_file_path_);
            backend();
            break;
        case 2:
            backend(input_file_path_);
            break;
        default:
            return RetVal::FAILURE;
    }
    for (auto graph : NNIR_graphs_) {
        NNIR_graphs.push_back(graph);
    }

    return RetVal::SUCCESS;
}

RetVal NNCompiler::finalize() {
    Log::NC::I() << "NNCompiler::finalize(void) is called";

    return RetVal::SUCCESS;
}

RetVal NNCompiler::frontend(const std::string& file_path) {
    Log::NC::I() << "NNCompiler::frontend(file_path) is called";

    return RetVal::SUCCESS;
}

RetVal NNCompiler::middlend(const std::string& file_path) {
    Log::NC::I() << "NNCompiler::middlend(file_path) is called";

    middlend_driver_->initialize(file_path);
    middlend_driver_->build();
    middlend_driver_->run();
    middlend_driver_->wrapup();
    middlend_driver_->finalize();

    return RetVal::SUCCESS;
}

RetVal NNCompiler::middlend() {
    Log::NC::I() << "NNCompiler::middlend(void) is called";

    middlend_driver_->initialize(NNIR_graphs_);
    middlend_driver_->build();
    middlend_driver_->run();
    middlend_driver_->wrapup();
    middlend_driver_->finalize();

    return RetVal::SUCCESS;
}

RetVal NNCompiler::backend(const std::string& file_path) {
    Log::NC::I() << "NNCompiler::backend(file_path) is called";

    // TODO: backend pipeline

    return RetVal::SUCCESS;
}

RetVal NNCompiler::backend() {
    Log::NC::I() << "NNCompiler::backend(void) is called";

    // TODO: backend pipeline

    return RetVal::SUCCESS;
}

} //namespace nn_compiler
