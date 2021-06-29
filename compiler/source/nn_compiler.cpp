#include "compiler/include/nn_compiler.hpp"

namespace nn_compiler {

RetVal NNCompiler::initialize(const std::string& file_path) {
    Log::ME::I() << "NNCompiler::initialize() is called";

    input_file_path_ = file_path;

    //TODO: frontend initialize

    // middlend initialize
    middlend_driver_ = std::make_unique<middlend::MiddlendDriver>();

    //TODO: backend initialize

    return RetVal::SUCCESS;
}

RetVal NNCompiler::compile() {
    Log::ME::I() << "NNCompiler::compile() is called";

    // frontend();
    middlend(input_file_path_);
    // backend();

    return RetVal::SUCCESS;
}

RetVal NNCompiler::compile(std::vector<std::shared_ptr<const nn_compiler::nn_ir::NNIR>>& NNIR_graphs) {
    Log::ME::I() << "NNCompiler::compile() is called";

    // frontend();
    middlend(input_file_path_);
    // backend();

    for (auto graph : NNIR_graphs_) {
        NNIR_graphs.push_back(graph);
    }

    return RetVal::SUCCESS;
}

RetVal NNCompiler::finalize() {
    Log::ME::I() << "NNCompiler::finalize() is called";

    return RetVal::SUCCESS;
}

RetVal NNCompiler::frontend() {
    Log::ME::I() << "NNCompiler::frontend() is called";

    return RetVal::SUCCESS;
}

RetVal NNCompiler::middlend(const std::string& file_path) {
    Log::ME::I() << "NNCompiler::middlend() is called";

    middlend_driver_->initialize(file_path);
    middlend_driver_->build();
    middlend_driver_->run();
    middlend_driver_->wrapup();
    middlend_driver_->finalize();

    return RetVal::SUCCESS;
}

RetVal NNCompiler::backend() {
    Log::ME::I() << "NNCompiler::backend() is called";

    // TODO: backend pipeline

    return RetVal::SUCCESS;
}

} //namespace nn_compiler
