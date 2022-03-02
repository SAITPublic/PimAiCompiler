#include "compiler/include/nn_compiler.hpp"

namespace nn_compiler {
namespace compiler {

RetVal NNCompiler::initialize(const int& compile_level, const std::string& file_path,
                              const std::string& model_name) {
    Log::NC::I() << "NNCompiler::initialize(compile_level, file_path) is called";

    compile_level_ = compile_level;
    input_file_path_ = file_path;
    model_name_ = model_name;

    // frontend initialize
    frontend_driver_ = std::make_unique<frontend::FrontendDriver>();
    // middlend initialize
    middlend_driver_ = std::make_unique<middlend::MiddlendDriver>();

    //TODO: backend initialize

    return RetVal::SUCCESS;
}

RetVal NNCompiler::compile() {
    Log::NC::I() << "NNCompiler::compile(void) is called";

    switch(compile_level_) {
        case 1:
            middlend(input_file_path_);
            backend();
            break;
        default:
            return RetVal::FAILURE;
    }

    return RetVal::SUCCESS;
}

RetVal NNCompiler::compile(std::unique_ptr<ir::NNModel>& model) {
    Log::NC::I() << "NNCompiler::compile() is called";

    frontend(input_file_path_, model_name_, model);
    middlend(model);
    backend();

    return RetVal::SUCCESS;
}

RetVal NNCompiler::compile(std::vector<std::shared_ptr<nn_compiler::nn_ir::NNIR>>& NNIR_graphs) {
    Log::NC::I() << "NNCompiler::compile() is called";

    switch(compile_level_) {
        case 1:
            middlend(input_file_path_);
            backend();
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

RetVal NNCompiler::frontend(const std::string& file_path, const std::string& model_name,
                            std::unique_ptr<ir::NNModel>& model) {
    Log::NC::I() << "NNCompiler::frontend() is called";

    frontend_driver_->initialize(file_path, model_name);
    frontend_driver_->run(model);
    frontend_driver_->finalize();

    return RetVal::SUCCESS;
}

RetVal NNCompiler::middlend(const std::string& file_path) {
    Log::NC::I() << "NNCompiler::middlend(file_path) is called";

    middlend_driver_->initialize(file_path);
    middlend_driver_->build();
    middlend_driver_->run();
    middlend_driver_->wrapup(NNIR_graphs_);
    middlend_driver_->finalize();

    return RetVal::SUCCESS;
}

RetVal NNCompiler::middlend(std::unique_ptr<ir::NNModel>& model) {
    Log::NC::I() << "NNCompiler::middlend() is called";

    // TODO(SRCX): implementation

    return RetVal::SUCCESS;
}

RetVal NNCompiler::backend() {
    Log::NC::I() << "NNCompiler::backend(void) is called";

    // TODO: backend pipeline

    return RetVal::SUCCESS;
}

} // namespace compiler
} //namespace nn_compiler
