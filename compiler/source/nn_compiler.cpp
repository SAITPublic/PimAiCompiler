#include "compiler/include/nn_compiler.hpp"

namespace nn_compiler
{
namespace compiler
{
RetVal NNCompiler::initialize(const std::string& file_path, const std::string& model_name)
{
    Log::NC::I() << "NNCompiler::initialize(compile_level, file_path) is called";

    input_file_path_ = file_path;
    model_name_ = model_name;

    // frontend initialize
    frontend_driver_ = std::make_unique<frontend::FrontendDriver>();
    // middlend initialize
    middlend_driver_ = std::make_unique<middlend::MiddlendDriver>();

    // TODO: backend initialize

    return RetVal::SUCCESS;
}

RetVal NNCompiler::compile(std::unique_ptr<ir::NNModel>& model)
{
    Log::NC::I() << "NNCompiler::compile() is called";

    frontend(input_file_path_, model_name_, model);
    middlend(model);
    backend();

    return RetVal::SUCCESS;
}

RetVal NNCompiler::finalize()
{
    Log::NC::I() << "NNCompiler::finalize(void) is called";

    return RetVal::SUCCESS;
}

RetVal NNCompiler::frontend(const std::string& file_path, const std::string& model_name,
                            std::unique_ptr<ir::NNModel>& model)
{
    Log::NC::I() << "NNCompiler::frontend() is called";

    frontend_driver_->initialize(file_path, model_name);
    frontend_driver_->run(model);
    frontend_driver_->finalize();

    return RetVal::SUCCESS;
}

RetVal NNCompiler::middlend(std::unique_ptr<ir::NNModel>& model)
{
    Log::NC::I() << "NNCompiler::middlend() is called";

    middlend_driver_->initialize();
    middlend_driver_->run(model);
    middlend_driver_->finalize();

    return RetVal::SUCCESS;
}

RetVal NNCompiler::backend()
{
    Log::NC::I() << "NNCompiler::backend(void) is called";

    // TODO: backend pipeline

    return RetVal::SUCCESS;
}

}  // namespace compiler
}  // namespace nn_compiler
