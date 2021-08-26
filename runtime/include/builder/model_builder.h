#pragma once

#include "ir/include/nn_ir.hpp"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
class ModelBuilder
{
   public:
    ModelBuilder(std::string model_path)
    {
        this->model_path_ = model_path;
        this->runnable_ir_ = nullptr;
    }

    // compiler intput model_path; output NNIR(LLO)
    RetVal compileModel(int compile_level, const std::string model_type);

    RetVal preloadModel();

    std::shared_ptr<nncir::NNIR> get_runnable_ir() { return this->runnable_ir_; }

   private:
    // Runnable NNIR
    std::shared_ptr<nncir::NNIR> runnable_ir_;

    std::string model_path_;
};

}  // namespace nnrt
