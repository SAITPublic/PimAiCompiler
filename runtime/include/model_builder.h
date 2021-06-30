#pragma once

#include "ir/include/nn_ir.hpp"


namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
class ModelBuilder
{
   public:
    ModelBuilder(std::string model_path) {
        this->model_path = model_path;
        this->runnable_ir = nullptr;
    }

    // compiler intput model_path; output NNIR(LLO)
    RetVal compileModel();

    RetVal preloadModel();

    // Runnable NNIR
    std::shared_ptr<nncir::NNIR> runnable_ir;
   private:
    std::string model_path;

};

}  // namespace nnrt
