#pragma once

#include <torch/script.h>

#include "common/include/cast.hpp"
#include "ir/include/all_nodes.hpp"
#include "ir/include/data_blob.hpp"
#include "ir/include/ir_types.hpp"
#include "ir/include/nn_ir.hpp"
#include "nnrt_types.h"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
class ModelBuilder
{
   public:
    typedef std::unordered_map<int64_t, std::pair<nnrt::DataType, torch::jit::IValue>> blob_store_type;

    ModelBuilder(std::string model_path)
    {
        this->model_path_ = model_path;
        this->runnable_ir_ = nullptr;
    }

    // compiler intput model_path; output NNIR(LLO)
    RetVal compileModel(int compile_level, const std::string model_type);

    RetVal preloadModel();

    void loadWeightAndBias(nncir::Blob* blob);

    std::pair<std::shared_ptr<nncir::NNIR>, blob_store_type> getModel();

   private:
    std::shared_ptr<nncir::NNIR> runnable_ir_;

    blob_store_type preloaded_blobs_container_;

    std::string model_path_;
};

}  // namespace nnrt
