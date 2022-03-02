#include <sys/time.h>

#include "c10/hip/HIPFunctions.h"
#include "common/log.hpp"
#include "new_runtime/include/executor/stream_executor.h"
#include "new_runtime/include/executor/utils.h"
#include "types.h"

namespace nn_compiler {
namespace runtime {
StreamExecutor::StreamExecutor(blob_store_type pre_loaded_data, std::string model_type){
    global_blobs_ = pre_loaded_data;
    model_type_ = model_type;

    this->registerOp();
}

StreamExecutor::~StreamExecutor() {
}

RetVal StreamExecutor::inferenceModel(std::unique_ptr<nn_compiler::ir::NNModel> &model,
                                      const std::vector<torch::Tensor>& input_tensors,
                                      std::vector<torch::Tensor>& output_tensors) {

    return RetVal::SUCCESS;
}

RetVal StreamExecutor::inferenceModelwithProfiling(std::unique_ptr<nn_compiler::ir::NNModel> &model,
                                                   const std::vector<torch::Tensor>& input_tensors,
                                                   std::vector<torch::Tensor>& output_tensors) {

    return RetVal::SUCCESS;
}

void StreamExecutor::updateBlob(int64_t blob_id, DataType dtype, const torch::jit::IValue& iv) {
}

std::pair<DataType, torch::jit::IValue>& StreamExecutor::findBlob(int64_t blob_id) {
}

void StreamExecutor::setInputTensors(const std::vector<torch::Tensor>& input_tensors) {
}

std::vector<torch::Tensor> StreamExecutor::iValueParser(torch::jit::IValue& iv) {
    std::vector<torch::Tensor> out_tensor;
    std::vector<int64_t> out_list;
    
    // TODO(SRCX): implementation

    return out_tensor;
}

void StreamExecutor::getOutputTensors(std::vector<torch::Tensor>& output_tensors) {
}

void StreamExecutor::registerOp() {
}

}  // namespace runtime
}  // namespace nn_compiler
