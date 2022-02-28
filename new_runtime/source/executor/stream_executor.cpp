#include <sys/time.h>

#include "c10/hip/HIPFunctions.h"
#include "executor/stream_executor.h"
#include "executor/utils.h"
#include "types.h"
#include "tv_tools.h"

namespace nn_compiler {
namespace runtime {
StreamExecutor::StreamExecutor(blob_store_type pre_loaded_data, std::string model_type){
    global_blobs_ = pre_loaded_data;
    model_type_ = model_type;

    this->registerOp();
}

StreamExecutor::~StreamExecutor() {
}

RetVal StreamExecutor::inferenceModel(std::unique_ptr<nn_compiler::ir::NNModel &model,
                                      const std::vector<torch::Tensor>& input_tensors,
                                      std::vector<torch::Tensor>& output_tensors) {

    return RetVal::SUCCESS;
}

RetVal StreamExecutor::inferenceModelwithProfiling(std::unique_ptr<nn_compiler::ir::NNModel &model,
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
    if (iv.isTuple()) {
        auto tuple_ = iv.toTuple();
        auto ivs = primTupleUnpack(tuple_);
        for (auto iv_ : ivs) {
            if (iv_.isTensor()) {
                out_tensor.push_back(iv_.toTensor());
            } else if (iv_.isList()) {
                auto temp_out = iValueParser(iv_);
                for (auto& out : temp_out) {
                    out_tensor.push_back(out);
                }
            } else if (iv_.isInt()) {
                auto temp_out = iValueParser(iv_);
                for (auto& out : temp_out) {
                    out_tensor.push_back(out);
                }
            } else {
                LOG(FATAL) << "Dtype of output is unsupported !";
            }
        }
    } else if (iv.isList()) {
        auto list_ = iv.toListRef();
        out_list.clear();
        for (auto iv_ : list_) {
            if (iv_.isTensor()) {
                out_tensor.push_back(iv_.toTensor());
            } else if (iv_.isInt()) {
                out_list.push_back(iv_.toInt());
            } else if (iv_.isList()) {
                auto temp_out = iValueParser(iv_);
                for (auto& out : temp_out) {
                    out_tensor.push_back(out);
                }
            } else {
                LOG(FATAL) << "Dtype of output is unsupported !";
            }
        }
        if (out_list.size() != 0) {
            torch::Tensor out_ =
                torch::from_blob(out_list.data(), {1, static_cast<int64_t>(out_list.size())}, torch::kLong).clone();
            out_tensor.push_back(std::move(out_));
        }
    } else if (iv.isInt()) {
        out_list.push_back(iv.toInt());
        torch::Tensor out_ =
            torch::from_blob(out_list.data(), {static_cast<int64_t>(out_list.size())}, torch::kLong).clone();
        out_tensor.push_back(std::move(out_));
    } else if (iv.isTensor()) {
        out_tensor.push_back(iv.toTensor());
    } else {
        LOG(FATAL) << "Dtype of output is unsupported !";
    }
    return out_tensor;
}

void StreamExecutor::getOutputTensors(std::vector<torch::Tensor>& output_tensors) {
}

void StreamExecutor::registerOp() {
}

}  // namespace runtime
}  // namespace nn_compiler
