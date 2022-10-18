#pragma once

#include <miopen/miopen.h>
#include <torch/script.h>

#include "builder/model_builder.h"
#include "c10/hip/HIPFunctions.h"
#include "executor/utils/utils.h"

#define SELECT_OPTIMAL_LIB

namespace nn_compiler
{
namespace runtime
{
class StreamExecutor;
using OpExecutorFn = std::function<void(std::shared_ptr<ir::NNLayer>& layer, StreamExecutor& stream_executor)>;

class StreamExecutor
{
   public:
    typedef std::unordered_map<int64_t, std::pair<ir::DataType, torch::jit::IValue>> blob_store_type;

    StreamExecutor(std::pair<std::shared_ptr<ir::NNGraph>, blob_store_type> model, std::string model_type);

    ~StreamExecutor();

    RetVal preProcess();

    RetVal inferenceModel(const std::vector<torch::Tensor>& input_tensors, std::vector<torch::Tensor>& output_tensors);

    RetVal inferenceModelwithProfiling(const std::vector<torch::Tensor>& input_tensors,
                                       std::vector<torch::Tensor>& output_tensors);

    void updateBlob(int64_t blob_id, ir::DataType dtype, const torch::jit::IValue& iv);

    std::pair<ir::DataType, torch::jit::IValue>& findBlob(int64_t blob_id);

    bool checkValidBlobID(int64_t blob_id);

    OpExecutorFn findAtenExecutorByType(ir::LayerType& op_type);

    OpExecutorFn findMIOpenExecutorByType(ir::LayerType& op_type);

    OpExecutorFn getOpExecutor(uint32_t& id);

    void setOptimalLibForOpExecution();

    void registerOp();

    void setInputTensors(const std::vector<torch::Tensor>& input_tensors);

    void getOutputTensors(std::vector<torch::Tensor>& output_tensors);

    const std::shared_ptr<ir::NNGraph> getGraph();

    std::vector<torch::Tensor> iValueParser(torch::jit::IValue& iv);

    void setCursor(int64_t cursor)
    {
        assert(cursor >= 0);
        cursor_ = cursor;
    }

    void pushInLoopConditionStack(int64_t id) { loop_condition_stack_.push(id); }

    int64_t getTopOfLoopConditionStack() { return loop_condition_stack_.top(); }

    void popOfLoopConditionStack() { loop_condition_stack_.pop(); }

    void insertInRelationBlobIDsMap(int key, int value1, int value2)
    {
        releation_blob_ids_map_.insert({key, {value1, value2}});
    }

    std::pair<int, int> findInRelationBlobIDsMap(int key)
    {
        auto it = releation_blob_ids_map_.find(key);
        assert(it != releation_blob_ids_map_.end());
        return it->second;
    }

    void setModelType(const std::string& model_type) { model_type_ = model_type; }

    const std::string& getModelType() const { return model_type_; }

    // miopen
    const miopenHandle_t& getMIOpenHandle() const { return handle_; }

    const miopenRNNDescriptor_t& getMIOpenRNNDesc() const { return rnn_desc_; }

    struct MiopenLstmTensors {
        miopenTensorDescriptor_t* input_tensor_;
        miopenTensorDescriptor_t* hidden_tensor_;
        miopenTensorDescriptor_t* weight_tensor_;
        miopenTensorDescriptor_t* output_tensor_;
        std::vector<miopenTensorDescriptor_t>* input_tensors_;
        std::vector<miopenTensorDescriptor_t>* output_tensors_;
    };

    MiopenLstmTensors getMiopenLstmTensors()
    {
        MiopenLstmTensors tensors;
        tensors.input_tensor_ = &input_tensor_;
        tensors.hidden_tensor_ = &hidden_tensor_;
        tensors.weight_tensor_ = &weight_tensor_;
        tensors.output_tensor_ = &output_tensor_;
        tensors.input_tensors_ = &input_tensors_;
        tensors.output_tensors_ = &output_tensors_;
        return tensors;
    }

    void setStreams()
    {
        for (int i = 0; i < stream_num_; i++) {
            hipStream_t stream;
            hipStreamCreate(&stream);
            streams_.push_back(stream);
        }
    }

    std::vector<hipStream_t>& getStreams() { return streams_; }

    void setStreamNum(int stream_num) { stream_num_ = stream_num; }

    int getStreamNum() { return stream_num_; }

   private:
    std::shared_ptr<ir::NNGraph> graph_;

    // op executor map
    std::unordered_map<uint32_t, OpExecutorFn> op_executor_;
    std::unordered_map<ir::LayerType, OpExecutorFn> aten_executor_map_;
    std::unordered_map<ir::LayerType, OpExecutorFn> miopen_executor_map_;

    std::pair<ir::DataType, torch::jit::IValue> undefined_data_;

    int64_t cursor_ = 0;  // the program counter

    blob_store_type global_blobs_;
    std::vector<int64_t> input_blob_ids_;
    std::vector<int64_t> output_blob_ids_;

    std::stack<int64_t> loop_condition_stack_;
    std::unordered_map<int, std::pair<int, int>> releation_blob_ids_map_;

    std::string model_type_ = "";

    // miopen
    miopenTensorDescriptor_t input_tensor_, hidden_tensor_, weight_tensor_, output_tensor_;
    std::vector<miopenTensorDescriptor_t> input_tensors_;
    std::vector<miopenTensorDescriptor_t> output_tensors_;
    miopenHandle_t handle_;
    miopenRNNDescriptor_t rnn_desc_;

    // multi-stream
    std::vector<hipStream_t> streams_;
    int stream_num_ = 0;

    static std::mutex stream_exec_mutex_;

    bool has_set_optimal_lib_ = false;
    std::set<uint32_t> layers_to_select_lib_;
};

}  // namespace runtime
}  // namespace nn_compiler
