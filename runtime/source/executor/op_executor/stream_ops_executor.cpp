#include "c10/hip/HIPFunctions.h"
#include "executor/op_executor/aten_ops_executor.h"
#include "executor/op_executor/stream_ops_executor.h"
#include "executor/utils/utils.h"
#include "utils/utils.h"


using namespace nn_compiler::runtime::utils;

namespace nn_compiler
{
namespace runtime
{
namespace op_executor
{
void executeMultiStream(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Multi-stream node";

    auto muti_stream_layer = std::static_pointer_cast<nn_compiler::ir::MultiStreamLayer>(layer);
    auto execution_layers = muti_stream_layer->getLayers();
    auto layer_num = muti_stream_layer->getLayerNum();

    hipStream_t streams[layer_num];

    for (int i = 0; i < layer_num; i++) {
        auto execution_layer = execution_layers[i];
        auto in_stensor_id = execution_layer->getInSTensorID();
        auto out_stensor_id = execution_layer->getOutSTensorID();

        torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
        torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
        assert(iv_self.isTensor() && iv_other.isTensor());

        auto self_tensor = iv_self.toTensor();
        auto other_tensor = iv_other.toTensor();
        torch::jit::IValue output;

        hipStreamCreate(&streams[i]);
        customAtenMatmul(self_tensor, other_tensor, output, streams[i]);
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, output);
    }
    hipDeviceSynchronize();

    stream_executor.setCursor(layer->getID() + layer_num);
}

}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler
