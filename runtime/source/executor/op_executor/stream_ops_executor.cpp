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
void executeStartMultiStream(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Start Multi-stream node";

    auto start_muti_stream_layer = std::static_pointer_cast<nn_compiler::ir::StartMultiStreamLayer>(layer);
    auto execution_layers = start_muti_stream_layer->getLayers();
    auto stream_num = start_muti_stream_layer->getLayerNum();
    hipStream_t streams[stream_num];

    for (int i = 0; i < stream_num; i++) {
        hipStreamCreate(&streams[i]);

        // TODO: run kernels for execution layers

    }
    hipDeviceSynchronize();

}

void executeEndMultiStream(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute End Multi-stream node";

    auto end_muti_stream_layer = std::static_pointer_cast<nn_compiler::ir::EndMultiStreamLayer>(layer);
    auto layer_num = end_muti_stream_layer->getLayerNum();
    stream_executor.setCursor(layer->getID() + layer_num + 1);
}
}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler
