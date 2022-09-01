#include "c10/hip/HIPFunctions.h"
#include "executor/op_executor/aten_ops_executor.h"
#include "executor/op_executor/custom_ops.h"
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
void executeMultiStream(std::shared_ptr<nn_compiler::ir::NNLayer> &layer, StreamExecutor &stream_executor)
{
    DLOG(INFO) << "execute Multi-stream node";

    auto muti_stream_layer = std::static_pointer_cast<nn_compiler::ir::MultiStreamLayer>(layer);
    auto execution_layers = muti_stream_layer->getLayers();
    auto layer_num = muti_stream_layer->getLayerNum();

    auto streams = stream_executor.getStreams();
    for (int i = 0; i < layer_num; i++) {
        auto execution_layer = execution_layers[i];
        auto layer_type = execution_layer->getType();
        if (layer_type == ir::LayerType::ATENADDMM) {
            executeAtenAddmmWithStream(execution_layer, stream_executor, streams[i]);
        } else if (layer_type == ir::LayerType::ATENMATMUL) {
            executeAtenMatmulWithStream(execution_layer, stream_executor, streams[i]);
        } else {
            DLOG(FATAL) << "Unsupported layer type: " << ir::convertLayerTypeToString(layer_type)
                        << " found in multi-stream execution.";
        }
    }

    stream_executor.setCursor(layer->getID() + layer_num);
}

}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler
