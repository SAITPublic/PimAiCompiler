/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

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
