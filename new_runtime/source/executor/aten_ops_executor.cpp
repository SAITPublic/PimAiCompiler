#include "new_runtime/include/executor/aten_ops_executor.h"

namespace nn_compiler
{
namespace runtime
{

void executorAtenReshape(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute AtenReshape";
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    auto input_tensor = iv_tensor.toTensor();

    auto iv = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv.isList());
    std::vector<int64_t> shape;
    int size = 1;
    for (auto item : iv.toList().vec()) {
        int64_t val = item.toInt();
        shape.push_back(val);
        size *= val;
    }

    auto output_tensor = atenReshape(input_tensor, at::IntArrayRef(shape));
    auto out_blob_id = getUniqueOutStensorIds(layer)[0];
    stream_executor.updateBlob(out_blob_id, DataType::TENSOR, tensorToIValue(output_tensor));
}

}  // namespace runtime
}  // namespace nn_compiler
