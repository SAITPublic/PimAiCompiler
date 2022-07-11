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

    hipStream_t streams[layer_num];

    std::vector<_Float16 *> vector_values;
    std::vector<_Float16 *> matrix_values;
    std::vector<_Float16 *> outputs;
    std::vector<at::Tensor> output_tensors;

    std::vector<int> m_values;
    std::vector<int> n_values;
    std::vector<int> k_values;

    std::vector<uint32_t> output_ids;

    float alpha = 1.0f;
    float beta = 0.0f;
    static constexpr int NB = 256;

    for (int i = 0; i < layer_num; i++) {
        auto execution_layer = execution_layers[i];
        auto in_stensor_id = execution_layer->getInSTensorID();
        auto out_stensor_id = execution_layer->getOutSTensorID();

        torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
        torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
        assert(iv_self.isTensor() && iv_other.isTensor());
        auto self_tensor = iv_self.toTensor();
        auto other_tensor = iv_other.toTensor();

        int dim_i0 = self_tensor.dim();
        int dim_i1 = other_tensor.dim();

        m_values.push_back(self_tensor.size(dim_i0 - 2));
        n_values.push_back(1);
        k_values.push_back(self_tensor.size(dim_i0 - 1));

        matrix_values.push_back((_Float16 *)self_tensor.data_ptr());
        vector_values.push_back((_Float16 *)other_tensor.data_ptr());

        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto output_shape = other_tensor.sizes().vec();
        if (dim_i0 > dim_i1) {
            output_shape = self_tensor.sizes().vec();
            output_shape[dim_i0 - 2] = m_values[i];
            output_shape.pop_back();
        } else {
            output_shape[dim_i1 - 1] = 1;
            output_shape[dim_i1 - 2] = m_values[i];
        }
        auto output = at::zeros(output_shape, options);
        outputs.push_back((_Float16 *)output.data_ptr());
        output_tensors.push_back(output);
        output_ids.push_back(out_stensor_id[0]);

        hipStreamCreate(&streams[i]);
    }

    for (int i = 0; i < layer_num; i++) {
        // rocblas_gemv_template_Axy(stream[i], A, x, y, m, n, k, alpha, beta);
        hipLaunchKernelGGL((gemvt_kernel_Axy<NB>), dim3(1, m_values[i]), dim3(NB), 0, streams[i],
                           m_values[i], k_values[i], alpha,
                           matrix_values[i],  // k
                           k_values[i],
                           vector_values[i],  // kxn
                           n_values[i], beta, outputs[i]);
    }

    for (int i = 0; i < layer_num; i++) {
        hipStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < layer_num; i++) {
        stream_executor.updateBlob(output_ids[i], DataType::TENSOR, output_tensors[i]);
    }

    stream_executor.setCursor(layer->getID() + layer_num);
}

}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler
