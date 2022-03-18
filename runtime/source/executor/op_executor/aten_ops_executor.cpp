#include <algorithm>
#include <vector>

#include "c10/hip/HIPFunctions.h"
#include "ir/include/nn_model.h"
#include "ir/include/nn_network.h"
#include "ir/include/types.h"
#include "ir/include/common/utils.hpp"
#include "ir/include/layers/all_layers.h"
#include "runtime/include/executor/op_executor/aten_ops.h"
#include "runtime/include/executor/op_executor/aten_ops_executor.h"
#include "runtime/include/executor/op_executor/custom_ops.h"
#include "runtime/include/executor/op_executor/prim_ops_executor.h"
#include "runtime/include/executor/stream_executor.h"
#include "runtime/include/executor/utils/utils.h"
#include "pim_runtime_api.h"
#include "tv_tools.h"

namespace nn_compiler {
namespace runtime {

void executorAtenAdd(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Add node";

    auto add_layer = std::static_pointer_cast<nn_compiler::ir::AtenAddLayer>(layer);

    int in_id = 0;
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();
    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[in_id++]).second;

    int64_t alpha = add_layer->getAlpha();
    if (nn_compiler::ir::isDefaultValue(alpha)) {
        if (in_stensor_id.size() == 3) {
            // if "prim::if" layer linked to current, this edge has no practical meaning
            if (stream_executor.checkValidBlobID(in_id)) {
                auto alpha_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
                assert(alpha_iv.isInt());
                alpha = alpha_iv.toInt();
            } else {
                alpha = 1;
            }
        } else {
            alpha = 1;
        }
    }

    if (iv_self.isInt() && iv_other.isInt()) {
        int64_t in_self = iv_self.toInt();
        auto output = atenAdd(in_self, iv_other.toInt(), alpha);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::INT64, intToIValue(output));
        return;
    }
    at::Tensor self_tensor = iv_self.toTensor();
    auto dtype = stream_executor.findBlob(in_stensor_id[1]).first;
    if (dtype == DataType::TENSOR) {
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        // auto output = atenAdd(self_tensor, other_tensor, alpha);
        {
            if (!self_tensor.is_contiguous()) self_tensor = self_tensor.contiguous();
            if (!other_tensor.is_contiguous()) other_tensor = other_tensor.contiguous();
            int dim0 = self_tensor.dim();
            int dim1 = other_tensor.dim();

            auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
            auto output_tmp = at::zeros(self_tensor.sizes().vec(), options);

            _Float16* A = (_Float16*)self_tensor.data_ptr();
            _Float16* B = (_Float16*)other_tensor.data_ptr();
            _Float16* C = A;

            at::Tensor tmp;
            auto input1_layer = stream_executor.getGraph()->getLayerByPosition((layer->getPreLayerIDs())[0]);
            auto input2_layer = stream_executor.getGraph()->getLayerByPosition((layer->getPreLayerIDs())[1]);
            bool add_opt_flag = input1_layer->getType() == nn_compiler::ir::LayerType::ATENLSTM1 &&
                input2_layer->getType() == nn_compiler::ir::LayerType::ATENLSTM1 &&
                std::static_pointer_cast<nn_compiler::ir::AtenLSTM1Layer>(input1_layer)->getCustomOptNumber() == 2 &&
                std::static_pointer_cast<nn_compiler::ir::AtenLSTM1Layer>(input2_layer)->getCustomOptNumber() == 1;
            if (stream_executor.model_type_ == "GNMT" && add_opt_flag) {
                    auto out1_layer = stream_executor.getGraph()->getLayerByPosition((layer->getNextLayerIDs())[0]);
                    auto out1_out1_layer = stream_executor.getGraph()->getLayerByPosition((out1_layer->getNextLayerIDs())[0]);
                int cat_mem_id =
                    std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(out1_out1_layer)->getMemLayerId();
                auto it = stream_executor.global_blobs_.find(cat_mem_id);
                auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
                tmp = torch::from_blob((_Float16*)(it->second.second.toTensor().data_ptr()), {1, 1, 1024}, options);
                C = (_Float16*)tmp.data_ptr();
            }

            int m = 1;
            int n = 1;
            int a_m_s = 1;
            int a_n_s = 1;
            int b_m_s = 1;
            int b_n_s = 1;

            m = self_tensor.size(dim0 - 2);
            n = self_tensor.size(dim0 - 1);

            bool sym = (dim0 == dim1);
            custom_add(nullptr, A, B, C, m, n, alpha, sym, a_m_s, a_n_s, b_m_s, b_n_s);

            // update output
            if (add_opt_flag) {
                stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(tmp));
            } else {
                stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(self_tensor));
            }
        }
        // update output
        // stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (isScalarType(dtype)) {
        assert(iv_other.isScalar());
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = atenAdd(self_tensor, other_scalar, alpha);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::add";
    }
}  // executorAtenAdd

void executorAtenAddmm(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Addmm node";

    auto addmm_layer = std::static_pointer_cast<nn_compiler::ir::AtenAddmmLayer>(layer);
    auto act_type = addmm_layer->get_act_type();

    // TODO choose the corresponding kernel when activation type is aten::none, aten::relu, aten::max
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_mat1 = stream_executor.findBlob(in_stensor_id[1]).second;
    torch::jit::IValue iv_mat2 = stream_executor.findBlob(in_stensor_id[2]).second;
    assert(iv_self.isTensor() && iv_mat1.isTensor() && iv_mat2.isTensor());
    torch::jit::IValue iv_beta = stream_executor.findBlob(in_stensor_id[3]).second;
    torch::jit::IValue iv_alpha = stream_executor.findBlob(in_stensor_id[4]).second;

    int dim_i0 = iv_mat1.toTensor().dim();
    int dim_i1 = iv_mat2.toTensor().dim();
    int dim_self = iv_self.toTensor().dim();
    int i0_is_vector = 0;
    int i1_is_vector = 0;

    for (int i = 0; i < dim_i0; ++i) {
        if (iv_mat1.toTensor().size(i) != 1) {
            i0_is_vector += 1;
        }
    }

    for (int i = 0; i < dim_i1; ++i) {
        if (iv_mat2.toTensor().size(i) != 1) {
            i1_is_vector += 1;
        }
    }

    if (i0_is_vector == 1 && i1_is_vector != 1 && dim_i0 > 1) {
        float alpha = 1.0f;
        float beta = 0.0f;
        bool relu = false;
        int m = 1;
        if (act_type == "aten::relu") {
            relu = true;
        }

        auto self = iv_self.toTensor();
        auto mat1 = iv_mat1.toTensor();
        auto mat2 = iv_mat2.toTensor();
        if (!self.is_contiguous()) self = self.contiguous();
        if (!mat1.is_contiguous()) mat1 = mat1.contiguous();
        if (!mat2.is_contiguous()) mat2 = mat2.contiguous();

        int n = mat2.size(dim_i1 - 1);
        int k = mat2.size(dim_i1 - 2);

        _Float16* b = (_Float16*)self.data_ptr();
        _Float16* x = (_Float16*)mat1.data_ptr();
        _Float16* A = (_Float16*)mat2.data_ptr();
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto output_shape = mat1.sizes().vec();
        output_shape[dim_i0 - 1] = n;
        output_shape[dim_i0 - 2] = 1;
        auto output = at::zeros(output_shape, options);
        _Float16* y = (_Float16*)output.data_ptr();

        PimDesc* pim_desc = PimCreateDesc(1, 1, n, k, PIM_FP16, OP_GEMV);
        PimBo* dev_op0 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT, x);
        PimBo* dev_op1 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT, A);
        PimBo* dev_op2 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT, b);
        PimBo* dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT, y);

        PimExecuteGemvAdd(dev_out, dev_op0, dev_op1, dev_op2, relu, nullptr);

        PimDestroyBo(dev_op0);
        PimDestroyBo(dev_op1);
        PimDestroyBo(dev_op2);
        PimDestroyBo(dev_out);
        PimDestroyDesc(pim_desc);

        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (i0_is_vector != 1 && i1_is_vector == 1 && dim_i1 > 1) {
        bool relu = false;
        if (act_type == "aten::relu") {
            relu = true;
        }

        auto self = iv_self.toTensor();
        auto mat1 = iv_mat1.toTensor();
        auto mat2 = iv_mat2.toTensor();
        if (!self.is_contiguous()) self = self.contiguous();
        if (!mat1.is_contiguous()) mat1 = mat1.contiguous();
        if (!mat2.is_contiguous()) mat2 = mat2.contiguous();

        int m = mat1.size(dim_i0 - 2);
        int n = 1;
        int k = mat1.size(dim_i0 - 1);

        _Float16* b = (_Float16*)self.data_ptr();
        _Float16* A = (_Float16*)mat1.data_ptr();
        _Float16* x = (_Float16*)mat2.data_ptr();
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto output_shape = mat2.sizes().vec();
        output_shape[dim_i1 - 1] = 1;
        output_shape[dim_i1 - 2] = m;
        auto output = at::zeros(output_shape, options);
        _Float16* y = (_Float16*)output.data_ptr();

        PimDesc* pim_desc = PimCreateDesc(1, 1, m, k, PIM_FP16, OP_GEMV);
        PimBo* dev_op0 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT, x);
        PimBo* dev_op1 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT_T, A);
        PimBo* dev_op2 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT, b);
        PimBo* dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT, y);

        PimExecuteGemvAdd(dev_out, dev_op0, dev_op1, dev_op2, relu, nullptr);

        PimDestroyBo(dev_op0);
        PimDestroyBo(dev_op1);
        PimDestroyBo(dev_op2);
        PimDestroyBo(dev_out);
        PimDestroyDesc(pim_desc);

        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (dim_self == 1) {
        auto self = iv_self.toTensor();
        auto mat1 = iv_mat1.toTensor();
        auto mat2 = iv_mat2.toTensor();

        self = self.unsqueeze(1);
        self = self.repeat({1, mat2.size(dim_i1 - 1)});

        if (dim_i1 != 2) {
            mat2 = mat2.squeeze(0);
        }

        auto output = atenAddmm(self, mat1, mat2, iv_beta.toScalar(), iv_alpha.toScalar());
        // update output
        for (int i = 0; i < dim_i1 - 2; ++i) {
            output = output.unsqueeze(0);
        }
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        auto output = atenAddmm(iv_self.toTensor(), iv_mat1.toTensor(), iv_mat2.toTensor(), iv_beta.toScalar(),
                                      iv_alpha.toScalar());

        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    }
}

void executorAtenAnd(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten And node";
    
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_a = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_b = stream_executor.findBlob(in_stensor_id[1]).second;

    bool value_a;
    bool value_b;
    if (iv_a.isBool() && iv_b.isBool()) {
        value_a = iv_a.toBool();
        value_b = iv_b.toBool();
    } else if (iv_a.isInt() && iv_b.isInt()) {
        value_a = iv_a.toInt();
        value_b = iv_b.toInt();
    } else {
        DLOG(FATAL) << "Wrong input type for AtenAdd.";
    }

    auto output = atenAnd(value_a, value_b);
    stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, boolToIValue(output));
}

void executorAtenAny(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Any node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    auto output = atenAny(self_tensor);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, torch::jit::IValue(output));
}

void executorAtenAppend(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Append node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();
    assert(in_stensor_id.size() == 2);

    // Find the input blob
    torch::jit::IValue iv_list = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_el = stream_executor.findBlob(in_stensor_id[1]).second;

    assert(iv_list.isList());

    c10::List<at::IValue> list = iv_list.toList();
    atenAppend(list, iv_el);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::LIST, listToIValue(list));
}

void executorAtenArange1(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Arange1 node";

    auto arange1_layer = std::static_pointer_cast<nn_compiler::ir::AtenArange1Layer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    auto end = arange1_layer->getEnd();
    if (nn_compiler::ir::isDefaultValue(end)) {
        auto iv_end = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        end = iv_end.toInt();
    }

    at::TensorOptions options;
    auto dtype = arange1_layer->getDtype();
    if (nn_compiler::ir::isDefaultValue(dtype)) {
        auto iv_dtype = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_dtype.isNone()) {
            options = options.dtype(iv_dtype.toScalarType());
        }
    } else {
        options = options.dtype(at::ScalarType(dtype));
    }

    auto layout = arange1_layer->getLayout();
    if (nn_compiler::ir::isDefaultValue(layout)) {
        auto iv_layout = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_layout.isNone()) {
            options = options.layout(iv_layout.toLayout());
        }
    } else {
        options = options.layout(at::Layout(layout));
    }

    auto device = arange1_layer->getDevice();
    if (nn_compiler::ir::isDefaultValue(device)) {
        auto iv_device = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_device.isNone()) {
            options = options.device(iv_device.toDevice());
        }
    } else {
        options = options.device(device);
    }

    auto pin_memory = arange1_layer->getPinMemory();
    if (nn_compiler::ir::isDefaultValue(pin_memory)) {
        auto iv_pin_memory = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_pin_memory.isNone()) {
            options = options.pinned_memory(iv_pin_memory.toBool());
        }
    } else {
        options = options.pinned_memory(static_cast<bool>(pin_memory));
    }

    auto output = atenArange1(end, options);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenArange2(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Arange2 node";

    auto arange2_layer = std::static_pointer_cast<nn_compiler::ir::AtenArange2Layer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    auto start = arange2_layer->getStart();
    if (nn_compiler::ir::isDefaultValue(start)) {
        auto iv_start = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        start = iv_start.toInt();
    }

    auto end = arange2_layer->getEnd();
    if (nn_compiler::ir::isDefaultValue(end)) {
        auto iv_end = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        end = iv_end.toInt();
    }

    at::TensorOptions options;
    auto dtype = arange2_layer->getDtype();
    if (nn_compiler::ir::isDefaultValue(dtype)) {
        auto iv_dtype = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_dtype.isNone()) {
            options = options.dtype(iv_dtype.toScalarType());
        }
    } else {
        options = options.dtype(at::ScalarType(dtype));
    }

    auto layout = arange2_layer->getLayout();
    if (nn_compiler::ir::isDefaultValue(layout)) {
        auto iv_layout = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_layout.isNone()) {
            options = options.layout(iv_layout.toLayout());
        }
    } else {
        options = options.layout(at::Layout(layout));
    }

    auto device = arange2_layer->getDevice();
    if (nn_compiler::ir::isDefaultValue(device)) {
        auto iv_device = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_device.isNone()) {
            options = options.device(iv_device.toDevice());
        }
    } else {
        options = options.device(device);
    }

    auto pin_memory = arange2_layer->getPinMemory();
    if (nn_compiler::ir::isDefaultValue(pin_memory)) {
        auto iv_pin_memory = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_pin_memory.isNone()) {
            options = options.pinned_memory(iv_pin_memory.toBool());
        }
    } else {
        options = options.pinned_memory(static_cast<bool>(pin_memory));
    }

    auto output = atenArange2(start, end, options);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenArange3(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Arange3 node";

    auto arange3_layer = std::static_pointer_cast<nn_compiler::ir::AtenArange3Layer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    auto start = arange3_layer->getStart();
    if (nn_compiler::ir::isDefaultValue(start)) {
        auto iv_start = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        start = iv_start.toInt();
    }

    auto end = arange3_layer->getEnd();
    if (nn_compiler::ir::isDefaultValue(end)) {
        auto iv_end = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        end = iv_end.toInt();
    }

    auto step = arange3_layer->getStep();
    if (nn_compiler::ir::isDefaultValue(step)) {
        auto iv_step = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        step = iv_step.toInt();
    }

    at::TensorOptions options;
    auto dtype = arange3_layer->getDtype();
    if (nn_compiler::ir::isDefaultValue(dtype)) {
        auto iv_dtype = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_dtype.isNone()) {
            options = options.dtype(iv_dtype.toScalarType());
        }
    } else {
        options = options.dtype(at::ScalarType(dtype));
    }

    auto layout = arange3_layer->getLayout();
    if (nn_compiler::ir::isDefaultValue(layout)) {
        auto iv_layout = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_layout.isNone()) {
            options = options.layout(iv_layout.toLayout());
        }
    } else {
        options = options.layout(at::Layout(layout));
    }

    auto device = arange3_layer->getDevice();
    if (nn_compiler::ir::isDefaultValue(device)) {
        auto iv_device = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_device.isNone()) {
            options = options.device(iv_device.toDevice());
        }
    } else {
        options = options.device(device);
    }

    auto pin_memory = arange3_layer->getPinMemory();
    if (nn_compiler::ir::isDefaultValue(pin_memory)) {
        auto iv_pin_memory = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_pin_memory.isNone()) {
            options = options.pinned_memory(iv_pin_memory.toBool());
        }
    } else {
        options = options.pinned_memory(static_cast<bool>(pin_memory));
    }

    auto output = atenArange3(start, end, step, options);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenAsTensor(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten AsTensor node";

    auto as_tensor_layer = std::static_pointer_cast<nn_compiler::ir::AtenAsTensorLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_tensor.isTensor());
    auto in_tensor = iv_tensor.toTensor();

    auto int_dtype = as_tensor_layer->getDtype();
    if (nn_compiler::ir::isDefaultValue(int_dtype)) {
        auto iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        int_dtype = iv.toInt();
    }
    auto dtype = at::ScalarType(int_dtype);

    auto str_device = as_tensor_layer->getDevice();
    if (nn_compiler::ir::isDefaultValue(str_device)) {
        auto map_value = stream_executor.findBlob(in_stensor_id[in_id++]);
        auto iv = map_value.second;
        if (map_value.first != DataType::NONE) {
            str_device = iv.toDevice().str();
        } else {
            str_device = in_tensor.device().str();
        }
    }
    auto device = at::Device(str_device);

    auto output = atenAsTensor(in_tensor, dtype, device);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenBatchNorm2d(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten BN2d node";

    auto batch_norm_2d_layer = std::static_pointer_cast<nn_compiler::ir::AtenBatchNorm2dLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    auto get_tensor = [&stream_executor](int id) {
        auto blob = stream_executor.findBlob(id);
        assert(blob.second.isTensor());
        return blob.second.toTensor();
    };
    at::Tensor input = get_tensor(in_stensor_id[0]);
    at::Tensor running_mean = get_tensor(in_stensor_id[1]);
    at::Tensor running_var = get_tensor(in_stensor_id[2]);

    auto weight_ids = batch_norm_2d_layer->getWeightIds();
    auto bias_ids = batch_norm_2d_layer->getBiasIds();
    assert(weight_ids.size() == 1 && bias_ids.size() == 1);
    auto weight_iv = stream_executor.findBlob(weight_ids[0]).second;
    assert(weight_iv.isTensor());
    at::Tensor weight = weight_iv.toTensor();
    auto bias_iv = stream_executor.findBlob(bias_ids[0]).second;
    assert(bias_iv.isTensor());
    at::Tensor bias = bias_iv.toTensor();

    // Get input attrs
    int training = batch_norm_2d_layer->getTraining();
    double monentum = batch_norm_2d_layer->getMomentum();
    double eps = batch_norm_2d_layer->getEps();
    int cudnn_enabled = batch_norm_2d_layer->getCudnnEnabled();

    int offest = 3;
    if (nn_compiler::ir::isDefaultValue(training)) {
        auto iv = stream_executor.findBlob(in_stensor_id[3]).second;
        assert(iv.isInt());
        training = static_cast<int>(iv.toInt());
    }
    if (nn_compiler::ir::isDefaultValue(monentum)) {
        auto iv = stream_executor.findBlob(in_stensor_id[4]).second;
        assert(iv.isDouble());
        monentum = iv.toDouble();
    }
    if (nn_compiler::ir::isDefaultValue(eps)) {
        auto iv = stream_executor.findBlob(in_stensor_id[5]).second;
        assert(iv.isDouble());
        eps = iv.toDouble();
    }
    if (nn_compiler::ir::isDefaultValue(cudnn_enabled)) {
        auto iv = stream_executor.findBlob(in_stensor_id[6]).second;
        assert(iv.isInt());
        cudnn_enabled = static_cast<int>(iv.toInt());
    }

    if (training == 1) {
        DLOG(FATAL) << "Currently, NNRuntime only support inference !";
    }

    // Call kernel
    auto output = atenBatchNorm2d(input, weight, bias, running_mean, running_var, static_cast<bool>(training), monentum,
                                  eps, static_cast<bool>(cudnn_enabled));
    // save outputs
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenBitwiseNot(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten BitwiseNot node";
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    auto output = atenBitwiseNot(self_tensor);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, torch::jit::IValue(output));
}

void executorAtenBmm(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Bmm node";
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor() && iv_other.isTensor());

    at::Tensor tmp0, tmp1, tmp2;
    if (stream_executor.model_type_ == "GNMT") {
        auto out2_layer = stream_executor.getGraph()->getLayerByPosition((layer->getNextLayerIDs())[1]);
        auto out2_out1_layer = stream_executor.getGraph()->getLayerByPosition((out2_layer->getNextLayerIDs())[0]);
        auto out2_out2_layer = stream_executor.getGraph()->getLayerByPosition((out2_layer->getNextLayerIDs())[1]);
        auto out2_out3_layer = stream_executor.getGraph()->getLayerByPosition((out2_layer->getNextLayerIDs())[2]);
        auto out2_out1_out1_layer = stream_executor.getGraph()->getLayerByPosition((out2_out1_layer->getNextLayerIDs())[0]);
        auto out2_out2_out1_layer = stream_executor.getGraph()->getLayerByPosition((out2_out2_layer->getNextLayerIDs())[0]);
        auto out2_out3_out1_layer = stream_executor.getGraph()->getLayerByPosition((out2_out3_layer->getNextLayerIDs())[0]);
        // auto end_if_node = op_node.getOutEdge(1).getOutNode();
        int64_t cat_mem_id =
            std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(out2_out1_out1_layer)->getMemLayerId();
        auto it = stream_executor.global_blobs_.find(cat_mem_id);
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        tmp0 = torch::from_blob((_Float16*)(it->second.second.toTensor().data_ptr()) + 1024, {1, 1, 1024}, options);

        cat_mem_id =
            std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(out2_out2_out1_layer)->getMemLayerId();
        it = stream_executor.global_blobs_.find(cat_mem_id);
        tmp1 = torch::from_blob((_Float16*)(it->second.second.toTensor().data_ptr()) + 1024, {1, 1, 1024}, options);

        cat_mem_id =
            std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(out2_out3_out1_layer)->getMemLayerId();
        it = stream_executor.global_blobs_.find(cat_mem_id);
        tmp2 = torch::from_blob((_Float16*)(it->second.second.toTensor().data_ptr()) + 1024, {1, 1, 1024}, options);
    }

    auto self_tensor = iv_self.toTensor();
    auto other_tensor = iv_other.toTensor();

    if (!self_tensor.is_contiguous()) self_tensor = self_tensor.contiguous();
    if (!other_tensor.is_contiguous()) other_tensor = other_tensor.contiguous();

    int dim0 = self_tensor.dim();
    int dim1 = other_tensor.dim();
    int m = self_tensor.size(dim0 - 2);
    int k = self_tensor.size(dim0 - 1);
    int n = other_tensor.size(dim1 - 1);
    int k_ = other_tensor.size(dim1 - 2);
    assert(k_ == k);

    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
    auto output_shape = self_tensor.sizes().vec();
    output_shape[dim0 - 1] = n;
    output_shape[dim0 - 2] = m;

    _Float16* x = (_Float16*)self_tensor.data_ptr();
    _Float16* A = (_Float16*)other_tensor.data_ptr();
    auto output = at::zeros(output_shape, options);
    _Float16* y = (_Float16*)output.data_ptr();

    if (stream_executor.model_type_ == "GNMT") {
        y = (_Float16*)tmp0.data_ptr();
    }

    rocblas_bmm_template_xAy(nullptr, x, A, y, m, n, k);
    if (stream_executor.model_type_ == "GNMT") {
        atenCopy_(tmp1, tmp0, c10::attr::non_blocking);
        atenCopy_(tmp2, tmp0, c10::attr::non_blocking);
    }

    // update output
    if (stream_executor.model_type_ == "GNMT") {
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(tmp0));
    } else {
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    }
}

void executorAtenBool(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Bool node";
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    auto ivalue = stream_executor.findBlob(in_stensor_id[0]).second;
    bool output = false;
    if (ivalue.isTensor()) {
        auto tensor = ivalue.toTensor();
        output = atenBool(tensor);
    } else if (ivalue.isInt()) {
        auto integer = ivalue.toInt();
        output = atenBool(integer);
    } else if (ivalue.isDouble()) {
        auto double_value = ivalue.toDouble();
        output = atenBool(double_value);
    } else {
        DLOG(FATAL) << "Unsupported type for aten::Bool.";
    }
    stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, boolToIValue(output));
}

void executorAtenCat(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Cat node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();
    
    auto cat_layer = std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(layer);

    std::vector<at::Tensor> tensor_vec;

    auto input1_layer = stream_executor.getGraph()->getLayerByPosition((layer->getPreLayerIDs())[0]);
    auto in_input1_stensor_id = input1_layer->getInSTensorID();
    // TODO(SRCX): implement this part of optimization.
    if (stream_executor.model_type_ == "GNMT" && in_input1_stensor_id.size() == 0) {
        int cat_mem_id = cat_layer->getMemLayerId();
        auto it = stream_executor.global_blobs_.find(cat_mem_id);
        auto output = it->second.second.toTensor().clone();
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        return;
    }

    int cat_mem_id = cat_layer->getMemLayerId();
    auto it = stream_executor.global_blobs_.find(cat_mem_id);
    if (stream_executor.model_type_ == "GNMT" && it != stream_executor.global_blobs_.end()) {
        auto output = it->second.second;
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, output);
        return;
    }

    auto ivalue = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(ivalue.isTensorList());

    auto c10_tensor_list = ivalue.toTensorList();
    for (auto tensor : c10_tensor_list) {
        tensor_vec.push_back(tensor);
    }
    at::TensorList tensor_list(tensor_vec);

    auto dim = cat_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto dim_iv = stream_executor.findBlob(in_stensor_id[1]).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }

    auto output = atenCat(tensor_list, dim);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenCeil(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Ceil node";
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    assert(in_stensor_id.size() == 1);
    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    auto output = atenCeil(self_tensor);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, torch::jit::IValue(output));
}

void executorAtenChunk(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Chunk node";

    auto chunk_layer = std::static_pointer_cast<nn_compiler::ir::AtenChunkLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_tensor.isTensor());
    auto in_tensor = iv_tensor.toTensor();

    auto chunks = chunk_layer->getChunks();
    if (nn_compiler::ir::isDefaultValue(chunks)) {
        auto iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        chunks = iv.toInt();
    }

    auto dim = chunk_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(chunks)) {
        auto iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        dim = iv.toInt();
    }

    auto output = atenChunk(in_tensor, chunks, dim);
    stream_executor.updateBlob(out_stensor_id[0], DataType::LIST, vectorToIValue(output));
}

void executorAtenClamp(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Clamp node";

    auto clamp_layer = std::static_pointer_cast<nn_compiler::ir::AtenClampLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto min = clamp_layer->getMin();
    auto max = clamp_layer->getMax();
    if (nn_compiler::ir::isDefaultValue(min)) {
        auto iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        min = static_cast<int>(iv.toInt());
    }
    if (nn_compiler::ir::isDefaultValue(max)) {
        auto iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        max = static_cast<int>(iv.toInt());
    }

    auto output = atenClamp(self_tensor, min, max);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenClear(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Clear node";
    auto in_stensor_id = layer->getInSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isList());
    at::List self_list = iv_self.toList();
    atenClear(self_list);
    // update list
    stream_executor.updateBlob(in_stensor_id[0], DataType::LIST, torch::jit::IValue(self_list));
}

void executorAtenContiguous(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Contiguous node";

    auto contiguous_layer = std::static_pointer_cast<nn_compiler::ir::AtenContiguousLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto memory_format = contiguous_layer->getMemoryFormat();
    if (nn_compiler::ir::isDefaultValue(memory_format)) {
        auto iv = stream_executor.findBlob(in_stensor_id[1]).second;
        memory_format = static_cast<int>(iv.toInt());
    }

    auto output = atenContiguous(self_tensor, getMemoryFormat(memory_format));
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenConv2d(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Conv2d node";

    auto conv2d_layer = std::static_pointer_cast<nn_compiler::ir::AtenConv2dLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto weight_ids = conv2d_layer->getWeightIds();
    auto bias_ids = conv2d_layer->getBiasIds();
    auto weight_iv = stream_executor.findBlob(weight_ids[0]).second;
    auto bias_iv = stream_executor.findBlob(bias_ids[0]).second;
    assert(weight_iv.isTensor() && bias_iv.isTensor());
    at::Tensor weight_tensor = weight_iv.toTensor();
    at::Tensor bias_tensor = bias_iv.toTensor();

    // attributes of conv2d don't need default-value check, because its default values
    // are set as same as default values in aten::conv2d.
    auto stride = conv2d_layer->getStride();
    auto padding = conv2d_layer->getPadding();
    auto dilation = conv2d_layer->getDialation();
    auto groups = conv2d_layer->getGroups();

    std::vector<int64_t> stride_vec = {static_cast<int64_t>(stride[0]), static_cast<int64_t>(stride[1])};
    std::vector<int64_t> padding_vec = {static_cast<int64_t>(padding[0]), static_cast<int64_t>(padding[1])};
    std::vector<int64_t> dilation_vec = {static_cast<int64_t>(dilation[0]), static_cast<int64_t>(dilation[1])};

    auto output = atenConv2d(self_tensor, weight_tensor, bias_tensor, at::ArrayRef<int64_t>(stride_vec),
                                   at::ArrayRef<int64_t>(padding_vec), at::ArrayRef<int64_t>(dilation_vec), groups);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenCopy(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Copy node";

    auto copy_layer = std::static_pointer_cast<nn_compiler::ir::AtenCopyLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_src = stream_executor.findBlob(in_stensor_id[1]).second;
    at::Tensor self_tensor = iv_self.toTensor();
    at::Tensor src_tensor = iv_src.toTensor();

    int non_blocking = copy_layer->getNonBlocking();
    if (nn_compiler::ir::isDefaultValue(non_blocking)) {
        auto non_blocking_iv = stream_executor.findBlob(in_stensor_id[2]).second;
        non_blocking = non_blocking_iv.toInt();
    }

    auto output = atenCopy_(self_tensor, src_tensor, static_cast<bool>(non_blocking));
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, torch::jit::IValue(output));
}

void executorAtenCpu(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Cpu node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto output = atenCpu(self_tensor);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenCuda(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Cuda node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto output = atenCuda(self_tensor);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenDeriveIndex(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten derive_index node";

    auto derive_index_layer = std::static_pointer_cast<nn_compiler::ir::AtenDeriveIndexLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    // Index is an input, get Index
    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    auto index = iv.toInt();

    // Check and get start
    auto start = derive_index_layer->getStart();
    if (nn_compiler::ir::isDefaultValue(start)) {
        auto start_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        start = start_iv.toInt();
    }

    // Check and get step
    auto step = derive_index_layer->getStep();
    if (nn_compiler::ir::isDefaultValue(step)) {
        auto step_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        step = step_iv.toInt();
    }

    auto output = atenDeriveIndex(index, start, step);
    stream_executor.updateBlob(out_stensor_id[0], DataType::INT64, scalarToIValue(output));
}

void executorAtenDim(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Dim node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    assert(in_stensor_id.size() == 1);

    // Find the input blob
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();
    auto output = atenDim(tensor);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::INT64, intToIValue(output));
}

void executorAtenDiv(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Div node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    assert(in_stensor_id.size() == 2);

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    auto dtype = stream_executor.findBlob(in_stensor_id[1]).first;
    if (dtype == DataType::TENSOR) {
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = atenDiv(self_tensor, other_tensor);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (isScalarType(dtype)) {
        assert(iv_other.isScalar());
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = atenDiv(self_tensor, other_scalar);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::div";
    }
}

void executorAtenDropout(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Dropout node";

    auto dropout_layer = std::static_pointer_cast<nn_compiler::ir::AtenDropoutLayer>(layer);
    
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    double proportion = (double)dropout_layer->getProportion();
    if (nn_compiler::ir::isDefaultValue(proportion)) {
        auto proportion_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(proportion_iv.isDouble());
        proportion = proportion_iv.toDouble();
    }
    int train_val = dropout_layer->getTrain();
    if (nn_compiler::ir::isDefaultValue(train_val)) {
        auto train_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(train_iv.isBool());
        train_val = train_iv.toBool();
    }
    bool train = static_cast<bool>(train_val);
    at::Tensor output = atenDropout(tensor, proportion, train);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenEmbedding(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Embedding node";

    auto embedding_layer = std::static_pointer_cast<nn_compiler::ir::AtenEmbeddingLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    if (embedding_layer->getWeights().empty()) {
        int in_id = 0;
        torch::jit::IValue iv_weights = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        torch::jit::IValue iv_indices = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(iv_weights.isTensor());
        assert(iv_indices.isTensor());

        int64_t padding_idx = embedding_layer->getPaddingIdx();
        if (nn_compiler::ir::isDefaultValue(padding_idx)) {
            auto padding_idx_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
            assert(padding_idx_iv.isInt());
            padding_idx = padding_idx_iv.toInt();
        }

        int scale_grad_by_freq_val = embedding_layer->getScaleGrad();
        if (nn_compiler::ir::isDefaultValue(scale_grad_by_freq_val)) {
            auto scale_grad_by_freq_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
            assert(scale_grad_by_freq_iv.isInt());
            scale_grad_by_freq_val = scale_grad_by_freq_iv.toInt();
        }
        bool scale_grad_by_freq = static_cast<bool>(scale_grad_by_freq_val);

        int sparse_val = embedding_layer->getSparse();
        if (nn_compiler::ir::isDefaultValue(sparse_val)) {
            auto sparse_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
            assert(sparse_iv.isInt());
            sparse_val = sparse_iv.toInt();
        }
        bool sparse = static_cast<bool>(sparse_val);

        auto output =
            atenEmbedding(iv_weights.toTensor(), iv_indices.toTensor(), padding_idx, scale_grad_by_freq, sparse);

        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        auto weights = embedding_layer->getWeights();
        assert(weights.size() > 0);

        torch::jit::IValue iv_indices = stream_executor.findBlob(in_stensor_id[0]).second;
        assert(iv_indices.isTensor());
        auto indices_tensor = iv_indices.toTensor();
        assert(indices_tensor.item().type() == torch::kInt64);

        auto output = weights[indices_tensor.item().toInt()];
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    }
}

void executorAtenEq(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Eq node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    assert(in_stensor_id.size() == 2);

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;

    if (iv_self.isTensor()) {
        at::Tensor self_tensor = iv_self.toTensor();
        at::Tensor output;
        if (iv_other.isTensor()) {
            at::Tensor other_tensor = iv_other.toTensor();
            output = atenEq(self_tensor, other_tensor);
        } else if (iv_other.isScalar()) {
            at::Scalar other = iv_other.toScalar();
            output = atenEq(self_tensor, other);
        } else {
            DLOG(FATAL) << "Aten eq op's data type do not support!";
        }
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isScalar()) {
        assert(iv_other.isScalar());
        at::Scalar self_scalar = iv_self.toScalar();
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = atenEq(self_scalar, other_scalar);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, scalarToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::eq";
    }
}

void executorAtenEqual(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Equal node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor() && iv_other.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    at::Tensor other_tensor = iv_other.toTensor();

    auto output = atenEqual(self_tensor, other_tensor);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, boolToIValue(output));
}

void executorAtenExpand(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Expand node";

    auto expand_layer = std::static_pointer_cast<nn_compiler::ir::AtenExpandLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_size = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor());
    assert(iv_size.isList());

    auto self_tensor = iv_self.toTensor();
    auto size_ivalue_list = iv_size.toListRef();
    auto size_list = parseIValueVector<int64_t>(size_ivalue_list);

    int implicit = expand_layer->getImplicit();
    if (nn_compiler::ir::isDefaultValue(implicit)) {
        auto implicit_iv = stream_executor.findBlob(in_stensor_id[2]).second;
        assert(implicit_iv.isInt());
        implicit = implicit_iv.toInt();
    }

    auto output = atenExpand(self_tensor, at::ArrayRef<int64_t>(size_list), static_cast<bool>(implicit));
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenFill(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Fill node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    if (iv_other.isTensor()) {
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = atenFill(self_tensor, other_tensor);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        stream_executor.updateBlob(in_stensor_id[0], DataType::TENSOR, tensorToIValue(output));  // in-place op
    } else if (iv_other.isScalar()) {
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = atenFill(self_tensor, other_scalar);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        stream_executor.updateBlob(in_stensor_id[0], DataType::TENSOR, tensorToIValue(output));  // in-place op
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::fill_";
    }
}

void executorAtenFloorDivide(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten FloorDivide node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    if (iv_other.isTensor()) {
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = atenFloorDivide(self_tensor, other_tensor);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (iv_other.isScalar()) {
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = atenFloorDivide(self_tensor, other_scalar);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::floor_divide";
    }
}

void executorAtenFormat(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Format node";

    auto format_layer = std::static_pointer_cast<nn_compiler::ir::AtenFormatLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();
    int in_id = 0;
    auto assembly_format = format_layer->getAssemblyFormat();

    if (assembly_format == "") {
        auto assembly_format_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(assembly_format_iv.isString());
        assembly_format = assembly_format_iv.toStringRef();
    }

    // Find the input blob
    auto i_value1 = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    auto i_value2 = stream_executor.findBlob(in_stensor_id[in_id++]).second;

    auto dtype = stream_executor.findBlob(in_stensor_id[1]).first;
    if (dtype == DataType::TUPLE) {
        // aten::format(string, tuple(int,..., int), list(int,..., int))
        c10::intrusive_ptr<c10::ivalue::Tuple> i_tuple_values = i_value1.toTuple();
        auto i_list_values = i_value2.toList();
        std::vector<std::string> value1;
        std::vector<std::string> value2;
        for (auto& item : i_tuple_values->elements()) {
            value1.push_back(*(item.toString()));
        }
        for (auto item : i_list_values) {
            auto ivalue_item = static_cast<c10::IValue>(item);
            value2.push_back(*(ivalue_item.toString()));
        }
        assert(value1.size() > 0 && value2.size() > 0);
        std::string str1 = value1[0], str2 = value2[0];
        for (int idx = 1; idx < value1.size(); idx++) {
            str1 += (", " + value1[idx]);
        }
        for (int idx = 1; idx < value2.size(); idx++) {
            str2 += (", " + value2[idx]);
        }

        auto output = atenFormat(assembly_format, str1, str2);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::STRING, strToIValue(output));
    } else if (dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
               dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::INT64) {
        // aten::format(string, int, int)
        std::string str1 = std::to_string(i_value1.toInt());
        std::string str2 = std::to_string(i_value2.toInt());

        auto output = atenFormat(assembly_format, str1, str2);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::STRING, strToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::format";
    }
}

void executorAtenGather(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Gather node";

    auto gather_layer = std::static_pointer_cast<nn_compiler::ir::AtenGatherLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    auto dim = gather_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto dim_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }

    torch::jit::IValue iv_index = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_index.isTensor());
    auto index_tensor = iv_index.toTensor();

    auto sparse_grad = gather_layer->getSparseGrad();
    if (nn_compiler::ir::isDefaultValue(sparse_grad)) {
        auto sparse_grad_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(sparse_grad_iv.isInt());
        sparse_grad = static_cast<int>(sparse_grad_iv.toInt());
    }

    auto output = atenGather(self_tensor, dim, index_tensor, sparse_grad);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenGe(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Ge node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    if (iv_other.isTensor()) {
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = atenGe(self_tensor, other_tensor);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (iv_other.isScalar()) {
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = atenGe(self_tensor, other_scalar);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::ge";
    }
}

void executorAtenGetItem(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten GetItem node";

    auto get_item_layer = std::static_pointer_cast<nn_compiler::ir::AtenGetItemLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isList());
    auto self_list = iv_self.toList();

    int idx = get_item_layer->getIdx();
    if (nn_compiler::ir::isDefaultValue(idx)) {
        auto idx_iv = stream_executor.findBlob(in_stensor_id[1]).second;
        assert(idx_iv.isInt());
        idx = idx_iv.toInt();
    }

    auto output = atenGetItem(self_list, idx);
    // update output
    stream_executor.releation_blob_ids_map_.insert({out_stensor_id[0], {in_stensor_id[0], idx}});
    stream_executor.updateBlob(out_stensor_id[0], DataType::IVALUE, output);
}

void executorAtenGt(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Gt node";
    
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;

    // Find output blob
    assert(out_stensor_id.size() == 1);

    if (iv_self.isTensor() && iv_other.isTensor()) {
        // tensor = Gt(tensor, tensor)
        auto output = atenGt(iv_self.toTensor(), iv_other.toTensor());
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isTensor() && iv_other.isScalar()) {
        // tensor = Gt(tensor, scalar)
        auto output = atenGt(iv_self.toTensor(), iv_other.toScalar());
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isScalar() && iv_other.isInt()) {
        // int/bool Gt(scalar, int)
        int64_t output = iv_self.toScalar().toInt() > iv_other.toInt();
        stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, scalarToIValue<int64_t>(output));
    } else if (iv_self.isInt() && iv_other.isInt()) {
        // int/bool = Gt(int, int)
        int64_t output = iv_self.toInt() > iv_other.toInt();
        stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, scalarToIValue<int64_t>(output));
    }
}

void executorAtenIndex(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Index node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    auto indices_iv = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(indices_iv.isTensorList());
    auto indices_list_ivalue = indices_iv.toList();
    c10::List<c10::optional<at::Tensor>> indices_optional_list;
    for (torch::jit::IValue iv : indices_list_ivalue) {
        indices_optional_list.push_back(iv.toOptional<at::Tensor>());
    }

    auto output = atenIndex(self_tensor, indices_optional_list);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenIndexPut(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten IndexPut node";

    auto index_put_layer = std::static_pointer_cast<nn_compiler::ir::AtenIndexPutLayer>(layer);
    
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    auto indices_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(indices_iv.isTensorList());
    auto indices_list_ivalue = indices_iv.toList();
    c10::List<c10::optional<at::Tensor>> indices_optional_list;
    for (torch::jit::IValue iv : indices_list_ivalue) {
        indices_optional_list.push_back(iv.toOptional<at::Tensor>());
    }

    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_other.isTensor());
    auto value_tensor = iv_other.toTensor();

    auto accumulate = index_put_layer->getAccumulate();
    if (nn_compiler::ir::isDefaultValue(accumulate)) {
 
        auto accumulate_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(accumulate_iv.isInt());
        accumulate = accumulate_iv.toInt();
    }

    auto output = atenIndexPut(self_tensor, indices_optional_list, value_tensor, static_cast<bool>(accumulate));
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    stream_executor.updateBlob(in_stensor_id[0], DataType::TENSOR, tensorToIValue(output));  // in-place op
}

void executorAtenIndexSelect(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten IndexSelect node";

    auto index_select_layer = std::static_pointer_cast<nn_compiler::ir::AtenIndexSelectLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    auto dim = index_select_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto dim_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }

    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_other.isTensor());
    auto index_tensor = iv_other.toTensor();

    auto output = atenIndexSelect(self_tensor, dim, index_tensor);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenInt(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Int node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    int64_t output = -1;
    if (iv_self.isScalar()) {
        auto self_scalar = iv_self.toScalar();
        output = atenInt(self_scalar);
    } else if (iv_self.isTensor()) {
        auto self_tensor = iv_self.toTensor();
        output = atenInt(self_tensor);
    } else {
        DLOG(FATAL) << "AtenInt data type do not support!";
    }

    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::INT64, scalarToIValue(output));
}

void executorAtenIs(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Is node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();
    assert(in_stensor_id.size() == 2);

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;

    auto output = atenIs(iv_self, iv_other);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, boolToIValue(output));
}

void executorAtenItem(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Item node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    assert(in_stensor_id.size() == 1);

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    c10::Scalar output = atenItem(self_tensor);
    auto output_dtype = convertATScalarTypeToDType(output.type());

    // update output
    stream_executor.updateBlob(out_stensor_id[0], output_dtype, torch::jit::IValue(output));
}

void executorAtenLeakyRelu(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten LeakyRelu node";

    auto leaky_relu_layer = std::static_pointer_cast<nn_compiler::ir::AtenLeakyReluLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // auto node = cast<nncir::AtenLeakyReluNode>(op_node);
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    auto tensor = iv_tensor.toTensor();

    auto scalar = leaky_relu_layer->getScalar();
    if (nn_compiler::ir::isDefaultValue(scalar)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[1]).second;
        assert(data_iv.isDouble());
        scalar = data_iv.toDouble();
    }

    auto output = atenLeakyRelu(tensor, scalar);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenLen(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Len node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    assert(in_stensor_id.size() == 1);

    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_id[0]).second;

    int64_t output = -1;
    if (iv.isList()) {
        output = atenLen(iv.toList());
        // TODO(SRCX): implement this part of optimization.
        auto input1_layer = stream_executor.getGraph()->getLayerByPosition((layer->getPreLayerIDs())[0]);
        auto out1_layer = stream_executor.getGraph()->getLayerByPosition((layer->getNextLayerIDs())[0]);
        if (stream_executor.model_type_ == "GNMT" &&
            input1_layer->getType() == nn_compiler::ir::LayerType::PRIMVARIABLE &&
            iv.toList().size() == 4) {
            int next_node_id = std::static_pointer_cast<nn_compiler::ir::PrimLoopLayer>(out1_layer)->getGotoLayer() - 1;
            stream_executor.setCursor(next_node_id);
        }
    } else if (iv.isTensor()) {
        output = atenLen(iv.toTensor());
    } else {
        DLOG(FATAL) << "Aten len op's data type do not support!";
    }

    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::INT64, intToIValue(output));
}

void executorAtenLinear(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Linear node";

    auto linear_layer = std::static_pointer_cast<nn_compiler::ir::AtenLinearLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    auto tensor = iv_tensor.toTensor();

    auto weight_ids = linear_layer->getWeightIds();
    auto weight_iv = stream_executor.findBlob(weight_ids[0]).second;
    assert(weight_iv.isTensor());
    at::Tensor weight_tensor = weight_iv.toTensor();

    at::Tensor output;
    if (!linear_layer->getBiases().empty()) {
        auto bias_ids = linear_layer->getBiasIds();
        auto bias_iv = stream_executor.findBlob(bias_ids[0]).second;
        assert(bias_iv.isTensor());
        at::Tensor bias_tensor = bias_iv.toTensor();
        output = atenLinear(tensor, weight_tensor, bias_tensor);
    } else {
        output = atenLinear(tensor, weight_tensor);
    }
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenList(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten List node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    assert(in_stensor_id.size() == 1);

    torch::jit::IValue iv_list = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_list.isList());
    auto output = atenList(iv_list.toList());
    stream_executor.updateBlob(out_stensor_id[0], DataType::LIST, listToIValue(output));
}

void executorAtenLog(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Log node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    auto tensor = iv_tensor.toTensor();

    auto output = atenLog(tensor);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenLogSoftmax(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten LogSoftmax node";

    auto log_softmax_layer = std::static_pointer_cast<nn_compiler::ir::AtenLogSoftmaxLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_tensor.isTensor());
    auto tensor = iv_tensor.toTensor();

    auto dim = log_softmax_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        dim = data_iv.toInt();
    }

    auto ori_dtype = log_softmax_layer->getDType();
    bool dtype_is_none = false;
    if (nn_compiler::ir::isDefaultValue(ori_dtype)) {
        auto ori_dtype_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (ori_dtype_iv.isInt())
            ori_dtype = ori_dtype_iv.toInt();
        else if (ori_dtype_iv.isNone()) {
            dtype_is_none = true;
        }
    }

    torch::Tensor output;
    if (dtype_is_none) {
        output = atenLogSoftmax(tensor, dim);
    } else {
        output = atenLogSoftmax(tensor, dim, at::ScalarType(ori_dtype));
    }

    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenLSTM1(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten LSTM1 node";

    auto lstm1_layer = std::static_pointer_cast<nn_compiler::ir::AtenLSTM1Layer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    // const at::Tensor &input
    auto input_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(input_iv.isTensor());
    auto input = input_iv.toTensor();

    // at::TensorList hx
    auto hx_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(hx_iv.isTensorList());
    auto hx_list_tensor = hx_iv.toTensorList();
    std::vector<at::Tensor> hx_list_tensor_vector;
    for (auto tensor : hx_list_tensor) {
        hx_list_tensor_vector.push_back(tensor);
    }
    at::TensorList hx(hx_list_tensor_vector);

    // Check and skip, will handle params after getting all arguments
    if (in_stensor_id.size() > in_id) {
        auto params_iv = stream_executor.findBlob(in_stensor_id[in_id]).second;
        if (params_iv.isTensorList()) {
            in_id++;
        }
    }

    // bool has_biases
    int has_biases = lstm1_layer->getHasBiases();
    if (nn_compiler::ir::isDefaultValue(has_biases)) {
        auto has_biases_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(has_biases_iv.isInt());
        has_biases = has_biases_iv.toInt();
    }

    // int64_t num_layers
    int64_t num_layers = lstm1_layer->getNumLayers();
    if (nn_compiler::ir::isDefaultValue(num_layers)) {
        auto num_layers_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(num_layers_iv.isInt());
        num_layers = num_layers_iv.toInt();
    }

    // double dropout
    double dropout = lstm1_layer->getDropout();
    if (nn_compiler::ir::isDefaultValue(dropout)) {
        auto dropout_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(dropout_iv.isDouble());
        dropout = dropout_iv.toDouble();
    }

    // bool train
    int train = lstm1_layer->getTrain();
    if (nn_compiler::ir::isDefaultValue(train)) {
        auto train_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(train_iv.isInt());
        train = train_iv.toInt();
    }

    // bool bidirectional
    int bidirectional = lstm1_layer->getBidirectional();
    if (nn_compiler::ir::isDefaultValue(bidirectional)) {
        auto bidirectional_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(bidirectional_iv.isInt());
        bidirectional = bidirectional_iv.toInt();
    }

    // bool batch_first
    int batch_first = lstm1_layer->getBatchFirst();
    if (nn_compiler::ir::isDefaultValue(batch_first)) {
        auto batch_first_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(batch_first_iv.isInt());
        batch_first = batch_first_iv.toInt();
    }

    // at::TensorList params
    // param layerout                --> (fw_w_ih, fw_w_hh, fw_b_ih?, fw_b_hh?) * layers
    // param layerout (bidirctional) --> (fw_w_ih, fw_w_hh, fw_b_ih?, fw_b_hh?, bw_w_ih, bw_w_hh, bw_b_ih?, bw_b_hh?) *
    // layers

    auto weight_ids = lstm1_layer->getWeightIds();
    auto bias_ids = lstm1_layer->getBiasIds();
    std::vector<at::Tensor> param_vector;
    assert((bidirectional == 0 || bidirectional == 1));
    int hash_id = 0;
    for (int i = 0; i < num_layers * (bidirectional + 1); i++) {
        // w_ih
        auto w_ih_iv = stream_executor.findBlob(weight_ids[i * 2]).second;
        hash_id += weight_ids[i * 2];
        if (w_ih_iv.isTensor()) {
            param_vector.push_back(w_ih_iv.toTensor());
        }
        // w_hh
        auto w_hh_iv = stream_executor.findBlob(weight_ids[i * 2 + 1]).second;
        hash_id += weight_ids[i * 2 + 1];
        if (w_hh_iv.isTensor()) {
            param_vector.push_back(w_hh_iv.toTensor());
        }
        if (has_biases) {
            // b_ih? (optional)
            auto b_ih_iv = stream_executor.findBlob(bias_ids[i * 2]).second;
            hash_id += bias_ids[i * 2];
            if (b_ih_iv.isTensor()) {
                param_vector.push_back(b_ih_iv.toTensor());
            }
            // b_hh? (optional)
            auto b_hh_iv = stream_executor.findBlob(bias_ids[i * 2 + 1]).second;
            hash_id += bias_ids[i * 2 + 1];
            if (b_hh_iv.isTensor()) {
                param_vector.push_back(b_hh_iv.toTensor());
            }
        }
    }
    at::TensorList params(param_vector);
    auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
    out_stensor_id.erase(pos, out_stensor_id.end());
    assert(out_stensor_id.size() == 3);
    {
        if (!input.is_contiguous()) input = input.contiguous();
        if (!static_cast<bool>(batch_first)) input = input.transpose(0, 1);
        void *in_dev, *hx_dev, *out_dev, *wei_dev, *cx_dev, *workspace_dev, *hy_dev, *cy_dev;

        stream_executor.input_tensors_.clear();
        stream_executor.output_tensors_.clear();

        int batch_size = 1;
        int in_dim = input.dim();
        int input_size = input.size(in_dim - 1);
        int seq_len = input.size(in_dim - 2);
        int bidirectional_int = static_cast<bool>(bidirectional) ? 2 : 1;

        int hx_dim = hx_list_tensor_vector[0].dim();
        int hidden_size = hx_list_tensor_vector[0].size(hx_dim - 1);

        std::vector<int> in_len({batch_size, input_size});
        std::vector<int> hid_len({bidirectional_int *  (int)(num_layers),  hidden_size});
        std::vector<int> out_len({bidirectional_int * hidden_size});

        int dims = 2;
        for (int i = 0; i < seq_len; i++) {
            std::array<int, 2> in_lens = {in_len[0],  in_len.back() };
            miopenCreateTensorDescriptor(&stream_executor.input_tensor_);
            miopenSetTensorDescriptor(stream_executor.input_tensor_, miopenHalf, dims, in_lens.data(), nullptr);
            stream_executor.input_tensors_.push_back(stream_executor.input_tensor_);

            std::array<int, 2> out_lens = {{in_len[0], out_len[0]}};
            miopenCreateTensorDescriptor(&stream_executor.output_tensor_);
            miopenSetTensorDescriptor(stream_executor.output_tensor_, miopenHalf, dims, out_lens.data(), nullptr);
            stream_executor.output_tensors_.push_back(stream_executor.output_tensor_);
        }
        std::array<int, 3> hid_lens = {{hid_len[0], in_len[0], hid_len[1]}};
        miopenSetTensorDescriptor(stream_executor.hidden_tensor_, miopenHalf, 3, hid_lens.data(), nullptr);

        miopenRNNMode_t mode = miopenRNNMode_t::miopenLSTM;;
        miopenRNNBiasMode_t biasMode = static_cast<bool>(has_biases) ? miopenRNNwithBias : miopenRNNNoBias;
        miopenRNNDirectionMode_t directionMode = bidirectional_int == 2 ? miopenRNNbidirection : miopenRNNunidirection;
        miopenRNNInputMode_t inMode = miopenRNNlinear;
        miopenRNNAlgo_t algo = miopenRNNdefault;

        miopenSetRNNDescriptor(stream_executor.rnn_desc_, hidden_size, num_layers, inMode, directionMode, mode, biasMode, algo, miopenHalf);
        miopenGetRNNParamsDescriptor(stream_executor.handle_, stream_executor.rnn_desc_, stream_executor.input_tensor_, stream_executor.weight_tensor_, miopenHalf);
        size_t workspace_size;
        miopenGetRNNWorkspaceSize(stream_executor.handle_, stream_executor.rnn_desc_, seq_len, stream_executor.input_tensors_.data(), &workspace_size);
        auto workspace = at::empty(workspace_size, input.options().dtype(at::kByte));

        int datasize = 2; //miopenHalf
        in_dev = input.data_ptr();


        hash_id += 10000; // avert id conflict
        auto it = stream_executor.global_blobs_.find(hash_id);
        if (it == stream_executor.global_blobs_.end()) {
            size_t weight_size = 0;
            miopenGetRNNParamsSize(stream_executor.handle_, stream_executor.rnn_desc_, stream_executor.input_tensor_, &weight_size, miopenHalf);
            auto weight_buf = at::empty(weight_size / datasize, input.options());
            int expected_weight_size = hidden_size * 4 * (input_size + hidden_size + 2) * bidirectional_int + hidden_size * 4 * (hidden_size + hidden_size + 2) * (num_layers - 1) * bidirectional_int;
            assert((weight_size / datasize) == expected_weight_size);

            size_t offset = 0;
            size_t param_size = 0;

            offset = 0;
            param_size = 0;

            int param_num = static_cast<bool>(has_biases) ? 4 * num_layers * bidirectional_int: 2 * num_layers * bidirectional_int;
            int num = 0;
            int num_offset = static_cast<bool>(has_biases) ? 4 * bidirectional_int : 2 * bidirectional_int;
            for (; num < param_num; num += num_offset) {
                param_size = 1;
                auto in_wei_vec = param_vector[num].sizes().vec();
                for (int i = 0; i < in_wei_vec.size(); ++i) {
                    param_size *= in_wei_vec[i];
                }

                at::Tensor param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                auto sliced_tensor = param_vector[num].chunk(4, 0);
                auto permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                param.copy_(permuted_wei.view_as(param));
                offset += param_size;

                if (bidirectional_int == 2) {
                    param_size = 1;
                    in_wei_vec = param_vector[num + (num_offset / bidirectional_int)].sizes().vec();
                    for (int i = 0; i < in_wei_vec.size(); ++i) {
                        param_size *= in_wei_vec[i];
                    }
                    param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                    sliced_tensor = param_vector[num + (num_offset / bidirectional_int)].chunk(4, 0);
                    permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                    param.copy_(permuted_wei.view_as(param));
                    offset += param_size;
                }

                param_size = 1;
                in_wei_vec = param_vector[num + 1].sizes().vec();
                for (int i = 0; i < in_wei_vec.size(); ++i) {
                    param_size *= in_wei_vec[i];
                }
                param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                sliced_tensor = param_vector[num + 1].chunk(4, 0);
                permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                param.copy_(permuted_wei.view_as(param));
                offset += param_size;

                if (bidirectional_int == 2) {
                    param_size = 1;
                    in_wei_vec = param_vector[num + (num_offset / bidirectional_int) + 1].sizes().vec();
                    for (int i = 0; i < in_wei_vec.size(); ++i) {
                        param_size *= in_wei_vec[i];
                    }
                    param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                    sliced_tensor = param_vector[num + (num_offset / bidirectional_int) + 1].chunk(4, 0);
                    permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                    param.copy_(permuted_wei.view_as(param));
                    offset += param_size;
                }
            }
            if (static_cast<bool>(has_biases)) {
                num = 2;
                for (; num < param_num; num += num_offset) {
                    param_size = 1;
                    auto in_wei_vec = param_vector[num].sizes().vec();
                    for (int i = 0; i < in_wei_vec.size(); ++i) {
                        param_size *= in_wei_vec[i];
                    }
                    at::Tensor param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                    auto sliced_tensor = param_vector[num].chunk(4, 0);
                    auto permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                    param.copy_(permuted_wei.view_as(param));
                    offset += param_size;

                    if (bidirectional_int == 2) {
                        param_size = 1;
                        in_wei_vec = param_vector[num + (num_offset / bidirectional_int)].sizes().vec();
                        for (int i = 0; i < in_wei_vec.size(); ++i) {
                            param_size *= in_wei_vec[i];
                        }
                        param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                        sliced_tensor = param_vector[num + (num_offset / bidirectional_int)].chunk(4, 0);
                        permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                        param.copy_(permuted_wei.view_as(param));
                        offset += param_size;
                    }

                    param_size = 1;
                    in_wei_vec = param_vector[num + 1].sizes().vec();
                    for (int i = 0; i < in_wei_vec.size(); ++i) {
                        param_size *= in_wei_vec[i];
                    }
                    param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                    sliced_tensor = param_vector[num + 1].chunk(4, 0);
                    permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                    param.copy_(permuted_wei.view_as(param));
                    offset += param_size;

                    if (bidirectional_int == 2) {
                        param_size = 1;
                        in_wei_vec = param_vector[num + (num_offset / bidirectional_int) + 1].sizes().vec();
                        for (int i = 0; i < in_wei_vec.size(); ++i) {
                            param_size *= in_wei_vec[i];
                        }
                        param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                        sliced_tensor = param_vector[num + (num_offset / bidirectional_int) + 1].chunk(4, 0);
                        permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                        param.copy_(permuted_wei.view_as(param));
                        offset += param_size;
                    }
                }
            }
            wei_dev = weight_buf.data_ptr();
            stream_executor.updateBlob(hash_id, DataType::TENSOR, tensorToIValue(weight_buf));
        } else {
            wei_dev = it->second.second.toTensor().data_ptr();
        }
        in_dev = input.data_ptr();
        hx_dev = hx_list_tensor_vector[0].data_ptr();
        cx_dev = hx_list_tensor_vector[1].data_ptr();
        workspace_dev = workspace.data_ptr();

        auto it0 = stream_executor.global_blobs_.find(out_stensor_id[0]);

        if (stream_executor.model_type_ == "GNMT" && it0 != stream_executor.global_blobs_.end() && seq_len == 1) {
            out_dev = it0->second.second.toTensor().data_ptr();
            hy_dev = stream_executor.global_blobs_.find(out_stensor_id[1])->second.second.toTensor().data_ptr();
            cy_dev = stream_executor.global_blobs_.find(out_stensor_id[2])->second.second.toTensor().data_ptr();
            miopenRNNForwardInference(stream_executor.handle_, stream_executor.rnn_desc_, seq_len, stream_executor.input_tensors_.data(), in_dev,
                                  stream_executor.hidden_tensor_, hx_dev, stream_executor.hidden_tensor_, cx_dev, stream_executor.weight_tensor_, wei_dev,
                                  stream_executor.output_tensors_.data(), out_dev, stream_executor.hidden_tensor_, hy_dev, stream_executor.hidden_tensor_, cy_dev,
                                  workspace_dev, workspace_size);
        } else {
            auto output = at::empty({in_len[0], seq_len, out_len[0]}, input.options());
            out_dev = output.data_ptr();
            at::Tensor hy, cy;
            {
                size_t hc_y_size = hid_lens[0] * hid_lens[1] * hid_lens[2];
                auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
                hy = at::empty({hid_lens[0], hid_lens[1], hid_lens[2]}, options);
                cy = at::empty({hid_lens[0], hid_lens[1], hid_lens[2]}, options);

                hy_dev = hy.data_ptr();
                cy_dev = cy.data_ptr();

                if (stream_executor.model_type_ == "GNMT" && lstm1_layer->getMatchCustomOpt()) {
                    int cat_f = 22222;
                    auto cat = stream_executor.global_blobs_.find(cat_f);
                    auto cat_mem = cat->second.second.toTensor();

                    if (lstm1_layer->getCustomOptNumber() == 0) {
                        hy = torch::from_blob((_Float16*)(cat_mem.data_ptr()), {1, 1, 1024}, options);
                        cy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 1024, {1, 1, 1024}, options);
                        hy_dev = hy.data_ptr();
                        cy_dev = cy.data_ptr();
                    }else if (lstm1_layer->getCustomOptNumber() == 1) {
                        hy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 2048, {1, 1, 1024}, options);
                        cy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 3072, {1, 1, 1024}, options);
                        hy_dev = hy.data_ptr();
                        cy_dev = cy.data_ptr();
                    }else if (lstm1_layer->getCustomOptNumber() == 2) {
                        hy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 4096, {1, 1, 1024}, options);
                        cy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 5120, {1, 1, 1024}, options);
                        hy_dev = hy.data_ptr();
                        cy_dev = cy.data_ptr();
                    }else if (lstm1_layer->getCustomOptNumber() == 3) {
                        hy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 6144, {1, 1, 1024}, options);
                        cy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 7168, {1, 1, 1024}, options);
                        hy_dev = hy.data_ptr();
                        cy_dev = cy.data_ptr();
                    }
                    if (stream_executor.model_type_ == "GNMT" && lstm1_layer->getCustomOptNumber() == 0) {
                        auto out4_layer = stream_executor.getGraph()->getLayerByPosition((layer->getNextLayerIDs())[3]);
                        auto out4_out1_layer = stream_executor.getGraph()->getLayerByPosition((out4_layer->getNextLayerIDs())[0]);
                        int64_t cat_mem_id =
                            std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(out4_out1_layer)->getMemLayerId();
                        auto it = stream_executor.global_blobs_.find(cat_mem_id);
                        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
                        auto cat_mem = it->second.second.toTensor();
                        output = torch::from_blob((_Float16*)(cat_mem.data_ptr()), {1, 1, 1024}, options);
                    }
                    if (stream_executor.model_type_ == "GNMT" && lstm1_layer->getCustomOptNumber() == 1) {
                        auto out1_layer = stream_executor.getGraph()->getLayerByPosition((layer->getNextLayerIDs())[0]);
                        auto out1_out1_layer = stream_executor.getGraph()->getLayerByPosition((out1_layer->getNextLayerIDs())[0]);
                        int64_t cat_mem_id = std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(out1_out1_layer)->getMemLayerId();
                        auto it = stream_executor.global_blobs_.find(cat_mem_id);
                        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
                        auto cat_mem = it->second.second.toTensor();
                        output = torch::from_blob((_Float16*)(cat_mem.data_ptr()), {1, 1, 1024}, options);
                    }
                }
            }
            miopenRNNForwardInference(stream_executor.handle_, stream_executor.rnn_desc_, seq_len, stream_executor.input_tensors_.data(), in_dev,
                                    stream_executor.hidden_tensor_, hx_dev, stream_executor.hidden_tensor_, cx_dev, stream_executor.weight_tensor_, wei_dev,
                                    stream_executor.output_tensors_.data(), out_dev, stream_executor.hidden_tensor_, hy_dev, stream_executor.hidden_tensor_, cy_dev,
                                    workspace_dev, workspace_size);

            if (!static_cast<bool>(batch_first)) output = output.transpose(0, 1);
            stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
            stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(hy));
            stream_executor.updateBlob(out_stensor_id[2], DataType::TENSOR, tensorToIValue(cy));
        }
    }
}

void executorAtenLSTM2(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten LSTM2 node";

    auto lstm2_layer = std::static_pointer_cast<nn_compiler::ir::AtenLSTM2Layer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    // const at::Tensor &input
    auto input_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(input_iv.isTensor());
    auto input = input_iv.toTensor();

    // const at::Tensor &batch_sizes,
    at::Tensor batch_sizes;
    auto batch_sizes_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(batch_sizes_iv.isTensor());
    batch_sizes = batch_sizes_iv.toTensor();

    // at::TensorList hx
    auto hx_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(hx_iv.isTensorList());
    auto hx_list_tensor = hx_iv.toTensorList();
    std::vector<at::Tensor> hx_list_tensor_vector;
    for (auto tensor : hx_list_tensor) {
        hx_list_tensor_vector.push_back(tensor);
    }
    at::TensorList hx(hx_list_tensor_vector);

    // at::TensorList params
    // Check and skip, will handle params after getting all arguments
    if (in_stensor_id.size() > 2) {
        auto params_iv = stream_executor.findBlob(in_stensor_id[in_id]).second;
        if (params_iv.isTensorList()) {
            in_id++;
        }
    }

    // bool has_biases
    int has_biases = lstm2_layer->getHasBiases();
    if (nn_compiler::ir::isDefaultValue(has_biases)) {
        auto has_biases_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(has_biases_iv.isInt());
        has_biases = has_biases_iv.toInt();
    }

    // int64_t num_layers
    int64_t num_layers = lstm2_layer->getNumLayers();
    if (nn_compiler::ir::isDefaultValue(num_layers)) {
        auto num_layers_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(num_layers_iv.isInt());
        num_layers = num_layers_iv.toInt();
    }

    // double dropout
    double dropout = lstm2_layer->getDropout();
    if (nn_compiler::ir::isDefaultValue(dropout)) {
        auto dropout_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(dropout_iv.isDouble());
        dropout = dropout_iv.toDouble();
    }

    // bool train
    int train = lstm2_layer->getTrain();
    if (nn_compiler::ir::isDefaultValue(train)) {
        auto train_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(train_iv.isInt());
        train = train_iv.toInt();
    }

    // bool bidirectional
    int bidirectional = lstm2_layer->getBidirectional();
    if (nn_compiler::ir::isDefaultValue(bidirectional)) {
        auto bidirectional_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(bidirectional_iv.isInt());
        bidirectional = bidirectional_iv.toInt();
    }

    // at::TensorList params
    // param layerout                --> (fw_w_ih, fw_w_hh, fw_b_ih?, fw_b_hh?) * layers
    // param layerout (bidirctional) --> (fw_w_ih, fw_w_hh, fw_b_ih?, fw_b_hh?, bw_w_ih, bw_w_hh, bw_b_ih?, bw_b_hh?) *
    // layers

    auto weight_ids = lstm2_layer->getWeightIds();
    auto bias_ids = lstm2_layer->getBiasIds();
    std::vector<at::Tensor> param_vector;
    assert((bidirectional == 0 || bidirectional == 1));

    int hash_id = 0;
    for (int i = 0; i < num_layers * (bidirectional + 1); i++) {
        // w_ih
        auto w_ih_iv = stream_executor.findBlob(weight_ids[i * 2]).second;
        hash_id += weight_ids[i * 2];
        if (w_ih_iv.isTensor()) {
            param_vector.push_back(w_ih_iv.toTensor());
        }
        // w_hh
        auto w_hh_iv = stream_executor.findBlob(weight_ids[i * 2 + 1]).second;
        hash_id += weight_ids[i * 2 + 1];
        if (w_hh_iv.isTensor()) {
            param_vector.push_back(w_hh_iv.toTensor());
        }
        if (has_biases) {
            // b_ih? (optional)
            auto b_ih_iv = stream_executor.findBlob(bias_ids[i * 2]).second;
            hash_id += bias_ids[i * 2];
            if (b_ih_iv.isTensor()) {
                param_vector.push_back(b_ih_iv.toTensor());
            }
            // b_hh? (optional)
            auto b_hh_iv = stream_executor.findBlob(bias_ids[i * 2 + 1]).second;
            hash_id += bias_ids[i * 2 + 1];
            if (b_hh_iv.isTensor()) {
                param_vector.push_back(b_hh_iv.toTensor());
            }
        }
    }
    at::TensorList params(param_vector);

    auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
    out_stensor_id.erase(pos, out_stensor_id.end());
    assert(out_stensor_id.size() == 3);
    {
        if (!input.is_contiguous()) input = input.contiguous();
        void *in_dev, *hx_dev, *out_dev, *wei_dev, *cx_dev, *workspace_dev, *hy_dev, *cy_dev;
        stream_executor.input_tensors_.clear();
        stream_executor.output_tensors_.clear();

        int batch_size = 1;
        int in_dim = input.dim();
        int input_size = input.size(in_dim - 1);
        int seq_len = input.size(in_dim - 2);
        int bidirectional_int = static_cast<bool>(bidirectional) ? 2 : 1;

        int hx_dim = hx_list_tensor_vector[0].dim();
        int hidden_size = hx_list_tensor_vector[0].size(hx_dim - 1);

        std::vector<int> in_len({batch_size, input_size});
        std::vector<int> hid_len({bidirectional_int *  (int)(num_layers),  hidden_size});
        std::vector<int> out_len({bidirectional_int * hidden_size});

        int dims = 2;
        for (int i = 0; i < seq_len; i++) {
            std::array<int, 2> in_lens = {in_len[0],  in_len.back() };
            miopenCreateTensorDescriptor(&stream_executor.input_tensor_);
            miopenSetTensorDescriptor(stream_executor.input_tensor_, miopenHalf, dims, in_lens.data(), nullptr);
            stream_executor.input_tensors_.push_back(stream_executor.input_tensor_);

            std::array<int, 2> out_lens = {{in_len[0], out_len[0]}};
            miopenCreateTensorDescriptor(&stream_executor.output_tensor_);
            miopenSetTensorDescriptor(stream_executor.output_tensor_, miopenHalf, dims, out_lens.data(), nullptr);
            stream_executor.output_tensors_.push_back(stream_executor.output_tensor_);
        }
        std::array<int, 3> hid_lens = {{hid_len[0], in_len[0], hid_len[1]}};
        miopenSetTensorDescriptor(stream_executor.hidden_tensor_, miopenHalf, 3, hid_lens.data(), nullptr);

        miopenRNNMode_t mode = miopenRNNMode_t::miopenLSTM;;
        miopenRNNBiasMode_t biasMode = static_cast<bool>(has_biases) ? miopenRNNwithBias : miopenRNNNoBias;
        miopenRNNDirectionMode_t directionMode = bidirectional_int == 2 ? miopenRNNbidirection : miopenRNNunidirection;
        miopenRNNInputMode_t inMode = miopenRNNlinear;
        miopenRNNAlgo_t algo = miopenRNNdefault;

        miopenSetRNNDescriptor(stream_executor.rnn_desc_, hidden_size, num_layers, inMode, directionMode, mode, biasMode, algo, miopenHalf);
        miopenGetRNNParamsDescriptor(stream_executor.handle_, stream_executor.rnn_desc_, stream_executor.input_tensor_, stream_executor.weight_tensor_, miopenHalf);
        size_t workspace_size;
        miopenGetRNNWorkspaceSize(stream_executor.handle_, stream_executor.rnn_desc_, seq_len, stream_executor.input_tensors_.data(), &workspace_size);
        auto workspace = at::empty(workspace_size, input.options().dtype(at::kByte));

        int datasize = 2; //miopenHalf
        in_dev = input.data_ptr();

        hash_id += 10000; // avert id conflict
        auto it = stream_executor.global_blobs_.find(hash_id);
        if (it == stream_executor.global_blobs_.end()) {
            size_t weight_size = 0;
            miopenGetRNNParamsSize(stream_executor.handle_, stream_executor.rnn_desc_, stream_executor.input_tensor_, &weight_size, miopenHalf);
            auto weight_buf = at::empty(weight_size / datasize, input.options());
            int expected_weight_size = hidden_size * 4 * (input_size + hidden_size + 2) * bidirectional_int + hidden_size * 4 * (hidden_size + hidden_size + 2) * (num_layers - 1) * bidirectional_int;
            assert((weight_size / datasize) == expected_weight_size);
            size_t offset = 0;
            size_t param_size = 0;

            offset = 0;
            param_size = 0;

            int param_num = static_cast<bool>(has_biases) ? 4 * num_layers * bidirectional_int: 2 * num_layers * bidirectional_int;
            int num = 0;
            int num_offset = static_cast<bool>(has_biases) ? 4 * bidirectional_int : 2 * bidirectional_int;
            for (; num < param_num; num += num_offset) {
                param_size = 1;
                auto in_wei_vec = param_vector[num].sizes().vec();
                for (int i = 0; i < in_wei_vec.size(); ++i) {
                    param_size *= in_wei_vec[i];
                }

                at::Tensor param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                auto sliced_tensor = param_vector[num].chunk(4, 0);
                auto permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                param.copy_(permuted_wei.view_as(param));
                offset += param_size;

                if (bidirectional_int == 2) {
                    param_size = 1;
                    in_wei_vec = param_vector[num + (num_offset / bidirectional_int)].sizes().vec();
                    for (int i = 0; i < in_wei_vec.size(); ++i) {
                        param_size *= in_wei_vec[i];
                    }
                    param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                    sliced_tensor = param_vector[num + (num_offset / bidirectional_int)].chunk(4, 0);
                    permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                    param.copy_(permuted_wei.view_as(param));
                    offset += param_size;
                }

                param_size = 1;
                in_wei_vec = param_vector[num + 1].sizes().vec();
                for (int i = 0; i < in_wei_vec.size(); ++i) {
                    param_size *= in_wei_vec[i];
                }
                param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                sliced_tensor = param_vector[num + 1].chunk(4, 0);
                permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                param.copy_(permuted_wei.view_as(param));
                offset += param_size;

                if (bidirectional_int == 2) {
                    param_size = 1;
                    in_wei_vec = param_vector[num + (num_offset / bidirectional_int) + 1].sizes().vec();
                    for (int i = 0; i < in_wei_vec.size(); ++i) {
                        param_size *= in_wei_vec[i];
                    }
                    param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                    sliced_tensor = param_vector[num + (num_offset / bidirectional_int) + 1].chunk(4, 0);
                    permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                    param.copy_(permuted_wei.view_as(param));
                    offset += param_size;
                }
            }
            if (static_cast<bool>(has_biases)) {
                num = 2;
                for (; num < param_num; num += num_offset) {
                    param_size = 1;
                    auto in_wei_vec = param_vector[num].sizes().vec();
                    for (int i = 0; i < in_wei_vec.size(); ++i) {
                        param_size *= in_wei_vec[i];
                    }
                    at::Tensor param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                    auto sliced_tensor = param_vector[num].chunk(4, 0);
                    auto permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                    param.copy_(permuted_wei.view_as(param));
                    offset += param_size;

                    if (bidirectional_int == 2) {
                        param_size = 1;
                        in_wei_vec = param_vector[num + (num_offset / bidirectional_int)].sizes().vec();
                        for (int i = 0; i < in_wei_vec.size(); ++i) {
                            param_size *= in_wei_vec[i];
                        }
                        param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                        sliced_tensor = param_vector[num + (num_offset / bidirectional_int)].chunk(4, 0);
                        permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                        param.copy_(permuted_wei.view_as(param));
                        offset += param_size;
                    }

                    param_size = 1;
                    in_wei_vec = param_vector[num + 1].sizes().vec();
                    for (int i = 0; i < in_wei_vec.size(); ++i) {
                        param_size *= in_wei_vec[i];
                    }
                    param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                    sliced_tensor = param_vector[num + 1].chunk(4, 0);
                    permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                    param.copy_(permuted_wei.view_as(param));
                    offset += param_size;

                    if (bidirectional_int == 2) {
                        param_size = 1;
                        in_wei_vec = param_vector[num + (num_offset / bidirectional_int) + 1].sizes().vec();
                        for (int i = 0; i < in_wei_vec.size(); ++i) {
                            param_size *= in_wei_vec[i];
                        }
                        param = at::from_blob((_Float16*)(weight_buf.data_ptr()) + offset, {static_cast<long>(param_size)}, weight_buf.options());
                        sliced_tensor = param_vector[num + (num_offset / bidirectional_int) + 1].chunk(4, 0);
                        permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                        param.copy_(permuted_wei.view_as(param));
                        offset += param_size;
                    }
                }
            }
            wei_dev = weight_buf.data_ptr();
            stream_executor.updateBlob(hash_id, DataType::TENSOR, tensorToIValue(weight_buf));
        } else {
            wei_dev = it->second.second.toTensor().data_ptr();
        }
        in_dev = input.data_ptr();
        hx_dev = hx_list_tensor_vector[0].data_ptr();
        cx_dev = hx_list_tensor_vector[1].data_ptr();
        workspace_dev = workspace.data_ptr();
        auto it0 = stream_executor.global_blobs_.find(out_stensor_id[0]);

        if (stream_executor.model_type_ == "GNMT" && it0 != stream_executor.global_blobs_.end() && seq_len == 1) {
            out_dev = it0->second.second.toTensor().data_ptr();
            hy_dev = stream_executor.global_blobs_.find(out_stensor_id[1])->second.second.toTensor().data_ptr();
            cy_dev = stream_executor.global_blobs_.find(out_stensor_id[2])->second.second.toTensor().data_ptr();
            miopenRNNForwardInference(stream_executor.handle_, stream_executor.rnn_desc_, seq_len, stream_executor.input_tensors_.data(), in_dev,
                                  stream_executor.hidden_tensor_, hx_dev, stream_executor.hidden_tensor_, cx_dev, stream_executor.weight_tensor_, wei_dev,
                                  stream_executor.output_tensors_.data(), out_dev, stream_executor.hidden_tensor_, hy_dev, stream_executor.hidden_tensor_, cy_dev,
                                  workspace_dev, workspace_size);

        } else {
            auto output = at::empty({seq_len, out_len[0]}, input.options());
            out_dev = output.data_ptr();
            at::Tensor hy, cy;
            {
                size_t hc_y_size = hid_lens[0] * hid_lens[1] * hid_lens[2];
                auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
                hy = at::empty({hid_lens[0], hid_lens[1], hid_lens[2]}, options);
                cy = at::empty({hid_lens[0], hid_lens[1], hid_lens[2]}, options);

                hy_dev = hy.data_ptr();
                cy_dev = cy.data_ptr();
            }
            miopenRNNForwardInference(stream_executor.handle_, stream_executor.rnn_desc_, seq_len, stream_executor.input_tensors_.data(), in_dev,
                                    stream_executor.hidden_tensor_, hx_dev, stream_executor.hidden_tensor_, cx_dev, stream_executor.weight_tensor_, wei_dev,
                                    stream_executor.output_tensors_.data(), out_dev, stream_executor.hidden_tensor_, hy_dev, stream_executor.hidden_tensor_, cy_dev,
                                    workspace_dev, workspace_size);

            stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
            stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(hy));
            stream_executor.updateBlob(out_stensor_id[2], DataType::TENSOR, tensorToIValue(cy));
        }
    }
}

void executorAtenLt(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Lt node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;

    assert(out_stensor_id.size() == 1);

    if (iv_self.isTensor() && iv_other.isTensor()) {
        // tensor = Lt(tensor, tensor)
        auto output = atenLt(iv_self.toTensor(), iv_other.toTensor());
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isTensor() && iv_other.isScalar()) {
        // tensor = Lt(tensor, scalar)
        auto output = atenLt(iv_self.toTensor(), iv_other.toScalar());
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isScalar() && iv_other.isInt()) {
        // int/bool Lt(scalar, int)
        int64_t output = iv_self.toScalar().toInt() < iv_other.toInt();
        stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, scalarToIValue<int64_t>(output));
    } else if (iv_self.isInt() && iv_other.isInt()) {
        // int/bool = Lt(int, int)
        int64_t output = iv_self.toInt() < iv_other.toInt();
        stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, scalarToIValue<int64_t>(output));
    }
}

void executorAtenMaskedFill(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten MaskedFill node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    auto mask_fill_layer = std::static_pointer_cast<nn_compiler::ir::AtenMaskedFillLayer>(layer);

    // Find the input blob
    auto iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    auto iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    auto iv_value = stream_executor.findBlob(in_stensor_id[2]).second;
    assert(iv_self.isTensor() && iv_other.isTensor());
    auto self_tensor = iv_self.toTensor();
    auto other_tensor = iv_other.toTensor();

    at::Tensor output;
    if (iv_value.isTensor()) {
        auto value_tensor = iv_value.toTensor();
        output = atenMaskedFill(self_tensor, other_tensor, value_tensor);
    } else if (iv_value.isScalar()) {
        at::Scalar value_scalar = iv_value.toScalar();
        output = atenMaskedFill(self_tensor, other_tensor, value_scalar);
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::masked_fill";
    }

    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    bool is_inplace = mask_fill_layer->getIsInplace();
    if (is_inplace) {
        auto releation_blob_id = stream_executor.releation_blob_ids_map_.find(in_stensor_id[0]);
        assert(releation_blob_id != stream_executor.releation_blob_ids_map_.end());
        auto ori_id = releation_blob_id->second.first;
        stream_executor.updateBlob(ori_id, DataType::TENSOR, tensorToIValue(output));
    }
}

void executorAtenMaskedSelect(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten MaskedSelect node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    auto iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    auto iv_mask = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor() && iv_mask.isTensor());
    auto self_tensor = iv_self.toTensor();
    auto mask_tensor = iv_mask.toTensor();

    auto output = atenMaskedSelect(self_tensor, mask_tensor);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenMatmul(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Matmul node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor());
    assert(iv_other.isTensor());

    int dim_i0 = iv_self.toTensor().dim();
    int dim_i1 = iv_other.toTensor().dim();
    int i0_is_vector = 0;
    int i1_is_vector = 0;

    for (int i = 0; i < dim_i0; ++i) {
        if (iv_self.toTensor().size(i) != 1) {
            i0_is_vector += 1;
        }
    }

    for (int i = 0; i < dim_i1; ++i) {
        if (iv_other.toTensor().size(i) != 1) {
            i1_is_vector += 1;
        }
    }
    if (i0_is_vector == 1 && i1_is_vector != 1) {
        auto self = iv_self.toTensor();
        auto other = iv_other.toTensor();

        if (!self.is_contiguous()) self = self.contiguous();
        if (!other.is_contiguous()) other = other.contiguous();

        int n = other.size(dim_i1 - 1);
        int k = other.size(dim_i1 - 2);

        _Float16* x = (_Float16*)self.data_ptr();
        _Float16* A = (_Float16*)other.data_ptr();
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto output_shape = self.sizes().vec();
        if (dim_i1 > dim_i0) {
            output_shape = other.sizes().vec();
            output_shape[dim_i1 - 1] = n;
            output_shape[dim_i1 - 2] = 1;
        } else {
            output_shape[dim_i0 - 1] = n;
            output_shape[dim_i0 - 2] = 1;
        }
        auto output = at::zeros(output_shape, options);
        _Float16* y = (_Float16*)output.data_ptr();

        PimDesc* pim_desc = PimCreateDesc(1, 1, n, k, PIM_FP16, OP_GEMV);
        PimBo* dev_op0 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT, x);
        PimBo* dev_op1 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT, A);
        PimBo* dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT, y);

        PimExecuteGemv(dev_out, dev_op0, dev_op1, nullptr);

        PimDestroyBo(dev_op0);
        PimDestroyBo(dev_op1);
        PimDestroyBo(dev_out);
        PimDestroyDesc(pim_desc);

        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (i0_is_vector != 1 && i1_is_vector == 1) {
        auto self = iv_self.toTensor();
        auto other = iv_other.toTensor();

        if (!self.is_contiguous()) self = self.contiguous();
        if (!other.is_contiguous()) other = other.contiguous();

        int m = self.size(dim_i0 - 2);
        int n = 1;
        int k = self.size(dim_i0 - 1);

        _Float16* A = (_Float16*)self.data_ptr();
        _Float16* x = (_Float16*)other.data_ptr();
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto output_shape = other.sizes().vec();

        if (dim_i0 > dim_i1) {
            output_shape = self.sizes().vec();
            output_shape[dim_i0 - 2] = m;
            output_shape.pop_back();
        } else {
            output_shape[dim_i1 - 1] = 1;
            output_shape[dim_i1 - 2] = m;
        }
        auto output = at::zeros(output_shape, options);
        _Float16* y = (_Float16*)output.data_ptr();

        PimDesc* pim_desc = PimCreateDesc(1, 1, m, k, PIM_FP16, OP_GEMV);
        PimBo* dev_op0 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT, x);
        PimBo* dev_op1 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT_T, A);
        PimBo* dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT, y);

        PimExecuteGemv(dev_out, dev_op0, dev_op1, nullptr);

        PimDestroyBo(dev_op0);
        PimDestroyBo(dev_op1);
        PimDestroyBo(dev_out);
        PimDestroyDesc(pim_desc);

        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (i0_is_vector == 1 && i1_is_vector == 1) {
        auto self = iv_self.toTensor();
        auto other = iv_other.toTensor();

        if (!self.is_contiguous()) self = self.contiguous();
        if (!other.is_contiguous()) other = other.contiguous();

        int m = 1;
        int n = 1;
        int k = self.size(dim_i0 - 1);

        _Float16* A = (_Float16*)self.data_ptr();
        _Float16* x = (_Float16*)other.data_ptr();
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto output_shape = other.sizes().vec();

        if (dim_i0 > dim_i1) {
            output_shape = self.sizes().vec();
            output_shape[dim_i0 - 1] = 1;
            output_shape[dim_i0 - 2] = m;
        } else {
            output_shape[dim_i1 - 1] = 1;
            output_shape[dim_i1 - 2] = m;
        }

        auto output = at::zeros(output_shape, options);
        _Float16* y = (_Float16*)output.data_ptr();

        PimDesc* pim_desc = PimCreateDesc(1, 1, k, m, PIM_FP16, OP_GEMV);
        PimBo* dev_op0 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_INPUT, x);
        PimBo* dev_op1 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_WEIGHT_T, A);
        PimBo* dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMV_OUTPUT, y);

        PimExecuteGemv(dev_out, dev_op0, dev_op1, nullptr);

        PimDestroyBo(dev_op0);
        PimDestroyBo(dev_op1);
        PimDestroyBo(dev_out);
        PimDestroyDesc(pim_desc);

        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        auto output = atenMatmul(iv_self.toTensor(), iv_other.toTensor());
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    }
}

void executorAtenMax(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Max node";

    auto max_layer = std::static_pointer_cast<nn_compiler::ir::AtenMaxLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto dim = max_layer->getDim();
    int keep_dim = max_layer->getKeepDim();

    if (in_stensor_id.size() == 1) {
        if (nn_compiler::ir::isDefaultValue(dim) && nn_compiler::ir::isDefaultValue(keep_dim)) {
            // aten::max(Tensor)
            auto output = atenMax(self_tensor);
            // update output
            stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        } else {
            // aten::max(Tensor, dim, keepdim)
            auto output = atenMax(self_tensor, dim, static_cast<bool>(keep_dim));
            // update output
            auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
            out_stensor_id.erase(pos, out_stensor_id.end());
            stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
            stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
        }
        return;
    }

    auto dtype = stream_executor.findBlob(in_stensor_id[in_id]).first;
    if (dtype == DataType::TENSOR) {
        // aten::max(Tensor, Tensor)
        torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = atenMax(self_tensor, other_tensor);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (isScalarType(dtype)) {
        // aten::max(Tensor, dim, keepdim)
        auto dim = max_layer->getDim();
        if (nn_compiler::ir::isDefaultValue(dim)) {
            auto dim_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
            assert(dim_iv.isInt());
            dim = dim_iv.toInt();
        }
        int keep_dim = max_layer->getKeepDim();
        if (nn_compiler::ir::isDefaultValue(keep_dim)) {
            auto keep_dim_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
            assert(keep_dim_iv.isBool());
            keep_dim = keep_dim_iv.toBool();
        }

        auto output = atenMax(self_tensor, dim, static_cast<bool>(keep_dim));
        auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
        out_stensor_id.erase(pos, out_stensor_id.end());
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
        stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));

    } else {
        DLOG(FATAL) << "Unsupported input type for aten::max";
    }
}

void executorAtenMaxPool2d(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten MaxPool2d node";

    auto max_pool_2d_layer = std::static_pointer_cast<nn_compiler::ir::AtenMaxPool2dLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    // In PyTorch, kernel_size is a tuple(int, int)
    auto kernel_size =  getDataShapeFromVector(max_pool_2d_layer->getKernelSize());
    std::vector<int64_t> kernel_size_vec;
    if (kernel_size.size() == 0) {
        // empty stride vector means kernel size = 0
        kernel_size_vec.push_back(0);
        kernel_size_vec.push_back(0);
    } else if (kernel_size[0] == INT64_MIN && kernel_size[1] == INT64_MIN) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isList());
        auto data_list = data_iv.toListRef();
        kernel_size_vec = parseIValueVector<int64_t>(data_list);
    } else {
        kernel_size_vec.push_back(kernel_size[0]);
        kernel_size_vec.push_back(kernel_size[1]);
    }

    // In PyTorch, stride is a tuple(int, int)
    auto stride = getDataShapeFromVector(max_pool_2d_layer->getStride());
    std::vector<int64_t> stride_vec;
    if (stride.size() == 0) {
        // empty stride vector means stride 0
        stride_vec.push_back(0);
        stride_vec.push_back(0);
    } else if (stride[0] == INT64_MIN && stride[1] == INT64_MIN) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isList());
        auto data_list = data_iv.toListRef();
        stride_vec = parseIValueVector<int64_t>(data_list);
    } else {
        stride_vec.push_back(stride[0]);
        stride_vec.push_back(stride[1]);
    }

    // In PyTorch, pad is a tuple(int, int)
    auto padding = getDataShapeFromVector(max_pool_2d_layer->getPad());
    std::vector<int64_t> padding_vec;
    if (padding.size() == 0) {
        // empty padding vector means padding 0
        padding_vec.push_back(0);
        padding_vec.push_back(0);
    } else if (padding[0] == INT64_MIN && padding[1] == INT64_MIN) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isList());
        auto data_list = data_iv.toListRef();
        padding_vec = parseIValueVector<int64_t>(data_list);
    } else {
        padding_vec.push_back(padding[0]);
        padding_vec.push_back(padding[1]);
    }

    // In PyTorch, dilation is a tuple(int, int)
    auto dilation = getDataShapeFromVector(max_pool_2d_layer->getDilation());
    std::vector<int64_t> dilation_vec;
    if (dilation.size() == 0) {
        // empty dilation vector means dilation 0
        dilation_vec.push_back(0);
        dilation_vec.push_back(0);
    } else if (dilation[0] == INT64_MIN && dilation[1] == INT64_MIN) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isList());
        auto data_list = data_iv.toListRef();
        dilation_vec = parseIValueVector<int64_t>(data_list);
    } else {
        dilation_vec.push_back(dilation[0]);
        dilation_vec.push_back(dilation[1]);
    }

    auto ceil_mode = max_pool_2d_layer->getCeilMode();
    if (nn_compiler::ir::isDefaultValue(ceil_mode)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        ceil_mode = data_iv.toInt();
    }

    auto output = atenMaxPool2d(self_tensor, at::ArrayRef<int64_t>(kernel_size_vec),
                                      at::ArrayRef<int64_t>(stride_vec), at::ArrayRef<int64_t>(padding_vec),
                                      at::ArrayRef<int64_t>(dilation_vec), static_cast<bool>(ceil_mode));
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenMin(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Min node";

    auto min_layer = std::static_pointer_cast<nn_compiler::ir::AtenMinLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto dim = min_layer->getDimOrY();
    if (in_stensor_id.size() == 1 && nn_compiler::ir::isDefaultValue(dim)) {
        auto output = atenMin(self_tensor);
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        return;
    }

    if (in_stensor_id.size() == 2 && nn_compiler::ir::isDefaultValue(dim)) {
        torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = atenMin(self_tensor, other_tensor);
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        return;
    }

    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto dim_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }

    auto keepdim = min_layer->getKeepDim();
    if (nn_compiler::ir::isDefaultValue(keepdim)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        keepdim = static_cast<int>(data_iv.toInt());
    }

    auto output = atenMin(self_tensor, dim, static_cast<bool>(keepdim));
    // update output
    auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
    out_stensor_id.erase(pos, out_stensor_id.end());
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
    stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
}

void executorAtenMul(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Mul node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;

    if (iv_self.isInt()) {
        assert(iv_other.isInt());
        auto self_int = iv_self.toInt();
        auto other_int = iv_other.toInt();
        auto output = atenMul(self_int, other_int);
        stream_executor.updateBlob(out_stensor_id[0], DataType::INT64, intToIValue(output));
    } else if (iv_self.isDouble()) {
        assert(iv_other.isDouble());
        auto self_double = iv_self.toDouble();
        auto other_double = iv_other.toDouble();
        auto output = atenMul(self_double, other_double);
        stream_executor.updateBlob(out_stensor_id[0], DataType::FLOAT64, doubleToIValue(output));
    } else if (iv_self.isTensor()) {
        at::Tensor self_tensor = iv_self.toTensor();
        if (iv_other.isTensor()) {
            at::Tensor other_tensor = iv_other.toTensor();
            auto output = atenMul(self_tensor, other_tensor);
            stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        } else if (iv_other.isScalar()) {
            at::Scalar other_scalar = iv_other.toScalar();
            auto output = atenMul(self_tensor, other_scalar);
            stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        } else {
            DLOG(FATAL) << "Unsupported input type for aten::mul";
        }
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::mul";
    }
}

void executorAtenNe(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Ne node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    assert(in_stensor_id.size() == 2);

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;

    if (iv_self.isIntList() && iv_other.isIntList()) {
        c10::List<int64_t> la = iv_self.toIntList();
        c10::List<int64_t> lb = iv_other.toIntList();
        auto output = atenNe(la, lb);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, boolToIValue(output));
    } else if (iv_self.isTensor()) {
        assert(iv_other.isTensor());
        at::Tensor self_tensor = iv_self.toTensor();
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = atenNe(self_tensor, other_tensor);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isScalar()) {
        assert(iv_other.isScalar());
        at::Scalar self_scalar = iv_self.toScalar();
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = atenNe(self_scalar, other_scalar);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, boolToIValue(output));
    } else if (iv_self.isString()) {
        assert(iv_other.isString());
        auto self_scalar = iv_self.toString()->string();
        auto other_scalar = iv_other.toString()->string();
        auto output = atenNe(self_scalar, other_scalar);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, boolToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::ne";
    }
}

void executorAtenNeg(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Neg node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();
    assert(in_stensor_id.size() == 1);

    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_id[0]).second;
    auto dtype = stream_executor.findBlob(in_stensor_id[0]).first;

    if (isScalarType(dtype)) {
        if (iv.isInt()) {
            int out = iv.toInt() * -1;
            stream_executor.updateBlob(out_stensor_id[0], DataType::INT64, scalarToIValue<int>(out));
        } else if (iv.isDouble()) {
            double out = iv.toDouble() * -1;
            stream_executor.updateBlob(out_stensor_id[0], DataType::FLOAT64, scalarToIValue<double>(out));
        }
    } else if (iv.isTensor()) {
        at::Tensor tensor = iv.toTensor();
        auto output = atenNeg(tensor);
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));

    } else {
        DLOG(FATAL) << "AtenNeg: unsupported dtype!";
    }
}

void executorAtenNorm(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Norm node";

    auto norm_layer = std::static_pointer_cast<nn_compiler::ir::AtenNormLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    auto input_ivalue = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(input_ivalue.isTensor());
    auto input_tensor = input_ivalue.toTensor();

    auto p = norm_layer->getP();
    if (nn_compiler::ir::isDefaultValue(p)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[1]).second;
        assert(data_iv.isInt());
        p = data_iv.toInt();
    }

    auto output = atenNorm(input_tensor, p);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenNot(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Not node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    auto iv = stream_executor.findBlob(in_stensor_id[0]).second;
    if (iv.isTensor()) {
        auto tensor = iv.toTensor();
        auto output = atenNot(tensor);
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (iv.isBool()) {
        auto input = iv.toBool();
        auto output = atenNot(input);
        stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, boolToIValue(output));
    } else {
        DLOG(FATAL) << "Aten not op's data type do not support!";
    }
}

void executorAtenOnes(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Ones node";

    auto ones_layer = std::static_pointer_cast<nn_compiler::ir::AtenOnesLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    auto iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isList());
    auto self_list = iv_self.toListRef();
    auto array_ref = parseIValueVector<int64_t>(self_list);

    at::TensorOptions options;
    auto dtype = ones_layer->getDType();
    if (nn_compiler::ir::isDefaultValue(dtype)) {
        auto iv_dtype = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_dtype.isNone()) {
            options = options.dtype(iv_dtype.toScalarType());
        }
    } else {
        options = options.dtype(at::ScalarType(dtype));
    }

    auto layout = ones_layer->getLayout();
    if (nn_compiler::ir::isDefaultValue(layout)) {
        auto iv_layout = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_layout.isNone()) {
            options = options.layout(iv_layout.toLayout());
        }
    } else {
        options = options.layout(at::Layout(layout));
    }

    auto device = ones_layer->getDevice();
    if (nn_compiler::ir::isDefaultValue(device)) {
        auto iv_device = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_device.isNone()) {
            options = options.device(iv_device.toDevice());
        }
    } else {
        options = options.device(device);
    }

    auto pin_memory = ones_layer->getPinMemory();
    if (nn_compiler::ir::isDefaultValue(pin_memory)) {
        auto iv_pin_memory = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!iv_pin_memory.isNone()) {
            options = options.pinned_memory(iv_pin_memory.toBool());
        }
    } else {
        options = options.pinned_memory(static_cast<bool>(pin_memory));
    }

    auto output = atenOnes(at::ArrayRef<int64_t>(array_ref), options);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenPackPaddedSequence(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten PackPaddedSequence node";

    auto pack_padded_sequence_layer = std::static_pointer_cast<nn_compiler::ir::AtenPackPaddedSequenceLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor() && iv_other.isTensor());
    auto self_tensor = iv_self.toTensor();
    auto other_tensor = iv_other.toTensor();

    auto batch_first = pack_padded_sequence_layer->getBatchFirst();
    if (nn_compiler::ir::isDefaultValue(batch_first)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[2]).second;
        assert(data_iv.isInt());
        batch_first = static_cast<int>(data_iv.toInt());
    }

    auto output = atenPackPaddedSequence(self_tensor, other_tensor, static_cast<bool>(batch_first));
    auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
    out_stensor_id.erase(pos, out_stensor_id.end());
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
    stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
}

void executorAtenPadPackedSequence(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten PadPackedSequence node";

    auto pad_packed_sequence_layer = std::static_pointer_cast<nn_compiler::ir::AtenPadPackedSequenceLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isTensor() && iv_other.isTensor());
    auto self_tensor = iv_self.toTensor();
    auto other_tensor = iv_other.toTensor();

    auto batch_first = pad_packed_sequence_layer->getBatchFirst();
    if (nn_compiler::ir::isDefaultValue(batch_first)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        batch_first = static_cast<int>(data_iv.toInt());
    }

    auto padding_value = pad_packed_sequence_layer->getPaddingValue();
    if (nn_compiler::ir::isDefaultValue(padding_value)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isDouble());
        padding_value = static_cast<float>(data_iv.toDouble());
    }

    auto total_length = pad_packed_sequence_layer->getTotalLength();
    if (nn_compiler::ir::isDefaultValue(total_length)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        total_length = data_iv.toInt();
    }

    auto output = atenPadPackedSequence(self_tensor, other_tensor, static_cast<bool>(batch_first), padding_value,
                                              total_length);
    auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
    out_stensor_id.erase(pos, out_stensor_id.end());
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
    stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
}

void executorAtenPow(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Pow node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;

    if (iv_self.isTensor() && iv_other.isTensor()) {
        auto self_tensor = iv_self.toTensor();
        auto other_tensor = iv_other.toTensor();
        auto output = atenPow(self_tensor, other_tensor);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isTensor() && iv_other.isScalar()) {
        auto self_tensor = iv_self.toTensor();
        auto other_scalar = iv_other.toScalar();
        auto output = atenPow(self_tensor, other_scalar);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isScalar() && iv_other.isTensor()) {
        auto self_scalar = iv_self.toScalar();
        auto other_tensor = iv_other.toTensor();
        auto output = atenPow(self_scalar, other_tensor);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::pow";
    }
}

void executorAtenRelu(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Relu node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    assert(in_stensor_id.size() == 1);

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    auto output = atenRelu(tensor);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenReshape(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Reshape node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    auto get_tensor = [&stream_executor](int id) {
        auto blob = stream_executor.findBlob(id);
        assert(blob.second.isTensor());
        return blob.second.toTensor();
    };
    at::Tensor input_tensor = get_tensor(in_stensor_id[0]);

    // Get shape
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
    // save outputs
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output_tensor));
}

void executorAtenSelect(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Select node";

    auto select_layer = std::static_pointer_cast<nn_compiler::ir::AtenSelectLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto dim = select_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto dim_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }
    auto index = select_layer->getIndex();
    if (nn_compiler::ir::isDefaultValue(index)) {
        auto index_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(index_iv.isInt());
        index = index_iv.toInt();
    }

    auto output = atenSelect(self_tensor, dim, index);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenSetItem(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten SetItem node";

    auto set_item_layer = std::static_pointer_cast<nn_compiler::ir::AtenSetItemLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isList());
    auto self_list = iv_self.toList();

    auto indice = set_item_layer->getIndices();
    if (nn_compiler::ir::isDefaultValue(indice)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        indice = data_iv.toInt();
    }

    torch::jit::IValue iv_item = stream_executor.findBlob(in_stensor_id[in_id++]).second;

    auto output = atenSetItem(self_list, indice, iv_item);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::LIST, listToIValue(output));
    stream_executor.updateBlob(in_stensor_id[0], DataType::LIST, listToIValue(output));
}

void executorAtenSize(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Size node";

    auto size_layer = std::static_pointer_cast<nn_compiler::ir::AtenSizeLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    // int inedges_cnt = size_layer->getInEdgeIds().size();
    auto dim = size_layer->getDim();
    if (in_stensor_id.size() == 1 && nn_compiler::ir::isDefaultValue(dim)) {
        auto output = atenSize(tensor);
        stream_executor.updateBlob(out_stensor_id[0], DataType::LIST, torch::jit::IValue(output));
    } else {
        if (nn_compiler::ir::isDefaultValue(dim)) {
            auto dim_iv = stream_executor.findBlob(in_stensor_id[1]).second;
            assert(dim_iv.isInt());
            dim = dim_iv.toInt();
        }
        int64_t output = atenSize(tensor, dim);
        stream_executor.updateBlob(out_stensor_id[0], DataType::INT64, intToIValue(output));
    }
}

void executorAtenSlice(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Slice node";

    auto slice_layer = std::static_pointer_cast<nn_compiler::ir::AtenSliceLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_tensor.isTensor());

    auto dim = slice_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto dim_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }
    c10::optional<int64_t> optional_start;
    auto start = slice_layer->getStart();
    if (nn_compiler::ir::isDefaultValue(start)) {
        auto start_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        optional_start = start_iv.toOptional<int64_t>();
    } else {
        optional_start = (torch::jit::IValue(start)).toOptional<int64_t>();
    }
    c10::optional<int64_t> optional_end;
    auto end = slice_layer->getEnd();
    if (nn_compiler::ir::isDefaultValue(end)) {
        auto end_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        optional_end = end_iv.toOptional<int64_t>();
    } else {
        optional_end = (torch::jit::IValue(end)).toOptional<int64_t>();
    }
    auto step = slice_layer->getStep();
    if (nn_compiler::ir::isDefaultValue(step)) {
        auto step_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(step_iv.isInt());
        step = step_iv.toInt();
    }

    auto output = atenSlice(iv_tensor.toTensor(), dim, optional_start, optional_end, step);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenSoftmax(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Softmax node";

    auto softmax_layer = std::static_pointer_cast<nn_compiler::ir::AtenSoftmaxLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_tensor.isTensor());
    auto self_tensor = iv_tensor.toTensor();

    auto dim = softmax_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        dim = data_iv.toInt();
    }

    auto dtype = softmax_layer->getDtype();
    at::Tensor output;
    if (nn_compiler::ir::isDefaultValue(dtype)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!data_iv.isNone()) {
            dtype = data_iv.toInt();
            output = atenSoftmax(self_tensor, dim, at::ScalarType(dtype));
        } else {
            output = atenSoftmax(self_tensor, dim);
        }
    }

    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenSqueeze(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Squeeze node";

    auto squeeze_layer = std::static_pointer_cast<nn_compiler::ir::AtenSqueezeLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_tensor.isTensor());
    auto self_tensor = iv_tensor.toTensor();

    auto dim = squeeze_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        dim = data_iv.toInt();
    }

    auto output = atenSqueeze(self_tensor, dim);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenSub(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Sub node";

    auto sub_layer = std::static_pointer_cast<nn_compiler::ir::AtenSubLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int64_t alpha = sub_layer->getAlpha();
    if (nn_compiler::ir::isDefaultValue(alpha)) {
        auto alpha_iv = stream_executor.findBlob(in_stensor_id[2]).second;
        assert(alpha_iv.isInt());
        alpha = alpha_iv.toInt();
    }

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    auto dtype = stream_executor.findBlob(in_stensor_id[1]).first;
    if (dtype == DataType::TENSOR) {
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = atenSub(self_tensor, other_tensor, alpha);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (isScalarType(dtype)) {
        assert(iv_other.isScalar());
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = atenSub(self_tensor, other_scalar, alpha);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::sub";
    }
}

void executorAtenSum(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Sum node";

    auto sum_layer = std::static_pointer_cast<nn_compiler::ir::AtenSumLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_tensor.isTensor());
    auto self_tensor = iv_tensor.toTensor();

    auto dims = sum_layer->getDim();
    if (dims.size() == 0) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isList());
        auto dims_list = data_iv.toListRef();
        dims = parseIValueVector<int64_t>(dims_list);
    }

    auto keepdim = sum_layer->getKeepdim();
    if (nn_compiler::ir::isDefaultValue(keepdim)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        keepdim = static_cast<int>(data_iv.toInt());
    }

    auto dtype = sum_layer->getDtype();
    if (nn_compiler::ir::isDefaultValue(dtype)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (!data_iv.isNone()) {
            dtype = data_iv.toInt();
        }
    }

    at::Tensor output;
    if (nn_compiler::ir::isDefaultValue(dtype)) {
        output = atenSum(self_tensor, at::ArrayRef<int64_t>(dims), static_cast<bool>(keepdim), c10::nullopt);
    } else {
        output =
            atenSum(self_tensor, at::ArrayRef<int64_t>(dims), static_cast<bool>(keepdim), at::ScalarType(dtype));
    }
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenTanh(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Tanh node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    auto self_tensor = iv_tensor.toTensor();

    auto output = atenTanh(self_tensor);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenTensor(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Tensor node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;

    at::TensorOptions options;
    auto iv_dtype = stream_executor.findBlob(in_stensor_id[1]).second;
    auto iv_device = stream_executor.findBlob(in_stensor_id[2]).second;
    auto iv_pin_memory = stream_executor.findBlob(in_stensor_id[3]).second;

    if (!iv_dtype.isNone()) {
        options = options.dtype(iv_dtype.toScalarType());
    }
    if (!iv_device.isNone()) {
        options = options.device(iv_device.toDevice());
    }
    if (!iv_pin_memory.isNone()) {
        if (iv_pin_memory.isInt()) {
            options = options.pinned_memory(iv_pin_memory.toInt());
        } else if (iv_pin_memory.isBool()) {
            options = options.pinned_memory(iv_pin_memory.toBool());
        } else {
            DLOG(FATAL) << "Unsupported data type to parse iv_pin_memory.";
        }
    }
    // FIXME(SRCX): To get list item type, is there a better way?
    torch::jit::IValue value_item;
    if (iv_self.isList()) {
        while (iv_self.isList()) {
            if (!iv_self.toListRef()[0].isList()) {
                value_item = iv_self.toListRef()[0];
            }
            iv_self = iv_self.toListRef()[0];
        }
    } else if (iv_self.isScalar()) {
        value_item = iv_self;
    } else {
        DLOG(FATAL) << "Unsupported data type to IValue.";
    }

    at::Tensor output;
    if (value_item.isInt()) {
        std::vector<int64_t> value_vec;
        std::vector<int64_t> dim = {1};
        parseIValueList<int64_t>(stream_executor.findBlob(in_stensor_id[0]).second, value_vec, dim, 1);
        output = atenTensor(at::ArrayRef<int64_t>(value_vec), options).reshape(at::ArrayRef<int64_t>(dim));
    } else if (value_item.isDouble()) {
        std::vector<double> value_vec;
        std::vector<int64_t> dim = {1};
        parseIValueList<double>(stream_executor.findBlob(in_stensor_id[0]).second, value_vec, dim, 1);
        output = atenTensor(at::ArrayRef<double>(value_vec), options).reshape(at::ArrayRef<int64_t>(dim));
    } else {
        DLOG(FATAL) << "Unsupported data type to parse IValue list.";
    }
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenTo1(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten To1 node";

    auto to1_layer = std::static_pointer_cast<nn_compiler::ir::AtenTo1Layer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto ori_dtype = to1_layer->getDType();
    if (nn_compiler::ir::isDefaultValue(ori_dtype)) {
        auto ori_dtype_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(ori_dtype_iv.isInt());
        ori_dtype = ori_dtype_iv.toInt();
    }
    auto dtype = at::ScalarType(ori_dtype);

    int non_blocking_val = to1_layer->getNonBlocking();
    if (nn_compiler::ir::isDefaultValue(non_blocking_val)) {
        auto non_blocking_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (non_blocking_iv.isNone()) {
            non_blocking_val = 0;
        } else {
            non_blocking_val = non_blocking_iv.toInt();
        }
    }
    bool non_blocking = static_cast<bool>(non_blocking_val);

    int copy_val = to1_layer->getCopy();
    if (nn_compiler::ir::isDefaultValue(copy_val)) {
        auto copy_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (copy_iv.isNone()) {
            copy_val = 0;
        } else {
            copy_val = copy_iv.toInt();
        }
    }
    bool copy = static_cast<bool>(copy_val);

    auto optional_memory_format = to1_layer->getOptionalMemoryFormat();
    if (nn_compiler::ir::isDefaultValue(optional_memory_format)) {
        auto optional_memory_format_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(optional_memory_format_iv.isInt());
        optional_memory_format = optional_memory_format_iv.toInt();
    }

    if (optional_memory_format == -1) {  // optional_memory_format = NONE
        auto output = atenTo(self_tensor, dtype, non_blocking, copy);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        auto memory_format = getMemoryFormat(optional_memory_format);
        auto output = atenTo(self_tensor, dtype, non_blocking, copy, memory_format);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    }
}

void executorAtenTo2(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten To2 node";

    auto to2_layer = std::static_pointer_cast<nn_compiler::ir::AtenTo2Layer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_other.isTensor());
    at::Tensor other_tensor = iv_other.toTensor();

    int non_blocking_val = to2_layer->getNonBlocking();
    if (nn_compiler::ir::isDefaultValue(non_blocking_val)) {
        auto non_blocking_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (non_blocking_iv.isNone()) {
            non_blocking_val = 0;
        } else {
            non_blocking_val = non_blocking_iv.toInt();
        }
    }
    bool non_blocking = static_cast<bool>(non_blocking_val);

    int copy_val = to2_layer->getCopy();
    if (nn_compiler::ir::isDefaultValue(copy_val)) {
        auto copy_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        if (copy_iv.isNone()) {
            copy_val = 0;
        } else {
            copy_val = copy_iv.toInt();
        }
    }
    bool copy = static_cast<bool>(copy_val);

    auto optional_memory_format = to2_layer->getOptionalMemoryFormat();
    if (nn_compiler::ir::isDefaultValue(optional_memory_format)) {
        auto optional_memory_format_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(optional_memory_format_iv.isInt());
        optional_memory_format = optional_memory_format_iv.toInt();
    }

    if (optional_memory_format == -1) {  // optional_memory_format = NONE
        auto output = atenTo(self_tensor, other_tensor, non_blocking, copy);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        auto memory_format = getMemoryFormat(optional_memory_format);
        auto output = atenTo(self_tensor, other_tensor, non_blocking, copy, memory_format);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    }
}

void executorAtenTopk(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Topk node";

    auto topk_layer = std::static_pointer_cast<nn_compiler::ir::AtenTopkLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_tensor.isTensor());
    auto self_tensor = iv_tensor.toTensor();

    auto k = topk_layer->getK();
    if (nn_compiler::ir::isDefaultValue(k)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        k = data_iv.toInt();
    }

    auto dim = topk_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        dim = data_iv.toInt();
    }

    auto largest = topk_layer->getLargest();
    if (nn_compiler::ir::isDefaultValue(largest)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        largest = static_cast<int>(data_iv.toInt());
    }

    auto sorted = topk_layer->getSorted();
    if (nn_compiler::ir::isDefaultValue(sorted)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        sorted = static_cast<int>(data_iv.toInt());
    }

    auto output = atenTopk(self_tensor, k, dim, static_cast<bool>(largest), static_cast<bool>(sorted));
    auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
    out_stensor_id.erase(pos, out_stensor_id.end());
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
    stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
}

void executorAtenTranspose(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Transpose node";

    auto transpose_layer = std::static_pointer_cast<nn_compiler::ir::AtenTransposeLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto dim0 = transpose_layer->getDim0();
    if (nn_compiler::ir::isDefaultValue(dim0)) {
        auto dim0_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(dim0_iv.isInt());
        dim0 = dim0_iv.toInt();
    }
    auto dim1 = transpose_layer->getDim1();
    if (nn_compiler::ir::isDefaultValue(dim1)) {
        auto dim1_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(dim1_iv.isInt());
        dim1 = dim1_iv.toInt();
    }

    auto output = atenTranspose(self_tensor, dim0, dim1);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenUnsqueeze(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Unsqueeze node";

    auto unsqueeze_layer = std::static_pointer_cast<nn_compiler::ir::AtenUnsqueezeLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    auto dim = unsqueeze_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto dim_iv = stream_executor.findBlob(in_stensor_id[1]).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }
    auto is_inplace = unsqueeze_layer->getIsInplace();
    at::Tensor output = atenUnsqueeze(tensor, dim);
    // If Unsqueeze op is in-place op, it need change origin data
    if (is_inplace) {
        auto releation_blob_id = stream_executor.releation_blob_ids_map_.find(in_stensor_id[0]);
        assert(releation_blob_id != stream_executor.releation_blob_ids_map_.end());
        auto list_blob_id = releation_blob_id->second.first;
        auto in_list_pos = releation_blob_id->second.second;

        auto list_blob_iv = stream_executor.findBlob(list_blob_id).second;
        std::vector<torch::IValue> inputs;
        auto datas = list_blob_iv.toListRef();
        for (uint32_t idx = 0; idx < datas.size(); idx++) {
            if (idx == in_list_pos) {
                inputs.push_back(tensorToIValue(output));
            } else {
                inputs.push_back(datas[idx]);
            }
        }
        at::ListTypePtr type = inferTypeFromDataType(inferDataType(output));
        primListConstruct(inputs, inputs.size(), type);
        stream_executor.updateBlob(list_blob_id, DataType::LIST, inputs.at(0));
    } else {
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    }
}

void executorAtenView(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten View node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    torch::jit::IValue iv_size = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_size.isList());
    auto size_list = iv_size.toListRef();
    auto size_array = parseIValueVector<int64_t>(size_list);

    at::Tensor output = atenView(tensor, at::ArrayRef<int64_t>(size_array));
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenWarn(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute AtenWarn node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_id[0]).second;

    atenWarn(iv.toString()->string());
}

void executorAtenZeros(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Zeros node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isList());

    auto self_list = iv_self.toListRef();
    // input list -> at::IntArrayRef size, so datatype of elements in list must be int.
    auto array_ref = parseIValueVector<int64_t>(self_list);

    at::TensorOptions options;
    auto iv_dtype = stream_executor.findBlob(in_stensor_id[1]).second;
    auto iv_layout = stream_executor.findBlob(in_stensor_id[2]).second;
    auto iv_device = stream_executor.findBlob(in_stensor_id[3]).second;
    auto iv_pin_memory = stream_executor.findBlob(in_stensor_id[4]).second;

    if (!iv_dtype.isNone()) {
        options = options.dtype(iv_dtype.toScalarType());
    }
    if (!iv_layout.isNone()) {
        options = options.layout(iv_layout.toLayout());
    }
    if (iv_device.isDevice()) {
        options = options.device(iv_device.toDevice());
    } else if (iv_device.isString()) {
        options = options.device(iv_device.toStringRef());
    }
    if (!iv_pin_memory.isNone()) {
        options = options.pinned_memory(iv_pin_memory.toBool());
    }

    auto output = atenZeros(at::ArrayRef<int64_t>(array_ref), options);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executorAtenZerosLike(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten ZerosLike node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    assert(in_stensor_id.size() == 6);

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    at::Tensor tensor = iv_tensor.toTensor();

    at::TensorOptions options;
    auto iv_dtype = stream_executor.findBlob(in_stensor_id[1]).second;
    auto iv_layout = stream_executor.findBlob(in_stensor_id[2]).second;
    auto iv_device = stream_executor.findBlob(in_stensor_id[3]).second;
    auto iv_pin_memory = stream_executor.findBlob(in_stensor_id[4]).second;

    if (!iv_dtype.isNone()) {
        options = options.dtype(iv_dtype.toScalarType());
    }

    if (!iv_layout.isNone()) {
        options = options.layout(iv_layout.toLayout());
    }
    if (iv_device.isDevice()) {
        options = options.device(iv_device.toDevice());
    } else if (iv_device.isString()) {
        options = options.device(iv_device.toStringRef());
    }
    if (!iv_pin_memory.isNone()) {
        options = options.pinned_memory(iv_pin_memory.toBool());
    }

    auto iv_memory_format = stream_executor.findBlob(in_stensor_id[5]).second;
    at::Tensor output;
    if (iv_memory_format.isNone()) {
        output = atenZeroslike(tensor, options);
    } else {
        output = atenZeroslike(tensor, options, iv_memory_format.toMemoryFormat());
    }

    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

}  // namespace runtime
}  // namespace nn_compiler