#include <algorithm>
#include <vector>

#include "c10/hip/HIPFunctions.h"
// #include "common/include/cast.hpp"
#include "new_runtime/include/executor/aten_ops_executor.h"
#include "new_runtime/include/executor/aten_ops.h"
#include "new_runtime/include/executor/custom_ops.hpp"
#include "new_runtime/include/executor/stream_executor.h"
#include "new_runtime/include/executor/utils.h"
#include "glog/logging.h"
#include "new_ir/include/nn_model.h"
#include "new_ir/include/nn_network.h"
#include "new_ir/include/types.h"
#include "new_ir/include/common/utils.hpp"
#include "new_ir/include/layers/all_layers.h"
#include "pim_runtime_api.h"
#include "tv_tools.h"

namespace nn_compiler {
namespace runtime {

void executorAtenAdd(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Add node";

    auto add_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenAddLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int64_t alpha = add_layer->getAlpha();
    // if (nn_compiler::ir::isDefaultValue(alpha)) {
    //     if (in_stensor_id.size() == 3) {
    //         // if "prim::if" layer linked to current, this edge has no practical meaning
    //         if (stream_executor.findBlob(in_stensor_id[2]).second != stream_executor.global_blobs_.end()) {
    //             auto alpha_iv = stream_executor.findBlob(in_stensor_id[2]).second;
    //             assert(alpha_iv.isInt());
    //             alpha = alpha_iv.toInt();
    //         } else {
    //             alpha = 1;
    //         }
    //     } else {
    //         alpha = 1;
    //     }
    // }

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
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
        auto output = atenAdd(self_tensor, other_tensor, alpha);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
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

    auto addmm_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenAddmmLayer>(layer);
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

    auto arange1_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenArange1Layer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    auto end = arange1_layer->getEnd();
    if (nn_compiler::ir::isDefaultValue(end)) {
        auto iv_end = stream_executor.findBlob(in_stensor_id[0]).second;
        end = iv_end.toInt();
    }

    at::TensorOptions options;
    auto dtype = arange1_layer->getDtype();
    if (nn_compiler::ir::isDefaultValue(dtype)) {
        auto iv_dtype = stream_executor.findBlob(in_stensor_id[1]).second;
        if (!iv_dtype.isNone()) {
            options = options.dtype(iv_dtype.toScalarType());
        }
    } else {
        options = options.dtype(at::ScalarType(dtype));
    }

    auto layout = arange1_layer->getLayout();
    if (nn_compiler::ir::isDefaultValue(layout)) {
        auto iv_layout = stream_executor.findBlob(in_stensor_id[2]).second;
        if (!iv_layout.isNone()) {
            options = options.layout(iv_layout.toLayout());
        }
    } else {
        options = options.layout(at::Layout(layout));
    }

    auto device = arange1_layer->getDevice();
    if (nn_compiler::ir::isDefaultValue(device)) {
        auto iv_device = stream_executor.findBlob(in_stensor_id[3]).second;
        if (!iv_device.isNone()) {
            options = options.device(iv_device.toDevice());
        }
    } else {
        options = options.device(device);
    }

    auto pin_memory = arange1_layer->getPinMemory();
    if (nn_compiler::ir::isDefaultValue(pin_memory)) {
        auto iv_pin_memory = stream_executor.findBlob(in_stensor_id[4]).second;
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

    auto arange2_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenArange2Layer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    auto start = arange2_layer->getStart();
    if (nn_compiler::ir::isDefaultValue(start)) {
        auto iv_start = stream_executor.findBlob(in_stensor_id[0]).second;
        start = iv_start.toInt();
    }

    auto end = arange2_layer->getEnd();
    if (nn_compiler::ir::isDefaultValue(end)) {
        auto iv_end = stream_executor.findBlob(in_stensor_id[1]).second;
        end = iv_end.toInt();
    }

    at::TensorOptions options;
    auto dtype = arange2_layer->getDtype();
    if (nn_compiler::ir::isDefaultValue(dtype)) {
        auto iv_dtype = stream_executor.findBlob(in_stensor_id[2]).second;
        if (!iv_dtype.isNone()) {
            options = options.dtype(iv_dtype.toScalarType());
        }
    } else {
        options = options.dtype(at::ScalarType(dtype));
    }

    auto layout = arange2_layer->getLayout();
    if (nn_compiler::ir::isDefaultValue(layout)) {
        auto iv_layout = stream_executor.findBlob(in_stensor_id[3]).second;
        if (!iv_layout.isNone()) {
            options = options.layout(iv_layout.toLayout());
        }
    } else {
        options = options.layout(at::Layout(layout));
    }

    auto device = arange2_layer->getDevice();
    if (nn_compiler::ir::isDefaultValue(device)) {
        auto iv_device = stream_executor.findBlob(in_stensor_id[4]).second;
        if (!iv_device.isNone()) {
            options = options.device(iv_device.toDevice());
        }
    } else {
        options = options.device(device);
    }

    auto pin_memory = arange2_layer->getPinMemory();
    if (nn_compiler::ir::isDefaultValue(pin_memory)) {
        auto iv_pin_memory = stream_executor.findBlob(in_stensor_id[5]).second;
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

    auto arange3_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenArange3Layer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    auto start = arange3_layer->getStart();
    if (nn_compiler::ir::isDefaultValue(start)) {
        auto iv_start = stream_executor.findBlob(in_stensor_id[0]).second;
        start = iv_start.toInt();
    }

    auto end = arange3_layer->getEnd();
    if (nn_compiler::ir::isDefaultValue(end)) {
        auto iv_end = stream_executor.findBlob(in_stensor_id[1]).second;
        end = iv_end.toInt();
    }

    auto step = arange3_layer->getStep();
    if (nn_compiler::ir::isDefaultValue(step)) {
        auto iv_step = stream_executor.findBlob(in_stensor_id[2]).second;
        step = iv_step.toInt();
    }

    at::TensorOptions options;
    auto dtype = arange3_layer->getDtype();
    if (nn_compiler::ir::isDefaultValue(dtype)) {
        auto iv_dtype = stream_executor.findBlob(in_stensor_id[3]).second;
        if (!iv_dtype.isNone()) {
            options = options.dtype(iv_dtype.toScalarType());
        }
    } else {
        options = options.dtype(at::ScalarType(dtype));
    }

    auto layout = arange3_layer->getLayout();
    if (nn_compiler::ir::isDefaultValue(layout)) {
        auto iv_layout = stream_executor.findBlob(in_stensor_id[4]).second;
        if (!iv_layout.isNone()) {
            options = options.layout(iv_layout.toLayout());
        }
    } else {
        options = options.layout(at::Layout(layout));
    }

    auto device = arange3_layer->getDevice();
    if (nn_compiler::ir::isDefaultValue(device)) {
        auto iv_device = stream_executor.findBlob(in_stensor_id[5]).second;
        if (!iv_device.isNone()) {
            options = options.device(iv_device.toDevice());
        }
    } else {
        options = options.device(device);
    }

    auto pin_memory = arange3_layer->getPinMemory();
    if (nn_compiler::ir::isDefaultValue(pin_memory)) {
        auto iv_pin_memory = stream_executor.findBlob(in_stensor_id[6]).second;
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

    auto as_tensor_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenAsTensorLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    auto in_tensor = iv_tensor.toTensor();

    auto int_dtype = as_tensor_layer->getDtype();
    if (nn_compiler::ir::isDefaultValue(int_dtype)) {
        auto iv = stream_executor.findBlob(in_stensor_id[1]).second;
        int_dtype = iv.toInt();
    }
    auto dtype = at::ScalarType(int_dtype);

    auto str_device = as_tensor_layer->getDevice();
    if (nn_compiler::ir::isDefaultValue(str_device)) {
        auto map_value = stream_executor.findBlob(in_stensor_id[2]);
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

    auto self_tensor = iv_self.toTensor();
    auto other_tensor = iv_other.toTensor();
    auto output = atenBmm(self_tensor, other_tensor);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, torch::jit::IValue(output));
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
    
    std::vector<at::Tensor> tensor_vec;

    // auto& input_list_edge = cast<nncir::DataEdge>(cat_node.getInEdge(0));
    // auto input_blob_id = input_list_edge.getBlobId();

    // if (stream_executor.modelType == "GNMT" && op_node.getFirstInEdge().getInNode()->getNumInputs() == 0) {
    //     int cat_mem_id = cat_node.getMemBlobId();
    //     auto it = stream_executor.global_blobs_.find(cat_mem_id);
    //     auto& out_edge = cast<nncir::DataEdge>(cat_node.getFirstOutEdge());
    //     auto output = it->second.second.toTensor().clone();
    //     stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    //     return;
    // }

    // int cat_mem_id = cat_node.getMemBlobId();
    // auto it = stream_executor.global_blobs_.find(cat_mem_id);
    // if (stream_executor.modelType == "GNMT" && it != stream_executor.global_blobs_.end()) {
    //     auto& out_edge = cast<nncir::DataEdge>(cat_node.getFirstOutEdge());
    //     auto output = it->second.second;
    //     stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, output);
    //     return;
    // }

    auto ivalue = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(ivalue.isTensorList());

    auto c10_tensor_list = ivalue.toTensorList();
    for (auto tensor : c10_tensor_list) {
        tensor_vec.push_back(tensor);
    }
    at::TensorList tensor_list(tensor_vec);

    auto dim = cat_node.getDim();
    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto dim_iv = stream_executor.findBlob(in_stensor_id[1]).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }

    auto output = atenCat(tensor_list, dim);
    // stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorListToIValue(output));
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

    auto chunk_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenChunkLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    auto in_tensor = iv_tensor.toTensor();

    auto chunks = chunk_layer->getChunks();
    if (nn_compiler::ir::isDefaultValue(chunks)) {
        auto iv = stream_executor.findBlob(in_stensor_id[1]).second;
        chunks = iv.toInt();
    }

    auto dim = chunk_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(chunks)) {
        auto iv = stream_executor.findBlob(in_stensor_id[2]).second;
        dim = iv.toInt();
    }

    auto output = atenChunk(in_tensor, chunks, dim);
    stream_executor.updateBlob(out_stensor_id[0], DataType::LIST, vectorToIValue(output));
}

void executorAtenClamp(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Clamp node";

    auto clamp_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenClampLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto min = clamp_layer->getMin();
    auto max = clamp_layer->getMax();
    if (nn_compiler::ir::isDefaultValue(min)) {
        auto iv = stream_executor.findBlob(in_stensor_id[1]).second;
        min = static_cast<int>(iv.toInt());
    }
    if (nn_compiler::ir::isDefaultValue(max)) {
        auto iv = stream_executor.findBlob(in_stensor_id[2]).second;
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
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isList());
    at::List self_list = iv_self.toList();
    atenClear(self_list);
    // update list
    assert(in_stensor_id[0] == out_stensor_id[0]);
    stream_executor.updateBlob(in_stensor_id[0], DataType::LIST, torch::jit::IValue(self_list));
}

}  // namespace runtime
}  // namespace nn_compiler
