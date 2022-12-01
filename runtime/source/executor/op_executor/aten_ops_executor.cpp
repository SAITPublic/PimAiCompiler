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

#include "executor/op_executor/aten_ops_executor.h"
#include "executor/op_executor/custom_ops.h"
#include "executor/op_executor/prim_ops_executor.h"
#include "executor/stream_executor.h"
#include "utils/utils.h"

using namespace nn_compiler::runtime::utils;

namespace nn_compiler
{
namespace runtime
{
namespace op_executor
{
void executeAtenAbs(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Abs node";

    int in_id = 0;
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[in_id]).second;
    auto self_tensor = iv_self.toTensor();
    auto output = atenAbs(self_tensor);

    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenAdd(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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
    if (iv_other.isTensor()) {
        at::Tensor other_tensor = iv_other.toTensor();
        auto self_sizes = self_tensor.sizes();
        auto other_sizes = other_tensor.sizes();

        auto customAddDimensionsCheck = [=](const c10::IntArrayRef& input1_size, const c10::IntArrayRef& input2_size) {
            if (input1_size.size() == 1 || input1_size.size() == 2) {
                // there is no batch and channel dimension for tensors, default 1 batch and 1 channel.
                return true;
            } else if (input1_size.size() == 3) {  // tensor with channel, height, width
                if (input1_size[0] != 1 || (input2_size.size() == 3 && input2_size[0] != 1)) {
                    // tensor channel != 1
                    return false;
                } else {
                    return true;
                }
            } else if (input1_size.size() == 4) {  // tensor with batch, channel, height, width
                if ((input2_size.size() == 4 && input1_size[0] != input2_size[0]) || (input1_size[1] != 1) ||
                    (input2_size.size() == 4 && input2_size[1] != 1) ||
                    (input2_size.size() == 3 && input2_size[0] != 1)) {
                    // different batch of tensors, or tensor channel != 1
                    return false;
                } else {
                    return true;
                }
            } else {
                return false;
            }
        };

        // custom_large_add only works for:
        // 1) float16 data type, and 2) same batch size, and 3) channel == 1
        if (self_tensor.dtype() == c10::ScalarType::Half &&
            (self_sizes.size() >= other_sizes.size() ? customAddDimensionsCheck(self_sizes, other_sizes)
                                                     : customAddDimensionsCheck(other_sizes, self_sizes))) {
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
            auto pre_layers = layer->getPreLayerIDs();
            auto input1_layer = stream_executor.getGraph()->getLayerByPosition(pre_layers[0]);
            auto input2_layer = stream_executor.getGraph()->getLayerByPosition(pre_layers[1]);
            bool add_opt_flag =
                input1_layer->getType() == nn_compiler::ir::LayerType::ATENLSTM1 &&
                input2_layer->getType() == nn_compiler::ir::LayerType::ATENLSTM1 &&
                std::static_pointer_cast<nn_compiler::ir::AtenLSTM1Layer>(input1_layer)->getCustomOptNumber() == 2 &&
                std::static_pointer_cast<nn_compiler::ir::AtenLSTM1Layer>(input2_layer)->getCustomOptNumber() == 1;
            if (stream_executor.getModelType() == "GNMT" && add_opt_flag) {
                auto out1_layer = stream_executor.getGraph()->getLayerByPosition((layer->getNextLayerIDs())[0]);
                auto out1_out1_layer =
                    stream_executor.getGraph()->getLayerByPosition((out1_layer->getNextLayerIDs())[0]);
                int cat_mem_id =
                    std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(out1_out1_layer)->getMemLayerId();
                auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
                tmp = torch::from_blob((_Float16*)(stream_executor.findBlob(cat_mem_id).second.toTensor().data_ptr()),
                                       {1, 1, 1024}, options);
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
            if (stream_executor.getModelType() == "GNMT" && add_opt_flag) {
                stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(tmp));
            } else {
                stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(self_tensor));
            }
        } else {
            auto output = atenAdd(self_tensor, other_tensor, alpha);
            stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        }
    } else if (iv_other.isScalar()) {
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = atenAdd(self_tensor, other_scalar, alpha);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::add";
    }
}  // executeAtenAdd

void executeAtenAddmm(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Addmm node";

    auto addmm_layer = std::static_pointer_cast<nn_compiler::ir::AtenAddmmLayer>(layer);
    auto act_type = addmm_layer->get_act_type();

    // TODO(SRCX): choose the corresponding kernel when activation type is aten::none, aten::relu, aten::max
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_mat1 = stream_executor.findBlob(in_stensor_id[1]).second;
    torch::jit::IValue iv_mat2 = stream_executor.findBlob(in_stensor_id[2]).second;
    assert(iv_mat1.isTensor() && iv_mat2.isTensor());
    auto mat1_tensor = iv_mat1.toTensor();
    auto mat2_tensor = iv_mat2.toTensor();

    int alpha = 1, beta = 1;
    if (in_stensor_id.size() == 5) {
        torch::jit::IValue iv_beta = stream_executor.findBlob(in_stensor_id[3]).second;
        torch::jit::IValue iv_alpha = stream_executor.findBlob(in_stensor_id[4]).second;
        beta = iv_beta.toInt();
        alpha = iv_alpha.toInt();
    }

    torch::jit::IValue output;
    if (iv_self.isNone()) {
        customAtenMatmul(mat1_tensor, mat2_tensor, output);
    } else {
        auto self_tensor = iv_self.toTensor();
        customAtenAddmm(act_type, self_tensor, mat1_tensor, mat2_tensor, beta, alpha, output);
    }
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, output);
}

void executeAtenAddmmWithStream(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor,
                                void* stream)
{
    DLOG(INFO) << "execute Aten Addmm node";

    auto addmm_layer = std::static_pointer_cast<nn_compiler::ir::AtenAddmmLayer>(layer);
    auto act_type = addmm_layer->get_act_type();

    // TODO(SRCX): choose the corresponding kernel when activation type is aten::none, aten::relu, aten::max
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_mat1 = stream_executor.findBlob(in_stensor_id[1]).second;
    torch::jit::IValue iv_mat2 = stream_executor.findBlob(in_stensor_id[2]).second;
    assert(iv_mat1.isTensor() && iv_mat2.isTensor());
    auto mat1_tensor = iv_mat1.toTensor();
    auto mat2_tensor = iv_mat2.toTensor();

    int alpha = 1, beta = 1;
    if (in_stensor_id.size() == 5) {
        torch::jit::IValue iv_beta = stream_executor.findBlob(in_stensor_id[3]).second;
        torch::jit::IValue iv_alpha = stream_executor.findBlob(in_stensor_id[4]).second;
        beta = iv_beta.toInt();
        alpha = iv_alpha.toInt();
    }

    torch::jit::IValue output;
    if (iv_self.isNone()) {
        customAtenMatmul(mat1_tensor, mat2_tensor, output, stream);
    } else {
        auto self_tensor = iv_self.toTensor();
        customAtenAddmm(act_type, self_tensor, mat1_tensor, mat2_tensor, beta, alpha, output, stream);
    }
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, output);
}

void executeAtenAnd(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenAny(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenAppend(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenArange1(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenArange2(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenArange3(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenArgmax(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten argmax node";
    auto argmax_layer = std::static_pointer_cast<nn_compiler::ir::AtenArgmaxLayer>(layer);

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    assert(iv_tensor.isTensor());
    auto in_tensor = iv_tensor.toTensor();

    auto dim = argmax_layer->getDim();
    if (nn_compiler::ir::isDefaultValue(dim)) {
        auto iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        dim = iv.toInt();
    }

    auto keepdim = argmax_layer->getKeepDim();
    if (nn_compiler::ir::isDefaultValue(keepdim)) {
        auto iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        keepdim = iv.toInt();
    }

    auto output = atenArgmax(in_tensor, dim, static_cast<bool>(keepdim));
    stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenAsTensor(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenBatchNorm2d(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenBitwiseNot(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenBmm(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Bmm node";
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor() && iv_other.isTensor());

    at::Tensor tmp0, tmp1, tmp2;
    if (stream_executor.getModelType() == "GNMT") {
        auto out2_layer = stream_executor.getGraph()->getLayerByPosition((layer->getNextLayerIDs())[1]);
        auto out2_out1_layer = stream_executor.getGraph()->getLayerByPosition((out2_layer->getNextLayerIDs())[0]);
        auto out2_out2_layer = stream_executor.getGraph()->getLayerByPosition((out2_layer->getNextLayerIDs())[1]);
        auto out2_out3_layer = stream_executor.getGraph()->getLayerByPosition((out2_layer->getNextLayerIDs())[2]);
        auto out2_out1_out1_layer =
            stream_executor.getGraph()->getLayerByPosition((out2_out1_layer->getNextLayerIDs())[0]);
        auto out2_out2_out1_layer =
            stream_executor.getGraph()->getLayerByPosition((out2_out2_layer->getNextLayerIDs())[0]);
        auto out2_out3_out1_layer =
            stream_executor.getGraph()->getLayerByPosition((out2_out3_layer->getNextLayerIDs())[0]);

        int64_t cat_mem_id =
            std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(out2_out1_out1_layer)->getMemLayerId();
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        tmp0 = torch::from_blob((_Float16*)(stream_executor.findBlob(cat_mem_id).second.toTensor().data_ptr()) + 1024,
                                {1, 1, 1024}, options);

        cat_mem_id = std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(out2_out2_out1_layer)->getMemLayerId();
        tmp1 = torch::from_blob((_Float16*)(stream_executor.findBlob(cat_mem_id).second.toTensor().data_ptr()) + 1024,
                                {1, 1, 1024}, options);

        cat_mem_id = std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(out2_out3_out1_layer)->getMemLayerId();
        tmp2 = torch::from_blob((_Float16*)(stream_executor.findBlob(cat_mem_id).second.toTensor().data_ptr()) + 1024,
                                {1, 1, 1024}, options);
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

    if (stream_executor.getModelType() == "GNMT") {
        y = (_Float16*)tmp0.data_ptr();
    }

    rocblas_bmm_template_xAy(nullptr, x, A, y, m, n, k);
    if (stream_executor.getModelType() == "GNMT") {
        atenCopy_(tmp1, tmp0, c10::attr::non_blocking);
        atenCopy_(tmp2, tmp0, c10::attr::non_blocking);
    }

    // update output
    if (stream_executor.getModelType() == "GNMT") {
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(tmp0));
    } else {
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    }
}

void executeAtenBool(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenCat(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Cat node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    auto cat_layer = std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(layer);

    std::vector<at::Tensor> tensor_vec;

    auto input1_layer = stream_executor.getGraph()->getLayerByPosition((layer->getPreLayerIDs())[0]);
    auto in_input1_stensor_id = input1_layer->getInSTensorID();

    if (stream_executor.getModelType() == "GNMT" && in_input1_stensor_id.size() == 0) {
        int cat_mem_id = cat_layer->getMemLayerId();
        auto output = stream_executor.findBlob(cat_mem_id).second.toTensor().clone();
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        return;
    }

    int cat_mem_id = cat_layer->getMemLayerId();
    if (stream_executor.getModelType() == "GNMT" &&
        stream_executor.findBlob(cat_mem_id).first != ir::DataType::UNDEFINED) {
        auto output = stream_executor.findBlob(cat_mem_id).second;
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

void executeAtenCeil(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenChunk(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenClamp(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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
        min = static_cast<double>(iv.toDouble());
    }
    if (nn_compiler::ir::isDefaultValue(max)) {
        auto iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        max = static_cast<double>(iv.toDouble());
    }

    auto output = atenClamp(self_tensor, min, max);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenClear(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenClone(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Clone node";
    auto clone_layer = std::static_pointer_cast<nn_compiler::ir::AtenCloneLayer>(layer);
    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor(), output;

    auto memory_format = clone_layer->getMemoryFormat();
    if (nn_compiler::ir::isDefaultValue(memory_format)) {
        auto iv = stream_executor.findBlob(in_stensor_id[1]).second;
        if (iv.isInt()) {
            memory_format = static_cast<int>(iv.toInt());
            output = atenClone(self_tensor, getMemoryFormat(memory_format));
        } else {
            // NonType of memory_format
            output = atenClone(self_tensor);
        }
    }

    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenContiguous(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenConv2d(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenCopy(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenCpu(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenCuda(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenCumsum(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Cumsum node";

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    torch::jit::IValue iv_dim = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    torch::jit::IValue iv_dtype = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    at::Tensor output;
    if (iv_dtype.isNone() && iv_dim.isInt()) {
        output = atenCumsum(self_tensor, iv_dim.toInt());
    } else if (!iv_dtype.isNone() && iv_dim.isInt()) {
        auto dtype = iv_dtype.toScalarType();
        output = atenCumsum(self_tensor, iv_dim.toInt(), dtype);
    } else {
        DLOG(FATAL) << "Aten Cumsum has incorrect inputs";
    }
    stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenDeriveIndex(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenDetach(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Detach node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    at::Tensor self_tensor = iv_tensor.toTensor();
    auto output = atenDetach(self_tensor);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenDim(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenDiv(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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
    if (iv_other.isTensor()) {
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = atenDiv(self_tensor, other_tensor);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else if (iv_other.isScalar()) {
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = atenDiv(self_tensor, other_scalar);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::div";
    }
}

void executeAtenDropout(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenEinsum(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Einsum node";

    auto einsum_layer = std::static_pointer_cast<nn_compiler::ir::AtenEinsumLayer>(layer);

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_string = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    assert(iv_string.isString());
    auto equation = iv_string.toStringRef();

    torch::jit::IValue iv_tensors = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    assert(iv_tensors.isTensorList());
    auto tensors_list_ivalue = iv_tensors.toList();
    std::vector<at::Tensor> tensors_list;
    for (torch::jit::IValue iv : tensors_list_ivalue) {
        tensors_list.push_back(iv.toTensor());
    }
    auto output = atenEinsum(equation, tensors_list);
    stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenEmbedding(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenEq(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenEqual(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenExpand(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenFill(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenFloorDivide(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenFormat(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenFullLike(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten FullLike node";

    auto full_like_layer = std::static_pointer_cast<nn_compiler::ir::AtenFullLikeLayer>(layer);

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    auto full_value = full_like_layer->getFullValue();
    if (nn_compiler::ir::isDefaultValue(full_value)) {
        auto full_value_iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        assert(full_value_iv.isInt());
        full_value = full_value_iv.toInt();
    }

    at::TensorOptions options;
    auto iv_dtype = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    auto iv_layout = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    auto iv_device = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    auto iv_pin_memory = stream_executor.findBlob(in_stensor_ids[in_id++]).second;

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

    auto output = atenFullLike(self_tensor, full_value, options);
    stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenGather(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenGe(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Ge node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    if (iv_self.isTensor()) {
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
    } else if (iv_self.isInt()) {
        assert(iv_other.isInt());
        auto self_value = iv_self.toInt();
        auto other_value = iv_other.toInt();
        bool output = (self_value >= other_value);
        stream_executor.updateBlob(out_stensor_id[0], DataType::BOOL, boolToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::ge";
    }
}

void executeAtenGetItem(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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
    stream_executor.insertInRelationBlobIDsMap(out_stensor_id[0], in_stensor_id[0], idx);
    stream_executor.updateBlob(out_stensor_id[0], DataType::IVALUE, output);
}

void executeAtenGt(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenIndex(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

    at::Tensor output;
    bool all_zero_indice = true;
    for (torch::jit::IValue iv : indices_list_ivalue) {
        auto indice = iv.toTensor();
        if (!(indice.dim() == 1 && (indice.sizes())[0] == 1 && indice.item().toInt() == 0)) {
            all_zero_indice = false;
            break;
        }
    }
    if (all_zero_indice) {
        output = self_tensor;
        for (int i = self_tensor.sizes().size(); i > 1; i--) {
            output = atenSqueeze(output, -1);
        }
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        return;
    }

    for (torch::jit::IValue iv : indices_list_ivalue) {
        indices_optional_list.push_back(iv.toOptional<at::Tensor>());
    }

    output = atenIndex(self_tensor, indices_optional_list);
    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenIndexPut(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenIndexSelect(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenInt(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenIntImplicit(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Int Implicit node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();
    auto output = atenIntImplicit(self_tensor);

    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::INT64, intToIValue(output));
}

void executeAtenIs(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenIsInf(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten IsInf node";

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[0]).second;
    assert(iv_self.isTensor());
    auto tensor = iv_self.toTensor();

    auto output = atenIsInf(tensor);
    stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenIsNot(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten IsNot node";

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();
    assert(in_stensor_ids.size() == 2);

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_ids[1]).second;

    auto output = !(atenIs(iv_self, iv_other));
    // update output
    stream_executor.updateBlob(out_stensor_ids[0], DataType::BOOL, boolToIValue(output));
}

void executeAtenItem(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenLayerNorm(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten LayerNorm node";

    auto layer_norm_layer = std::static_pointer_cast<nn_compiler::ir::AtenLayerNormLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();
    auto in_id = 0;

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(iv_tensor.isTensor());
    auto tensor = iv_tensor.toTensor();

    auto normalized_shape = layer_norm_layer->getNormalizedShape();
    assert(normalized_shape.size() != 0);

    auto weight_ids = layer_norm_layer->getWeightIds();
    auto bias_ids = layer_norm_layer->getBiasIds();
    assert(weight_ids.size() == 1 && bias_ids.size() == 1);
    auto weight_iv = stream_executor.findBlob(weight_ids[0]).second;
    auto bias_iv = stream_executor.findBlob(bias_ids[0]).second;
    assert(weight_iv.isTensor() && bias_iv.isTensor());
    at::Tensor weight = weight_iv.toTensor();
    at::Tensor bias = bias_iv.toTensor();

    auto eps = layer_norm_layer->getEps();
    if (nn_compiler::ir::isDefaultValue(eps)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isDouble());
        eps = data_iv.toDouble();
    }
    auto cudnn_enable = layer_norm_layer->getCudnnEnable();
    if (nn_compiler::ir::isDefaultValue(cudnn_enable)) {
        auto data_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(data_iv.isInt());
        cudnn_enable = static_cast<int>(data_iv.toInt());
    }

    at::Tensor output =
        atenLayerNorm(tensor, at::IntArrayRef(normalized_shape), weight, bias, eps, static_cast<bool>(cudnn_enable));

    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenLeakyRelu(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenLe(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Le node";

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[0]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_ids[1]).second;
    at::Tensor output;
    if (iv_other.isScalar()) {
        auto other = iv_other.toScalar();
        output = atenLe(self_tensor, other);
    } else if (iv_other.isTensor()) {
        auto other = iv_other.toTensor();
        output = atenLe(self_tensor, other);
    } else {
        DLOG(FATAL) << "Aten le op's data type do not support!";
    }

    // update output
    stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenLen(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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
        auto input1_layer = stream_executor.getGraph()->getLayerByPosition((layer->getPreLayerIDs())[0]);
        auto out1_layer = stream_executor.getGraph()->getLayerByPosition((layer->getNextLayerIDs())[0]);
        if (stream_executor.getModelType() == "GNMT" &&
            input1_layer->getType() == nn_compiler::ir::LayerType::PRIMVARIABLE && iv.toList().size() == 4) {
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

void executeAtenLinear(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenList(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenLog(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenLogSoftmax(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenLSTM1(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

    auto param_vector = lstm1_layer->getParamVector();
    at::TensorList params(param_vector);
    auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
    out_stensor_id.erase(pos, out_stensor_id.end());
    assert(out_stensor_id.size() == 3);

    int batch_size = 1;
    int in_dim = input.dim();
    int input_size = input.size(in_dim - 1);
    int bidirectional_int = static_cast<bool>(bidirectional) ? 2 : 1;
    int hx_dim = hx_list_tensor_vector[0].dim();
    int hidden_size = hx_list_tensor_vector[0].size(hx_dim - 1);
    int seq_len = input.size(in_dim - 2);
    std::vector<int> in_len({batch_size, input_size});
    std::vector<int> out_len({bidirectional_int * hidden_size});

    auto output = atenLstm1(input, hx, params, static_cast<bool>(has_biases), num_layers, dropout,
                            static_cast<bool>(train), static_cast<bool>(bidirectional), static_cast<bool>(batch_first));
    auto lstm_output = std::get<0>(output);
    auto hidden_output = std::get<1>(output);
    auto cell_output = std::get<2>(output);

    if (stream_executor.getModelType() == "GNMT" && lstm1_layer->getMatchCustomOpt()) {
        int cat_forced_id = lstm1_layer->getCustomCatMemId();
        auto cat_tensor = stream_executor.findBlob(cat_forced_id).second.toTensor();
        std::vector<at::Tensor> hidden_cell_out_vec = {hidden_output, cell_output};
        at::TensorList hidden_cell_out_list(hidden_cell_out_vec);
        auto hidden_cell_out_tensor = atenCat(hidden_cell_out_list, 0);
        auto custom_opt_number = lstm1_layer->getCustomOptNumber();

        if (custom_opt_number == 0) {
            auto cat_sliced_tensor = atenSlice(cat_tensor, 0, 2, 8);
            std::vector<at::Tensor> cat_out_vec = {hidden_cell_out_tensor, cat_sliced_tensor};
            at::TensorList cat_out_list(cat_out_vec);
            auto cat_out_tensor = atenCat(cat_out_list, 0);
            stream_executor.updateBlob(cat_forced_id, DataType::TENSOR, tensorToIValue(cat_out_tensor));
        } else if (custom_opt_number == 1) {
            auto cat_sliced_tensor1 = atenSlice(cat_tensor, 0, 0, 2);
            auto cat_sliced_tensor2 = atenSlice(cat_tensor, 0, 4, 8);
            std::vector<at::Tensor> cat_out_vec = {cat_sliced_tensor1, hidden_cell_out_tensor, cat_sliced_tensor2};
            at::TensorList cat_out_list(cat_out_vec);
            auto cat_out_tensor = atenCat(cat_out_list, 0);
            stream_executor.updateBlob(cat_forced_id, DataType::TENSOR, tensorToIValue(cat_out_tensor));
        } else if (custom_opt_number == 2) {
            auto cat_sliced_tensor1 = atenSlice(cat_tensor, 0, 0, 4);
            auto cat_sliced_tensor2 = atenSlice(cat_tensor, 0, 6, 8);
            std::vector<at::Tensor> cat_out_vec = {cat_sliced_tensor1, hidden_cell_out_tensor, cat_sliced_tensor2};
            at::TensorList cat_out_list(cat_out_vec);
            auto cat_out_tensor = atenCat(cat_out_list, 0);
            stream_executor.updateBlob(cat_forced_id, DataType::TENSOR, tensorToIValue(cat_out_tensor));
        } else if (custom_opt_number == 3) {
            auto cat_sliced_tensor = atenSlice(cat_tensor, 0, 0, 6);
            std::vector<at::Tensor> cat_out_vec = {cat_sliced_tensor, hidden_cell_out_tensor};
            at::TensorList cat_out_list(cat_out_vec);
            auto cat_out_tensor = atenCat(cat_out_list, 0);
            stream_executor.updateBlob(cat_forced_id, DataType::TENSOR, tensorToIValue(cat_out_tensor));
        }

        if (custom_opt_number == 0 || custom_opt_number == 1) {
            if (stream_executor.findBlob(out_stensor_id[0]).first == ir::DataType::UNDEFINED) {
                lstm_output = at::zeros({in_len[0], seq_len, out_len[0]}, input.options());
            }
            auto out_layer = stream_executor.getGraph()->getLayerByPosition(
                (layer->getNextLayerIDs())[custom_opt_number == 0 ? 3 : 0]);
            auto out_out_layer = stream_executor.getGraph()->getLayerByPosition((out_layer->getNextLayerIDs())[0]);
            int64_t cat_mem_id =
                std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(out_out_layer)->getMemLayerId();
            cat_tensor = stream_executor.findBlob(cat_mem_id).second.toTensor();
            auto cat_sliced_tensor = atenSlice(cat_tensor, 2, 1024, 2048);
            std::vector<at::Tensor> cat_out_vec = {lstm_output, cat_sliced_tensor};
            at::TensorList cat_out_list(cat_out_vec);
            auto cat_out_tensor = atenCat(cat_out_list, 2);
#ifdef SELECT_OPTIMAL_LIB
            if (stream_executor.findBlob(out_stensor_id[0]).first == ir::DataType::UNDEFINED ||
                stream_executor.hasSelectOptimalLib()) {
                stream_executor.updateBlob(cat_mem_id, DataType::TENSOR, tensorToIValue(cat_out_tensor));
            }
        }
        if (stream_executor.findBlob(out_stensor_id[0]).first != ir::DataType::UNDEFINED &&
            !stream_executor.hasSelectOptimalLib()) {
            return;
        }
#else
            stream_executor.updateBlob(cat_mem_id, DataType::TENSOR, tensorToIValue(cat_out_tensor));
        }
#endif
    }

    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(lstm_output));
    stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(hidden_output));
    stream_executor.updateBlob(out_stensor_id[2], DataType::TENSOR, tensorToIValue(cell_output));
}

void executeAtenLSTM2(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

    auto param_vector = lstm2_layer->getParamVector();
    at::TensorList params(param_vector);
    auto output = atenLstm2(input, batch_sizes, hx, params, static_cast<bool>(has_biases), num_layers, dropout,
                            static_cast<bool>(train), static_cast<bool>(bidirectional));
    auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
    out_stensor_id.erase(pos, out_stensor_id.end());
    assert(out_stensor_id.size() == 3);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
    stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
    stream_executor.updateBlob(out_stensor_id[2], DataType::TENSOR, tensorToIValue(std::get<2>(output)));
}

void executeAtenLt(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenMaskedFill(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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
        auto ori_id = stream_executor.findInRelationBlobIDsMap(in_stensor_id[0]).first;
        stream_executor.updateBlob(ori_id, DataType::TENSOR, tensorToIValue(output));
    }
}

void executeAtenMaskedSelect(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenMatmul(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Matmul node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor() && iv_other.isTensor());
    auto self_tensor = iv_self.toTensor();
    auto other_tensor = iv_other.toTensor();

    torch::jit::IValue output;
    customAtenMatmul(self_tensor, other_tensor, output);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, output);
}

void executeAtenMatmulWithStream(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor,
                                 void* stream)
{
    DLOG(INFO) << "execute Aten Matmul node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor() && iv_other.isTensor());
    auto self_tensor = iv_self.toTensor();
    auto other_tensor = iv_other.toTensor();

    torch::jit::IValue output;
    customAtenMatmul(self_tensor, other_tensor, output, stream);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, output);
}

void executeAtenMax(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenMaxPool2d(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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
    auto kernel_size = getDataShapeFromVector(max_pool_2d_layer->getKernelSize());
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

    auto output = atenMaxPool2d(self_tensor, at::ArrayRef<int64_t>(kernel_size_vec), at::ArrayRef<int64_t>(stride_vec),
                                at::ArrayRef<int64_t>(padding_vec), at::ArrayRef<int64_t>(dilation_vec),
                                static_cast<bool>(ceil_mode));
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenMean(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Mean node";

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    at::Tensor output;
    if (in_stensor_ids.size() == 4) {
        torch::jit::IValue iv_dim = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        assert(iv_dim.isList());
        auto dim_list = iv_dim.toListRef();
        auto array_ref = parseIValueVector<int64_t>(dim_list);

        torch::jit::IValue iv_keepdim = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        auto keepdim = iv_keepdim.toInt();

        torch::jit::IValue iv_dtype = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        if (iv_dtype.isNone()) {
            output = atenMean(self_tensor, at::ArrayRef<int64_t>(array_ref), (bool)keepdim);
        } else {
            auto dtype = iv_dtype.toScalarType();
            output = atenMean(self_tensor, at::ArrayRef<int64_t>(array_ref), (bool)keepdim, dtype);
        }
        stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(FATAL) << "Aten mean's input size is incorrect! size: " << in_stensor_ids.size();
    }
}

void executeAtenMin(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenMul(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenNe(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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
        at::Tensor self_tensor = iv_self.toTensor(), output;
        if (iv_other.isTensor()) {
            at::Tensor other_tensor = iv_other.toTensor();
            output = atenNe(self_tensor, other_tensor);
        } else if (iv_other.isScalar()) {
            at::Scalar other_scalar = iv_other.toScalar();
            output = atenNe(self_tensor, other_scalar);
        } else {
            DLOG(FATAL) << "Unsupported input type for aten::ne";
        }
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

void executeAtenNeg(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Neg node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();
    assert(in_stensor_id.size() == 1);

    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_id[0]).second;

    if (iv.isScalar()) {
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

void executeAtenNorm(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenNot(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenOneHot(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten One hot node";

    auto one_hot_layer = std::static_pointer_cast<nn_compiler::ir::AtenOneHotLayer>(layer);

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    int in_id = 0;
    auto iv_self = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    auto num_classes = one_hot_layer->getNumClasses();
    if (nn_compiler::ir::isDefaultValue(num_classes)) {
        auto iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        assert(iv.isInt());
        num_classes = iv.toInt();
    }
    auto output = atenOneHot(self_tensor, num_classes);
    stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenOnes(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenPackPaddedSequence(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenPadPackedSequence(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

    auto output =
        atenPadPackedSequence(self_tensor, other_tensor, static_cast<bool>(batch_first), padding_value, total_length);
    auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
    out_stensor_id.erase(pos, out_stensor_id.end());
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
    stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
}

void executeAtenPermute(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Permute node";

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_ids[1]).second;

    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    assert(iv_other.isList());
    auto other_list = iv_other.toListRef();
    auto array_ref = parseIValueVector<int64_t>(other_list);

    auto output = atenPermute(self_tensor, at::ArrayRef<int64_t>(array_ref));
    stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenPow(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenRelu(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenReshape(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenRepeat(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Repeat node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    assert(iv_other.isList());
    std::vector<int64_t> repeats;
    for (auto item : iv_other.toList().vec()) {
        int64_t val = item.toInt();
        repeats.push_back(val);
    }
    auto output = atenRepeat(self_tensor, at::IntArrayRef(repeats));

    // update output
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenRsqrt(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Rsqrt node";

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[0]).second;
    assert(iv_self.isTensor());
    auto tensor = iv_self.toTensor();

    auto output = atenRsqrt(tensor);
    stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenRemainder(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Remainde node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_id[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_id[1]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    if (iv_other.isScalar()) {
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = atenRemainder(self_tensor, other_scalar);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::ge";
    }
}

void executeAtenSelect(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenSetItem(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenSize(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenSlice(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenSoftmax(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenSqueeze(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenSub(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Sub node";

    auto sub_layer = std::static_pointer_cast<nn_compiler::ir::AtenSubLayer>(layer);

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    int64_t alpha = sub_layer->getAlpha();
    if (nn_compiler::ir::isDefaultValue(alpha) && in_stensor_ids.size() == 3) {
        auto alpha_iv = stream_executor.findBlob(in_stensor_ids[2]).second;
        assert(alpha_iv.isInt());
        alpha = alpha_iv.toInt();
    } else {
        // default value for alpha is 1
        alpha = 1;
    }

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_ids[1]).second;
    if (iv_self.isTensor()) {
        at::Tensor self_tensor = iv_self.toTensor();
        if (iv_other.isTensor()) {
            at::Tensor other_tensor = iv_other.toTensor();
            auto output = atenSub(self_tensor, other_tensor, alpha);
            // update output
            stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
        } else if (iv_other.isScalar()) {
            at::Scalar other_scalar = iv_other.toScalar();
            auto output = atenSub(self_tensor, other_scalar, alpha);
            // update output
            stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
        } else {
            DLOG(FATAL) << "Unsupported input type for aten::sub";
        }
    } else if (iv_self.isInt()) {
        assert(iv_other.isInt());
        auto output = iv_self.toInt() - iv_other.toInt();
        stream_executor.updateBlob(out_stensor_ids[0], DataType::INT64, intToIValue(output));
    } else {
        DLOG(FATAL) << "Unsupported input type for aten::sub";
    }
}

void executeAtenSum(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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
        output = atenSum(self_tensor, at::ArrayRef<int64_t>(dims), static_cast<bool>(keepdim), at::ScalarType(dtype));
    }
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenTanh(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenTensor(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenTo1(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten To1 node";

    auto to1_layer = std::static_pointer_cast<nn_compiler::ir::AtenTo1Layer>(layer);

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto str_device = to1_layer->getDevice();
    at::Device device("cpu");
    if (nn_compiler::ir::isDefaultValue(str_device)) {
        auto iv_device = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        if (!iv_device.isNone()) {
            device = iv_device.toDevice();
        } else {
            DLOG(FATAL) << "None device found for aten::to layer.";
        }
    } else {
        device = at::Device(device);
    }

    auto ori_dtype = to1_layer->getDType();
    if (nn_compiler::ir::isDefaultValue(ori_dtype)) {
        auto ori_dtype_iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        assert(ori_dtype_iv.isInt());
        ori_dtype = ori_dtype_iv.toInt();
    }
    auto dtype = at::ScalarType(ori_dtype);

    int non_blocking_val = to1_layer->getNonBlocking();
    if (nn_compiler::ir::isDefaultValue(non_blocking_val)) {
        auto non_blocking_iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        if (non_blocking_iv.isNone()) {
            non_blocking_val = 0;
        } else {
            non_blocking_val = non_blocking_iv.toInt();
        }
    }
    bool non_blocking = static_cast<bool>(non_blocking_val);

    int copy_val = to1_layer->getCopy();
    if (nn_compiler::ir::isDefaultValue(copy_val)) {
        auto copy_iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        if (copy_iv.isNone()) {
            copy_val = 0;
        } else {
            copy_val = copy_iv.toInt();
        }
    }
    bool copy = static_cast<bool>(copy_val);

    auto optional_memory_format = to1_layer->getOptionalMemoryFormat();
    if (in_stensor_ids.size() > in_id) {
        auto optional_memory_format_iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        if (optional_memory_format_iv.isInt()) {
            optional_memory_format = optional_memory_format_iv.toInt();
        } else {
            assert(optional_memory_format_iv.isNone());
        }
    }

    if (optional_memory_format == -1) {  // optional_memory_format = NONE
        auto output = atenTo(self_tensor, device, dtype, non_blocking, copy);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        auto memory_format = getMemoryFormat(optional_memory_format);
        auto output = atenTo(self_tensor, device, dtype, non_blocking, copy, memory_format);
        // update output
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
    }
}

void executeAtenTo2(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten To2 node";

    auto to2_layer = std::static_pointer_cast<nn_compiler::ir::AtenTo2Layer>(layer);

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto ori_dtype = to2_layer->getDType();
    if (nn_compiler::ir::isDefaultValue(ori_dtype)) {
        auto ori_dtype_iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        assert(ori_dtype_iv.isInt());
        ori_dtype = ori_dtype_iv.toInt();
    }
    auto dtype = at::ScalarType(ori_dtype);

    int non_blocking_val = to2_layer->getNonBlocking();
    if (nn_compiler::ir::isDefaultValue(non_blocking_val)) {
        auto non_blocking_iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        if (non_blocking_iv.isNone()) {
            non_blocking_val = 0;
        } else {
            non_blocking_val = non_blocking_iv.toInt();
        }
    }
    bool non_blocking = static_cast<bool>(non_blocking_val);

    int copy_val = to2_layer->getCopy();
    if (nn_compiler::ir::isDefaultValue(copy_val)) {
        auto copy_iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        if (copy_iv.isNone()) {
            copy_val = 0;
        } else {
            copy_val = copy_iv.toInt();
        }
    }
    bool copy = static_cast<bool>(copy_val);

    auto optional_memory_format = to2_layer->getOptionalMemoryFormat();
    if (in_stensor_ids.size() > in_id) {
        auto optional_memory_format_iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        if (optional_memory_format_iv.isInt()) {
            optional_memory_format = optional_memory_format_iv.toInt();
        } else {
            assert(optional_memory_format_iv.isNone());
        }
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

void executeAtenTo3(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten To3 node";

    auto to3_layer = std::static_pointer_cast<nn_compiler::ir::AtenTo3Layer>(layer);

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    assert(iv_other.isTensor());
    at::Tensor other_tensor = iv_other.toTensor();

    int non_blocking_val = to3_layer->getNonBlocking();
    if (nn_compiler::ir::isDefaultValue(non_blocking_val)) {
        auto non_blocking_iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        if (non_blocking_iv.isNone()) {
            non_blocking_val = 0;
        } else {
            non_blocking_val = non_blocking_iv.toInt();
        }
    }
    bool non_blocking = static_cast<bool>(non_blocking_val);

    int copy_val = to3_layer->getCopy();
    if (nn_compiler::ir::isDefaultValue(copy_val)) {
        auto copy_iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        if (copy_iv.isNone()) {
            copy_val = 0;
        } else {
            copy_val = copy_iv.toInt();
        }
    }
    bool copy = static_cast<bool>(copy_val);

    auto optional_memory_format = to3_layer->getOptionalMemoryFormat();
    if (in_stensor_ids.size() > in_id) {
        auto optional_memory_format_iv = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        if (optional_memory_format_iv.isInt()) {
            optional_memory_format = optional_memory_format_iv.toInt();
        } else {
            assert(optional_memory_format_iv.isNone());
        }
    }

    if (optional_memory_format == -1) {  // optional_memory_format = NONE
        auto output = atenTo(self_tensor, other_tensor, non_blocking, copy);
        // update output
        stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        auto memory_format = getMemoryFormat(optional_memory_format);
        auto output = atenTo(self_tensor, other_tensor, non_blocking, copy, memory_format);
        // update output
        stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
    }
}

void executeAtenTopk(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenTranspose(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenTriu(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Triu node";

    auto triu_layer = std::static_pointer_cast<nn_compiler::ir::AtenTriuLayer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv_tensor = stream_executor.findBlob(in_stensor_id[0]).second;
    assert(iv_tensor.isTensor());
    auto self_tensor = iv_tensor.toTensor();

    auto diagonal = triu_layer->getDiagonal();
    if (nn_compiler::ir::isDefaultValue(diagonal)) {
        auto diagonal_iv = stream_executor.findBlob(in_stensor_id[1]).second;
        assert(diagonal_iv.isInt());
        diagonal = diagonal_iv.toInt();
    }
    auto output = atenTriu(self_tensor, diagonal);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenTypeAs(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Type as node";

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[0]).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_ids[0]).second;
    assert(iv_self.isTensor() && iv_other.isTensor());

    auto self_tensor = iv_self.toTensor();
    auto other_tensor = iv_other.toTensor();
    auto output = atenTypeAs(self_tensor, other_tensor);
    stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenUnsqueeze(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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
        auto releation_blob_ids = stream_executor.findInRelationBlobIDsMap(in_stensor_id[0]);
        auto list_blob_id = releation_blob_ids.first;
        auto in_list_pos = releation_blob_ids.second;

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

void executeAtenView(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

void executeAtenWarn(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute AtenWarn node";

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_id[0]).second;

    atenWarn(iv.toString()->string());
}

void executeAtenWhere(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Where node";

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    int in_id = 0;
    torch::jit::IValue iv_condition = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
    assert(iv_condition.isTensor());
    auto condition = iv_condition.toTensor();
    if (in_stensor_ids.size() == 3) {
        torch::jit::IValue iv_self = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        torch::jit::IValue iv_other = stream_executor.findBlob(in_stensor_ids[in_id++]).second;
        at::Tensor output;
        if (iv_self.isTensor() && iv_other.isTensor()) {
            output = atenWhere(condition, iv_self.toTensor(), iv_other.toTensor());
        } else if (iv_self.isScalar() && iv_other.isTensor()) {
            output = atenWhere(condition, iv_self.toScalar(), iv_other.toTensor());
        } else if (iv_self.isTensor() && iv_other.isScalar()) {
            output = atenWhere(condition, iv_self.toTensor(), iv_other.toScalar());
        } else if (iv_self.isScalar() && iv_other.isScalar()) {
            output = atenWhere(condition, iv_self.toScalar(), iv_other.toScalar());
        } else {
            DLOG(FATAL) << "aten::where node's input data type is incorrect! ";
        }
        stream_executor.updateBlob(out_stensor_ids[0], DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(FATAL) << "aten::where node's input size is: " << in_stensor_ids.size();
    }
}

void executeAtenZeros(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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
        DLOG(INFO) << "iv_device :" << iv_device.toStringRef();
    } else {
        options = options.device(at::kCUDA);
    }
    if (!iv_pin_memory.isNone()) {
        options = options.pinned_memory(iv_pin_memory.toBool());
    }

    auto output = atenZeros(at::ArrayRef<int64_t>(array_ref), options);
    stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
}

void executeAtenZerosLike(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
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

}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler
