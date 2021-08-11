#include <algorithm>
#include <vector>
#include "c10/hip/HIPFunctions.h"
#include "common/include/cast.hpp"
#include "executor/aten_ops.h"
#include "executor/stream_executor.h"
#include "tv_tools.h"
#include "executor/utils.h"
#include "glog/logging.h"
#include "ir/include/all_nodes.hpp"
#include "ir/include/common/utils.hpp"
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/ir_types.hpp"
#include "ir/include/nn_ir.hpp"
#include "executor/aten_ops_executor.h"


namespace nnrt
{
void executorAtenAdd(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Add node";

    auto add_node = cast<nncir::AtenAddNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(add_node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(add_node.getInEdge(1));
    int64_t alpha = add_node.getAlpha();
    if (nncir::isDefaultValue(alpha)) {
        if (add_node.getInEdgeIds().size() == 3) {
            auto& alpha_edge = cast<nncir::DataEdge>(add_node.getInEdge(2));
            auto edge_name = alpha_edge.getName();
            //if "prim::if" layer linked to current, this edge has no practical meaning 
            if (edge_name.find("prim::If") == std::string::npos) {
                int alpha_blob_id = alpha_edge.getBlobId();
                auto alpha_iv = stream_executor.findBlob(alpha_blob_id).second;
                assert(alpha_iv.isInt());
                alpha = alpha_iv.toInt();
            } else {
                alpha = 1;
            }

        } else {
            alpha = 1;
        }
    }

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    if (iv_self.isInt() && iv_other.isInt()) {
        int64_t in_self = iv_self.toInt();
        auto output = nnrt::atenAdd(in_self, iv_other.toInt(), alpha);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(add_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, intToIValue(output));
        return;
    }
    at::Tensor self_tensor = iv_self.toTensor();
    auto dtype = stream_executor.findBlob(input_other_blob_id).first;
    if (dtype == DataType::TENSOR) {
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenAdd(self_tensor, other_tensor, alpha);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(add_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (isScalarType(dtype)) {
        assert(iv_other.isScalar());
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = nnrt::atenAdd(self_tensor, other_scalar, alpha);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(add_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::add";
    }
}  // executorAtenAdd

void executorAtenAddmm(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Addmm node";

    auto addmm_node = cast<nncir::AtenAddmmNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(addmm_node.getInEdge(0));
    auto& input_mat1 = cast<nncir::DataEdge>(addmm_node.getInEdge(1));
    auto& input_mat2 = cast<nncir::DataEdge>(addmm_node.getInEdge(2));
    auto& input_beta = cast<nncir::DataEdge>(addmm_node.getInEdge(3));
    auto& input_alpha = cast<nncir::DataEdge>(addmm_node.getInEdge(4));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_mat1_blob_id = input_mat1.getBlobId();
    int input_mat2_blob_id = input_mat2.getBlobId();
    int input_beta_blob_id = input_beta.getBlobId();
    int input_alpha_blob_id = input_alpha.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_mat1 = stream_executor.findBlob(input_mat1_blob_id).second;
    torch::jit::IValue iv_mat2 = stream_executor.findBlob(input_mat2_blob_id).second;
    assert(iv_self.isTensor() && iv_mat1.isTensor() && iv_mat2.isTensor());
    torch::jit::IValue iv_beta = stream_executor.findBlob(input_beta_blob_id).second;
    torch::jit::IValue iv_alpha = stream_executor.findBlob(input_alpha_blob_id).second;

    auto output = nnrt::atenAddmm(iv_self.toTensor(), iv_mat1.toTensor(), iv_mat2.toTensor(), iv_beta.toScalar(),
                                  iv_alpha.toScalar());
    // update output
    auto& out_edge = cast<nncir::DataEdge>(addmm_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenAnd(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten And node";

    auto node = cast<nncir::AtenAndNode>(op_node);

    auto& input_a = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_b = cast<nncir::DataEdge>(node.getInEdge(1));

    // Get input blob
    int input_a_blob_id = input_a.getBlobId();
    int input_b_blob_id = input_b.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_a = stream_executor.findBlob(input_a_blob_id).second;
    torch::jit::IValue iv_b = stream_executor.findBlob(input_b_blob_id).second;

    assert(iv_a.isBool() && iv_b.isBool());
    auto value_a = iv_a.toBool();
    auto value_b = iv_b.toBool();

    auto output = nnrt::atenAnd(value_a, value_b);
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::BOOL, boolToIValue(output));
}

void executorAtenAny(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Any node";

    auto node = cast<nncir::AtenAnyNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    auto output = nnrt::atenAny(self_tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, torch::jit::IValue(output));
}

void executorAtenAppend(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Append node";

    auto node = cast<nncir::AtenAppendNode>(op_node);
    assert(node.getNumInputs() == 2);

    auto& input_list = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_el = cast<nncir::DataEdge>(node.getInEdge(1));

    // Get input blob
    int input_list_blob_id = input_list.getBlobId();
    int input_el_blob_id = input_el.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_list = stream_executor.findBlob(input_list_blob_id).second;
    torch::jit::IValue iv_el = stream_executor.findBlob(input_el_blob_id).second;

    assert(iv_list.isList());

    c10::List<at::IValue> list = iv_list.toList();
    nnrt::atenAppend(list, iv_el);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::LIST, listToIValue(list));
}

void executorAtenArange1(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Arange1 node";

    auto node = cast<nncir::AtenArange1Node>(op_node);
    int edge_id = 0;
    auto end = node.getEnd();
    if (nncir::isDefaultValue(end)) {
        auto& end_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int end_blob_id = end_edge.getBlobId();
        auto iv_end = stream_executor.findBlob(end_blob_id).second;
        end = iv_end.toInt();
    }

    at::TensorOptions options;
    auto dtype = node.getDtype();
    if (nncir::isDefaultValue(dtype)) {
        auto& edge_dtype = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto dtype_id = edge_dtype.getBlobId();
        auto iv_dtype = stream_executor.findBlob(dtype_id).second;
        if (!iv_dtype.isNone()) {
            options = options.dtype(iv_dtype.toScalarType());
        }
    } else {
        options = options.dtype(at::ScalarType(dtype));
    }

    auto layout = node.getLayout();
    if (nncir::isDefaultValue(layout)) {
        auto& edge_layout = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto layout_id = edge_layout.getBlobId();
        auto iv_layout = stream_executor.findBlob(layout_id).second;
        if (!iv_layout.isNone()) {
            options = options.layout(iv_layout.toLayout());
        }
    } else {
        options = options.layout(at::Layout(layout));
    }

    auto device = node.getDevice();
    if (nncir::isDefaultValue(device)) {
        auto& edge_device = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto device_id = edge_device.getBlobId();
        auto iv_device = stream_executor.findBlob(device_id).second;
        if (!iv_device.isNone()) {
            options = options.device(iv_device.toDevice());
        }
    } else {
        options = options.device(device);
    }

    auto pin_memory = node.getPinMemory();
    if (nncir::isDefaultValue(pin_memory)) {
        auto& edge_pin_memory = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto pin_memory_id = edge_pin_memory.getBlobId();
        auto iv_pin_memory = stream_executor.findBlob(pin_memory_id).second;
        if (!iv_pin_memory.isNone()) {
            options = options.pinned_memory(iv_pin_memory.toBool());
        }
    } else {
        options = options.pinned_memory(static_cast<bool>(pin_memory));
    }

    auto output = nnrt::atenArange1(end, options);
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenArange2(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Arange2 node";

    auto node = cast<nncir::AtenArange2Node>(op_node);
    int edge_id = 0;

    auto start = node.getStart();
    if (nncir::isDefaultValue(start)) {
        auto& start_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int start_blob_id = start_edge.getBlobId();
        auto iv_start = stream_executor.findBlob(start_blob_id).second;
        start = iv_start.toInt();
    }

    auto end = node.getEnd();
    if (nncir::isDefaultValue(end)) {
        auto& end_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int end_blob_id = end_edge.getBlobId();
        auto iv_end = stream_executor.findBlob(end_blob_id).second;
        end = iv_end.toInt();
    }

    at::TensorOptions options;
    auto dtype = node.getDtype();
    if (nncir::isDefaultValue(dtype)) {
        auto& edge_dtype = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto dtype_id = edge_dtype.getBlobId();
        auto iv_dtype = stream_executor.findBlob(dtype_id).second;
        if (!iv_dtype.isNone()) {
            options = options.dtype(iv_dtype.toScalarType());
        }
    } else {
        options = options.dtype(at::ScalarType(dtype));
    }

    auto layout = node.getLayout();
    if (nncir::isDefaultValue(layout)) {
        auto& edge_layout = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto layout_id = edge_layout.getBlobId();
        auto iv_layout = stream_executor.findBlob(layout_id).second;
        if (!iv_layout.isNone()) {
            options = options.layout(iv_layout.toLayout());
        }
    } else {
        options = options.layout(at::Layout(layout));
    }

    auto device = node.getDevice();
    if (nncir::isDefaultValue(device)) {
        auto& edge_device = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto device_id = edge_device.getBlobId();
        auto iv_device = stream_executor.findBlob(device_id).second;
        if (!iv_device.isNone()) {
            options = options.device(iv_device.toDevice());
        }
    } else {
        options = options.device(device);
    }

    auto pin_memory = node.getPinMemory();
    if (nncir::isDefaultValue(pin_memory)) {
        auto& edge_pin_memory = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto pin_memory_id = edge_pin_memory.getBlobId();
        auto iv_pin_memory = stream_executor.findBlob(pin_memory_id).second;
        if (!iv_pin_memory.isNone()) {
            options = options.pinned_memory(iv_pin_memory.toBool());
        }
    } else {
        options = options.pinned_memory(static_cast<bool>(pin_memory));
    }

    auto output = nnrt::atenArange2(start, end, options);
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenArange3(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Arange3 node";

    auto node = cast<nncir::AtenArange3Node>(op_node);
    int edge_id = 0;

    auto start = node.getStart();
    if (nncir::isDefaultValue(start)) {
        auto& start_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int start_blob_id = start_edge.getBlobId();
        auto iv_start = stream_executor.findBlob(start_blob_id).second;
        start = iv_start.toInt();
    }

    auto end = node.getEnd();
    if (nncir::isDefaultValue(end)) {
        auto& end_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int end_blob_id = end_edge.getBlobId();
        auto iv_end = stream_executor.findBlob(end_blob_id).second;
        end = iv_end.toInt();
    }

    auto step = node.getStep();
    if (nncir::isDefaultValue(step)) {
        auto& step_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int step_blob_id = step_edge.getBlobId();
        auto iv_step = stream_executor.findBlob(step_blob_id).second;
        step = iv_step.toInt();
    }

    at::TensorOptions options;
    auto dtype = node.getDtype();
    if (nncir::isDefaultValue(dtype)) {
        auto& edge_dtype = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto dtype_id = edge_dtype.getBlobId();
        auto iv_dtype = stream_executor.findBlob(dtype_id).second;
        if (!iv_dtype.isNone()) {
            options = options.dtype(iv_dtype.toScalarType());
        }
    } else {
        options = options.dtype(at::ScalarType(dtype));
    }

    auto layout = node.getLayout();
    if (nncir::isDefaultValue(layout)) {
        auto& edge_layout = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto layout_id = edge_layout.getBlobId();
        auto iv_layout = stream_executor.findBlob(layout_id).second;
        if (!iv_layout.isNone()) {
           options = options.layout(iv_layout.toLayout());
        }
    } else {
        options = options.layout(at::Layout(layout));
    }

    auto device = node.getDevice();
    if (nncir::isDefaultValue(device)) {
        auto& edge_device = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto device_id = edge_device.getBlobId();
        auto iv_device = stream_executor.findBlob(device_id).second;
        if (!iv_device.isNone()) {
            options = options.device(iv_device.toDevice());
        }
    } else {
        options = options.device(device);
    }

    auto pin_memory = node.getPinMemory();
    if (nncir::isDefaultValue(pin_memory)) {
        auto& edge_pin_memory = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto pin_memory_id = edge_pin_memory.getBlobId();
        auto iv_pin_memory = stream_executor.findBlob(pin_memory_id).second;
        if (!iv_pin_memory.isNone()) {
            options = options.pinned_memory(iv_pin_memory.toBool());
        }
    } else {
        options = options.pinned_memory(static_cast<bool>(pin_memory));
    }

    auto output = nnrt::atenArange3(start, end, step, options);
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenAsTensor(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten AsTensor node";

    auto node = cast<nncir::AtenAsTensorNode>(op_node);
    int edge_id = 0;
    auto& input_tensor = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    auto in_tensor = iv_tensor.toTensor();
    edge_id++;

    auto int_dtype = node.getDtype();
    if (nncir::isDefaultValue(int_dtype)) {
        auto& dtype_data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto dtype_blob_id = dtype_data_edge.getBlobId();
        auto iv = stream_executor.findBlob(dtype_blob_id).second;
        int_dtype = iv.toInt();
    }
    auto dtype = at::ScalarType(int_dtype);

    auto str_device = node.getDevice();
    if (nncir::isDefaultValue(str_device)) {
        auto& device_data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto device_blob_id = device_data_edge.getBlobId();
        auto map_value = stream_executor.findBlob(device_blob_id);
        auto iv = map_value.second;
        if (map_value.first != DataType::NONE) {
            str_device = iv.toDevice().str();
        } else {
            str_device = in_tensor.device().str();
        }
    }
    auto device = at::Device(str_device);

    auto output = nnrt::atenAsTensor(in_tensor, dtype, device);
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenBitwiseNot(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten BitwiseNot node";

    auto node = cast<nncir::AtenBitwiseNotNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    auto output = nnrt::atenBitwiseNot(self_tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, torch::jit::IValue(output));
}

void executorAtenBmm(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Bmm node";

    auto bmm_node = cast<nncir::AtenBmmNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(bmm_node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(bmm_node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    assert(iv_self.isTensor() && iv_other.isTensor());

    auto output = nnrt::atenBmm(iv_self.toTensor(), iv_other.toTensor());
    // update output
    auto& out_edge = cast<nncir::DataEdge>(bmm_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenBool(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Bool node";

    auto node = cast<nncir::AtenBoolNode>(op_node);

    auto& input_list_edge = cast<nncir::DataEdge>(node.getInEdge(0));
    auto input_blob_id = input_list_edge.getBlobId();
    auto ivalue = stream_executor.findBlob(input_blob_id).second;
    bool output = false;
    if (ivalue.isTensor()) {
        auto tensor = ivalue.toTensor();
        output = nnrt::atenBool(tensor);
    } else if (ivalue.isInt()) {
        auto integer = ivalue.toInt();
        output = nnrt::atenBool(integer);
    } else if (ivalue.isDouble()) {
        auto double_value = ivalue.toDouble();
        output = nnrt::atenBool(double_value);
    } else {
        DLOG(ERROR) << "Unsupported type for aten::Bool.";
    }
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::BOOL, boolToIValue(output));
}

void executorAtenCat(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Cat node";

    auto cat_node = cast<nncir::AtenCatNode>(op_node);
    std::vector<at::Tensor> tensor_vec;

    auto& input_list_edge = cast<nncir::DataEdge>(cat_node.getInEdge(0));
    auto input_blob_id = input_list_edge.getBlobId();

    auto ivalue = stream_executor.findBlob(input_blob_id).second;
    assert(ivalue.isTensorList());

    auto c10_tensor_list = ivalue.toTensorList();
    for (auto tensor : c10_tensor_list) {
        tensor_vec.push_back(tensor);
    }
    at::TensorList tensor_list(tensor_vec);

    auto dim = cat_node.getDim();
    if (nncir::isDefaultValue(dim)) {
        auto& dim_edge = cast<nncir::DataEdge>(cat_node.getInEdge(1));
        int dim_blob_id = dim_edge.getBlobId();
        auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }

    auto output = nnrt::atenCat(tensor_list, dim);
    auto& out_edge = cast<nncir::DataEdge>(cat_node.getFirstOutEdge());
    // stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorListToIValue(output));
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenCeil(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Ceil node";

    auto node = cast<nncir::AtenCeilNode>(op_node);
    assert(node.getNumInputs() == 1);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    auto output = nnrt::atenCeil(self_tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, torch::jit::IValue(output));
}

void executorAtenChunk(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Chunk node";

    auto node = cast<nncir::AtenChunkNode>(op_node);
    int edge_id = 0;
    auto& input_tensor = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    auto in_tensor = iv_tensor.toTensor();
    edge_id++;

    auto chunks = node.getChunks();
    if (nncir::isDefaultValue(chunks)) {
        auto& chunks_data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto chunks_blob_id = chunks_data_edge.getBlobId();
        auto iv = stream_executor.findBlob(chunks_blob_id).second;
        chunks = iv.toInt();
    }

    auto dim = node.getDim();
    if (nncir::isDefaultValue(chunks)) {
        auto& dim_data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto dim_blob_id = dim_data_edge.getBlobId();
        auto iv = stream_executor.findBlob(dim_blob_id).second;
        dim = iv.toInt();
    }

    auto output = nnrt::atenChunk(in_tensor, chunks, dim);
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::LIST, vectorToIValue(output));
}

void executorAtenClamp(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Clamp node";

    auto node = cast<nncir::AtenClampNode>(op_node);
    int edge_id = 0;

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    edge_id++;

    auto min = node.getMin();
    auto max = node.getMax();
    if (nncir::isDefaultValue(min)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto blob_id = data_edge.getBlobId();
        auto iv = stream_executor.findBlob(blob_id).second;
        min = static_cast<int>(iv.toInt());
    }
    if (nncir::isDefaultValue(max)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto blob_id = data_edge.getBlobId();
        auto iv = stream_executor.findBlob(blob_id).second;
        max = static_cast<int>(iv.toInt());
    }

    auto output = nnrt::atenClamp(self_tensor, min, max);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenClear(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Clear node";

    auto node = cast<nncir::AtenClearNode>(op_node);

    auto& input_self_edge = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_self_blob_id = input_self_edge.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isList());
    at::List self_list = iv_self.toList();
    nnrt::atenClear(self_list);
    // update list
    stream_executor.updateBlob(input_self_blob_id, DataType::LIST, torch::jit::IValue(self_list));
}

void executorAtenContiguous(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Contiguous node";

    auto node = cast<nncir::AtenContiguousNode>(op_node);
    int edge_id = 0;

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    edge_id++;

    auto memory_format = node.getMemoryFormat();
    if (nncir::isDefaultValue(memory_format)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto blob_id = data_edge.getBlobId();
        auto iv = stream_executor.findBlob(blob_id).second;
        memory_format = static_cast<int>(iv.toInt());
    }

    auto output = nnrt::atenContiguous(self_tensor, getMemoryFormat(memory_format));
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenConv2d(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Conv2d node";

    auto node = cast<nncir::AtenConv2dNode>(op_node);

    auto& input_self_edge = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_self_blob_id = input_self_edge.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto weight_blob_id = (node.getWeightBlobId())[0];
    auto bias_blob_id = (node.getBiasBlobId())[0];
    std::vector<at::Tensor> weights;
    auto weight_iv = stream_executor.findBlob(weight_blob_id).second;
    auto bias_iv = stream_executor.findBlob(bias_blob_id).second;
    assert(weight_iv.isTensor() && bias_iv.isTensor());
    auto weight_tensor = weight_iv.toTensor();
    auto bias_tensor = bias_iv.toTensor();

    // attributes of conv2d don't need default-value check, because its default values
    // are set as same as default values in aten::conv2d.
    auto stride = node.getStride();
    auto padding = node.getPadding();
    auto dilation = node.getDilation();
    auto groups = node.getGroups();

    std::vector<int64_t> stride_vec = {static_cast<int64_t>(stride.h), static_cast<int64_t>(stride.w)};
    std::vector<int64_t> padding_vec = {static_cast<int64_t>(padding.l), static_cast<int64_t>(padding.r)};
    std::vector<int64_t> dilation_vec = {static_cast<int64_t>(dilation.h), static_cast<int64_t>(dilation.w)};

    // DLOG(INFO) << "weight: " << weight_tensor;
    // DLOG(INFO) << "bias: " << bias_tensor;


    auto output = nnrt::atenConv2d(self_tensor, weight_tensor, bias_tensor, at::ArrayRef<int64_t>(stride_vec),
                                   at::ArrayRef<int64_t>(padding_vec), at::ArrayRef<int64_t>(dilation_vec), groups);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenCopy(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Copy node";
    auto copy_node = cast<nncir::AtenCopyNode>(op_node);
    // assert(copy_node.getNumInputs() == 2);

    auto& input_self = cast<nncir::DataEdge>(copy_node.getInEdge(0));
    auto& input_src = cast<nncir::DataEdge>(copy_node.getInEdge(1));
    int input_self_blob_id = input_self.getBlobId();
    int input_src_blob_id = input_src.getBlobId();

    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_src = stream_executor.findBlob(input_src_blob_id).second;
    at::Tensor self_tensor = iv_self.toTensor();
    at::Tensor src_tensor = iv_src.toTensor();

    int non_blocking = copy_node.getNonBlocking();
    if (nncir::isDefaultValue(non_blocking)) {
        auto& non_blocking_data_edge = cast<nncir::DataEdge>(copy_node.getInEdge(2));
        int non_blocking_blob_id = non_blocking_data_edge.getBlobId();
        auto non_blocking_iv = stream_executor.findBlob(non_blocking_blob_id).second;
        non_blocking = non_blocking_iv.toInt();
    }

    auto output = nnrt::atenCopy_(self_tensor, src_tensor, static_cast<bool>(non_blocking));
    // update output
    auto& out_edge = cast<nncir::DataEdge>(copy_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, torch::jit::IValue(output));
}

void executorAtenCpu(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Cpu node";

    auto node = cast<nncir::AtenCpuNode>(op_node);

    auto& input_self_edge = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_self_blob_id = input_self_edge.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto output = nnrt::atenCpu(self_tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenCuda(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Cuda node";

    auto node = cast<nncir::AtenCudaNode>(op_node);

    auto& input_self_edge = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_self_blob_id = input_self_edge.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto output = nnrt::atenCuda(self_tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenDeriveIndex(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten derive_index node";

    auto derive_index_node = cast<nncir::AtenDeriveIndexNode>(op_node);
    auto inedges = derive_index_node.getInEdgeIds();

    // Index is an input, get Index
    int edge_id = 0;
    auto& data_edge = cast<nncir::DataEdge>(derive_index_node.getInEdge(edge_id));
    auto input_blob_id = data_edge.getBlobId();
    torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;
    auto index = iv.toInt();
    edge_id++;

    // Check and get start
    auto start = derive_index_node.getStart();
    if (nncir::isDefaultValue(start)) {
        auto& start_data_edge = cast<nncir::DataEdge>(derive_index_node.getInEdge(edge_id));
        auto start_blob_id = start_data_edge.getBlobId();
        auto start_iv = stream_executor.findBlob(start_blob_id).second;
        start = start_iv.toInt();
        edge_id++;
    }

    // Check and get step
    auto step = derive_index_node.getStep();
    if (nncir::isDefaultValue(step)) {
        auto& step_data_edge = cast<nncir::DataEdge>(derive_index_node.getInEdge(edge_id));
        auto step_blob_id = step_data_edge.getBlobId();
        auto step_iv = stream_executor.findBlob(step_blob_id).second;
        step = step_iv.toInt();
        edge_id++;
    }

    auto output = nnrt::atenDeriveIndex(index, start, step);
    auto& out_edge = cast<nncir::DataEdge>(derive_index_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, scalarToIValue(output));
}

void executorAtenDim(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Dim node";

    auto node = cast<nncir::AtenDimNode>(op_node);
    assert(node.getNumInputs() == 1);

    auto& input_tensor = cast<nncir::DataEdge>(node.getInEdge(0));

    // Get input blob
    int input_tensor_blob_id = input_tensor.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();
    auto output = nnrt::atenDim(tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, intToIValue(output));
}

void executorAtenDiv(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Div node";

    auto div_node = cast<nncir::AtenDivNode>(op_node);
    assert(div_node.getNumInputs() == 2);

    auto& input_self = cast<nncir::DataEdge>(div_node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(div_node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    auto dtype = stream_executor.findBlob(input_other_blob_id).first;
    if (dtype == DataType::TENSOR) {
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenDiv(self_tensor, other_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(div_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (isScalarType(dtype)) {
        assert(iv_other.isScalar());
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = nnrt::atenDiv(self_tensor, other_scalar);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(div_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::div";
    }
}

void executorAtenDropout(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Dropout node";

    auto dropout_node = cast<nncir::AtenDropoutNode>(op_node);
    assert(dropout_node.getNumInputs() == 1);
    int edge_id = 0;
    auto& input_tensor = cast<nncir::DataEdge>(dropout_node.getInEdge(0));
    // Get input blob
    int input_tensor_blob_id = input_tensor.getBlobId();
    // Find the input blob
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();
    edge_id++;

    double proportion = (double)dropout_node.getProportion();
    if (nncir::isDefaultValue(proportion)) {
        auto& proportion_edge = cast<nncir::DataEdge>(dropout_node.getInEdge(edge_id++));
        int proportion_blob_id = proportion_edge.getBlobId();
        auto proportion_iv = stream_executor.findBlob(proportion_blob_id).second;
        assert(proportion_iv.isDouble());
        proportion = proportion_iv.toDouble();
    }
    int train_val = dropout_node.getTrain();
    if (nncir::isDefaultValue(train_val)) {
        auto& train_edge = cast<nncir::DataEdge>(dropout_node.getInEdge(edge_id++));
        int train_blob_id = train_edge.getBlobId();
        auto train_iv = stream_executor.findBlob(train_blob_id).second;
        assert(train_iv.isBool());
        train_val = train_iv.toBool();
    }
    bool train = static_cast<bool>(train_val);

    at::Tensor output = nnrt::atenDropout(tensor, proportion, train);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(dropout_node.getFirstOutEdge());

    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenEmbedding(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Embedding node";

    auto node = cast<nncir::AtenEmbeddingNode>(op_node);
    int edge_id = 0;

    auto& input_weights = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_indices = cast<nncir::DataEdge>(node.getInEdge(1));

    // Get input blob
    int input_weights_blob_id = input_weights.getBlobId();
    int input_indices_blob_id = input_indices.getBlobId();
    // Find the input blob
    torch::jit::IValue iv_weights = stream_executor.findBlob(input_weights_blob_id).second;
    torch::jit::IValue iv_indices = stream_executor.findBlob(input_indices_blob_id).second;
    assert(iv_weights.isTensor());
    assert(iv_indices.isTensor());
    edge_id += 2;

    int64_t padding_idx = node.getPaddingIdx();
    if (nncir::isDefaultValue(padding_idx)) {
        auto& padding_idx_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int padding_idx_blob_id = padding_idx_edge.getBlobId();
        auto padding_idx_iv = stream_executor.findBlob(padding_idx_blob_id).second;
        assert(padding_idx_iv.isInt());
        padding_idx = padding_idx_iv.toInt();
    }

    int scale_grad_by_freq_val = node.getScaleGradByFreq();
    if (nncir::isDefaultValue(scale_grad_by_freq_val)) {
        auto& scale_grad_by_freq_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int scale_grad_by_freq_blob_id = scale_grad_by_freq_edge.getBlobId();
        auto scale_grad_by_freq_iv = stream_executor.findBlob(scale_grad_by_freq_blob_id).second;
        assert(scale_grad_by_freq_iv.isInt());
        scale_grad_by_freq_val = scale_grad_by_freq_iv.toInt();
    }
    bool scale_grad_by_freq = static_cast<bool>(scale_grad_by_freq_val);

    int sparse_val = node.getSparse();
    if (nncir::isDefaultValue(sparse_val)) {
        auto& sparse_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int sparse_blob_id = sparse_edge.getBlobId();
        auto sparse_iv = stream_executor.findBlob(sparse_blob_id).second;
        assert(sparse_iv.isInt());
        sparse_val = sparse_iv.toInt();
    }
    bool sparse = static_cast<bool>(sparse_val);

    auto output =
        nnrt::atenEmbedding(iv_weights.toTensor(), iv_indices.toTensor(), padding_idx, scale_grad_by_freq, sparse);

    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenEq(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Eq node";

    auto eq_node = cast<nncir::AtenEqNode>(op_node);
    assert(eq_node.getNumInputs() == 2);

    auto& input_self = cast<nncir::DataEdge>(eq_node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(eq_node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;

    if (iv_self.isTensor()) {
        at::Tensor self_tensor = iv_self.toTensor();
        at::Tensor output;
        if (iv_other.isTensor()) {
            at::Tensor other_tensor = iv_other.toTensor();
            output = nnrt::atenEq(self_tensor, other_tensor);
        } else if (iv_other.isScalar()) {
            at::Scalar other = iv_other.toScalar();
            output = nnrt::atenEq(self_tensor, other);
        } else {
            DLOG(FATAL) << "Aten eq op's data type do not support!";
        }
        // update output
        auto& out_edge = cast<nncir::DataEdge>(eq_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isScalar()) {
        assert(iv_other.isScalar());
        at::Scalar self_scalar = iv_self.toScalar();
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = nnrt::atenEq(self_scalar, other_scalar);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(eq_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::BOOL, scalarToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::eq";
    }
}

void executorAtenEqual(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Equal node";

    auto node = cast<nncir::AtenEqualNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    assert(iv_self.isTensor() && iv_other.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    at::Tensor other_tensor = iv_other.toTensor();

    auto output = nnrt::atenEqual(self_tensor, other_tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::BOOL, boolToIValue(output));
}

void executorAtenExpand(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Expand node";

    auto expand_node = cast<nncir::AtenExpandNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(expand_node.getInEdge(0));
    auto& input_size = cast<nncir::DataEdge>(expand_node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_size_blob_id = input_size.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_size = stream_executor.findBlob(input_size_blob_id).second;
    assert(iv_self.isTensor());
    assert(iv_size.isList());

    auto self_tensor = iv_self.toTensor();
    auto size_ivalue_list = iv_size.toListRef();
    auto size_list = parseIValueVector<int64_t>(size_ivalue_list);

    int implicit = expand_node.getImplicit();
    if (nncir::isDefaultValue(implicit)) {
        auto& implicit_edge = cast<nncir::DataEdge>(expand_node.getInEdge(2));
        int implicit_blob_id = implicit_edge.getBlobId();
        auto implicit_iv = stream_executor.findBlob(implicit_blob_id).second;
        assert(implicit_iv.isInt());
        implicit = implicit_iv.toInt();
    }

    auto output = nnrt::atenExpand(self_tensor, at::ArrayRef<int64_t>(size_list), static_cast<bool>(implicit));
    // update output
    auto& out_edge = cast<nncir::DataEdge>(expand_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenFill(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Fill node";

    auto node = cast<nncir::AtenEqNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    if (iv_other.isTensor()) {
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenFill(self_tensor, other_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (iv_other.isScalar()) {
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = nnrt::atenFill(self_tensor, other_scalar);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::fill_";
    }
}

void executorAtenFloorDivide(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten FloorDivide node";

    auto node = cast<nncir::AtenFloorDivideNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    if (iv_other.isTensor()) {
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenFloorDivide(self_tensor, other_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (iv_other.isScalar()) {
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = nnrt::atenFloorDivide(self_tensor, other_scalar);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::floor_divide";
    }
}

void executorAtenFormat(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Format node";

    auto format_node = cast<nncir::AtenFormatNode>(op_node);
    int edge_id = 0;
    auto assembly_format = format_node.getAssemblyFormat();

    if (assembly_format == "") {
        auto& assembly_format_edge = cast<nncir::DataEdge>(format_node.getInEdge(edge_id++));
        int assembly_format_blob_id = assembly_format_edge.getBlobId();
        auto assembly_format_iv = stream_executor.findBlob(assembly_format_blob_id).second;
        assert(assembly_format_iv.isString());
        assembly_format = assembly_format_iv.toStringRef();
    }

    auto& input1 = cast<nncir::DataEdge>(format_node.getInEdge(edge_id++));
    auto& input2 = cast<nncir::DataEdge>(format_node.getInEdge(edge_id++));

    // Get input blob
    int input1_blob_id = input1.getBlobId();
    int input2_blob_id = input2.getBlobId();

    // Find the input blob
    auto i_value1 = stream_executor.findBlob(input1_blob_id).second;
    auto i_value2 = stream_executor.findBlob(input2_blob_id).second;

    auto dtype = stream_executor.findBlob(input1_blob_id).first;
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

        auto output = nnrt::atenFormat(assembly_format, str1, str2);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(format_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::STRING, strToIValue(output));
    } else if (dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
               dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::INT64) {
        // aten::format(string, int, int)
        std::string str1 = std::to_string(i_value1.toInt());
        std::string str2 = std::to_string(i_value2.toInt());

        auto output = nnrt::atenFormat(assembly_format, str1, str2);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(format_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::STRING, strToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::format";
    }
}

void executorAtenGather(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Gather node";

    auto node = cast<nncir::AtenGatherNode>(op_node);
    int edge_id = 0;

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();

    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    auto dim = node.getDim();
    if (nncir::isDefaultValue(dim)) {
        auto& dim_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int dim_blob_id = dim_edge.getBlobId();
        auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }

    auto& input_index = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    int input_index_blob_id = input_index.getBlobId();

    torch::jit::IValue iv_index = stream_executor.findBlob(input_index_blob_id).second;
    assert(iv_index.isTensor());
    auto index_tensor = iv_index.toTensor();

    auto sparse_grad = node.getSparseGrad();
    if (nncir::isDefaultValue(sparse_grad)) {
        auto& sparse_grad_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int sparse_grad_blob_id = sparse_grad_edge.getBlobId();
        auto sparse_grad_iv = stream_executor.findBlob(sparse_grad_blob_id).second;
        assert(sparse_grad_iv.isInt());
        sparse_grad = static_cast<int>(sparse_grad_iv.toInt());
    }

    auto output = nnrt::atenGather(self_tensor, dim, index_tensor, sparse_grad);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenGe(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Ge node";

    auto node = cast<nncir::AtenGeNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    if (iv_other.isTensor()) {
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenGe(self_tensor, other_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (iv_other.isScalar()) {
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = nnrt::atenGe(self_tensor, other_scalar);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::ge";
    }
}

void executorAtenGetItem(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten GetItem node";

    auto get_item_node = cast<nncir::AtenGetItemNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(get_item_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();

    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isList());
    auto self_list = iv_self.toList();

    int idx = get_item_node.getIdx();
    if (nncir::isDefaultValue(idx)) {
        auto& idx_edge = cast<nncir::DataEdge>(get_item_node.getInEdge(1));
        int idx_blob_id = idx_edge.getBlobId();
        auto idx_iv = stream_executor.findBlob(idx_blob_id).second;
        assert(idx_iv.isInt());
        idx = idx_iv.toInt();
    }

    auto output = nnrt::atenGetItem(self_list, idx);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(get_item_node.getFirstOutEdge());
    stream_executor.releation_blob_ids_map_.insert({out_edge.getBlobId(),{input_self_blob_id, idx}});
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::IVALUE, output);
}

void executorAtenGt(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Gt node";
    auto node = cast<nncir::AtenGtNode>(op_node);

    auto in_blob_ids = getInBlobIds(op_node);
    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_blob_ids.at(0)).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_blob_ids.at(1)).second;

    // Find output blob
    auto out_blob_ids = getUniqueOutBlobIds(op_node);
    assert(out_blob_ids.size() == 1);

    if (iv_self.isTensor() && iv_other.isTensor()) {
        // tensor = Gt(tensor, tensor)
        auto output = nnrt::atenGt(iv_self.toTensor(), iv_other.toTensor());
        stream_executor.updateBlob(out_blob_ids.at(0), DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isTensor() && iv_other.isScalar()) {
        // tensor = Gt(tensor, scalar)
        auto output = nnrt::atenGt(iv_self.toTensor(), iv_other.toScalar());
        stream_executor.updateBlob(out_blob_ids.at(0), DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isScalar() && iv_other.isInt()) {
        // int/bool Gt(scalar, int)
        int64_t output = iv_self.toScalar().toInt() > iv_other.toInt();
        stream_executor.updateBlob(out_blob_ids.at(0), DataType::BOOL, scalarToIValue<int64_t>(output));
    } else if (iv_self.isInt() && iv_other.isInt()) {
        // int/bool = Gt(int, int)
        int64_t output = iv_self.toInt() > iv_other.toInt();
        stream_executor.updateBlob(out_blob_ids.at(0), DataType::BOOL, scalarToIValue<int64_t>(output));
    }
}

void executorAtenIndex(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Index node";

    auto node = cast<nncir::AtenIndexNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();

    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    auto& indices_edge = cast<nncir::DataEdge>(node.getInEdge(1));
    int indices_blob_id = indices_edge.getBlobId();
    auto indices_iv = stream_executor.findBlob(indices_blob_id).second;
    assert(indices_iv.isTensorList());
    auto indices_list_tensor = indices_iv.toTensorList();
    std::vector<at::Tensor> indices_list_tensor_vector;
    for (auto tensor : indices_list_tensor) {
        indices_list_tensor_vector.push_back(tensor);
    }
    at::TensorList indices(indices_list_tensor_vector);

    auto output = nnrt::atenIndex(self_tensor, indices);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenIndexPut(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten IndexPut node";

    auto node = cast<nncir::AtenIndexPutNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    auto& indices_edge = cast<nncir::DataEdge>(node.getInEdge(1));
    int indices_blob_id = indices_edge.getBlobId();
    auto indices_iv = stream_executor.findBlob(indices_blob_id).second;
    assert(indices_iv.isTensorList());
    auto indices_list_tensor = indices_iv.toTensorList();
    std::vector<at::Tensor> indices_list_tensor_vector;
    for (auto tensor : indices_list_tensor) {
        indices_list_tensor_vector.push_back(tensor);
    }
    at::TensorList indices(indices_list_tensor_vector);

    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(2));
    int input_other_blob_id = input_other.getBlobId();
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    assert(iv_other.isTensor());
    auto value_tensor = iv_other.toTensor();

    auto accumulate = node.getAccumulate();
    if (nncir::isDefaultValue(accumulate)) {
        auto& accumulate_edge = cast<nncir::DataEdge>(node.getInEdge(3));
        int accumulate_blob_id = accumulate_edge.getBlobId();
        auto accumulate_iv = stream_executor.findBlob(accumulate_blob_id).second;
        assert(accumulate_iv.isInt());
        accumulate = accumulate_iv.toInt();
    }

    auto output = nnrt::atenIndexPut(self_tensor, indices, value_tensor, static_cast<bool>(accumulate));
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenIndexSelect(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten IndexSelect node";

    auto node = cast<nncir::AtenIndexSelectNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();

    auto dim = node.getDim();
    if (nncir::isDefaultValue(dim)) {
        auto& dim_edge = cast<nncir::DataEdge>(node.getInEdge(1));
        int dim_blob_id = dim_edge.getBlobId();
        auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }

    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(2));
    int input_other_blob_id = input_other.getBlobId();
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    assert(iv_other.isTensor());
    auto index_tensor = iv_other.toTensor();

    auto output = nnrt::atenIndexSelect(self_tensor, dim, index_tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenInt(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Int node";

    auto int_node = cast<nncir::AtenIntNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(int_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();

    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    int64_t output = -1;
    if (iv_self.isScalar()) {
        auto self_scalar = iv_self.toScalar();
        output = nnrt::atenInt(self_scalar);
    } else if (iv_self.isTensor()) {
        auto self_tensor = iv_self.toTensor();
        output = nnrt::atenInt(self_tensor);
    } else {
        DLOG(ERROR) << "AtenInt data type do not support!";
    }

    // update output
    auto& out_edge = cast<nncir::DataEdge>(int_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, scalarToIValue(output));
}

void executorAtenIs(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Is node";

    auto node = cast<nncir::AtenIsNode>(op_node);
    assert(node.getNumInputs() == 2);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;

    auto output = nnrt::atenIs(iv_self, iv_other);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::BOOL, boolToIValue(output));
}

void executorAtenItem(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Item node";

    auto node = cast<nncir::AtenItemNode>(op_node);
    assert(node.getNumInputs() == 1);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    c10::Scalar output = nnrt::atenItem(self_tensor);
    auto output_dtype = convertATScalarTypeToDType(output.type());

    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), output_dtype, torch::jit::IValue(output));
}

void executorAtenLeakyRelu(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten LeakyRelu node";

    auto node = cast<nncir::AtenLeakyReluNode>(op_node);

    auto& input_edge = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_blob_id = input_edge.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_blob_id).second;
    assert(iv_tensor.isTensor());
    auto tensor = iv_tensor.toTensor();

    auto scalar = node.getScalar();
    if (nncir::isDefaultValue(scalar)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(1));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isDouble());
        scalar = data_iv.toDouble();
    }

    auto output = nnrt::atenLeakyRelu(tensor, scalar);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenLen(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Len node";

    auto node = cast<nncir::AtenLenNode>(op_node);
    assert(node.getNumInputs() == 1);

    auto& input_list = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_list_blob_id = input_list.getBlobId();

    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(input_list_blob_id).second;

    int64_t output = -1;
    if (iv.isList()) {
        output = nnrt::atenLen(iv.toList());
    } else if (iv.isTensor()) {
        output = nnrt::atenLen(iv.toTensor());
    } else {
        DLOG(FATAL) << "Aten len op's data type do not support!";
    }
 
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, intToIValue(output));
}

void executorAtenLinear(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Linear node";

    auto node = cast<nncir::AtenLinearNode>(op_node);

    auto& input_edge = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_blob_id = input_edge.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_blob_id).second;
    assert(iv_tensor.isTensor());
    auto tensor = iv_tensor.toTensor();

    auto weight_blob_id = (node.getWeightBlobIds())[0];
    auto bias_blob_id = (node.getBiasBlobIds())[0];
    std::vector<at::Tensor> weights;
    auto weight_iv = stream_executor.findBlob(weight_blob_id).second;
    auto bias_iv = stream_executor.findBlob(weight_blob_id).second;
    assert(weight_iv.isTensor() && bias_iv.isTensor());
    auto weight_tensor = weight_iv.toTensor();
    auto bias_tensor = bias_iv.toTensor();

    auto output = nnrt::atenLinear(tensor, weight_tensor, bias_tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenList(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten List node";

    auto node = cast<nncir::AtenListNode>(op_node);
    assert(node.getNumInputs() == 1);

    auto& input_list = cast<nncir::DataEdge>(node.getInEdge(0));
    // Get input blob
    int input_list_blob_id = input_list.getBlobId();
    // Find the input blob
    torch::jit::IValue iv_list = stream_executor.findBlob(input_list_blob_id).second;
    assert(iv_list.isList());
    auto output = nnrt::atenList(iv_list.toList());
    // update iv_string
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::LIST, listToIValue(output));
}

void executorAtenLog(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Log node";

    auto node = cast<nncir::AtenLogNode>(op_node);

    auto& input_edge = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_blob_id = input_edge.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_blob_id).second;
    assert(iv_tensor.isTensor());
    auto tensor = iv_tensor.toTensor();

    auto output = nnrt::atenLog(tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenLogSoftmax(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten LogSoftmax node";

    auto node = cast<nncir::AtenLogSoftmaxNode>(op_node);
    int edge_id = 0;

    auto& input_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    int input_blob_id = input_edge.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_blob_id).second;
    assert(iv_tensor.isTensor());
    auto tensor = iv_tensor.toTensor();

    auto dim = node.getDim();
    if (nncir::isDefaultValue(dim)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        dim = data_iv.toInt();
    }

    auto ori_dtype = node.getDtype();
    bool dtype_is_none = false;
    if (nncir::isDefaultValue(ori_dtype)) {
        auto& ori_dtype_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int ori_dtype_blob_id = ori_dtype_edge.getBlobId();
        auto ori_dtype_iv = stream_executor.findBlob(ori_dtype_blob_id).second;
        if (ori_dtype_iv.isInt())
            ori_dtype = ori_dtype_iv.toInt();
        else if (ori_dtype_iv.isNone()) {
            dtype_is_none = true;
        }
    }

    torch::Tensor output;
    if (dtype_is_none) {
        output = nnrt::atenLogSoftmax(tensor, dim);
    } else {
        output = nnrt::atenLogSoftmax(tensor, dim, at::ScalarType(ori_dtype));
    }

    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));

}

void executorAtenLSTM1(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten LSTM1 node";

    auto lstm1_node = cast<nncir::AtenLSTM1Node>(op_node);
    int edge_idx = 0;

    // const at::Tensor &input
    auto& input_edge = cast<nncir::DataEdge>(lstm1_node.getInEdge(edge_idx));
    int input_blob_id = input_edge.getBlobId();
    auto input_iv = stream_executor.findBlob(input_blob_id).second;
    assert(input_iv.isTensor());
    auto input = input_iv.toTensor();
    edge_idx++;

    // at::TensorList hx
    auto& hx_edge = cast<nncir::DataEdge>(lstm1_node.getInEdge(edge_idx));
    int hx_blob_id = hx_edge.getBlobId();
    auto hx_iv = stream_executor.findBlob(hx_blob_id).second;
    assert(hx_iv.isTensorList());
    auto hx_list_tensor = hx_iv.toTensorList();
    std::vector<at::Tensor> hx_list_tensor_vector;
    for (auto tensor : hx_list_tensor) {
        hx_list_tensor_vector.push_back(tensor);
    }
    at::TensorList hx(hx_list_tensor_vector);
    edge_idx++;

    // Check and skip, will handle params after getting all arguments
    if (lstm1_node.getNumInputs() > edge_idx) {
        auto& params_edge = cast<nncir::DataEdge>(lstm1_node.getInEdge(edge_idx));
        int params_blob_id = params_edge.getBlobId();
        auto params_iv = stream_executor.findBlob(params_blob_id).second;
        if (params_iv.isTensorList()) {
            edge_idx++;
        }
    }

    // bool has_biases
    int has_biases = lstm1_node.getHasBiases();
    if (nncir::isDefaultValue(has_biases)) {
        auto& has_biases_edge = cast<nncir::DataEdge>(lstm1_node.getInEdge(edge_idx));
        int has_biases_blob_id = has_biases_edge.getBlobId();
        auto has_biases_iv = stream_executor.findBlob(has_biases_blob_id).second;
        assert(has_biases_iv.isInt());
        has_biases = has_biases_iv.toInt();
        edge_idx++;
    }

    // int64_t num_layers
    int64_t num_layers = lstm1_node.getNumLayers();
    if (nncir::isDefaultValue(num_layers)) {
        auto& num_layers_edge = cast<nncir::DataEdge>(lstm1_node.getInEdge(edge_idx));
        int num_layers_blob_id = num_layers_edge.getBlobId();
        auto num_layers_iv = stream_executor.findBlob(num_layers_blob_id).second;
        assert(num_layers_iv.isInt());
        num_layers = num_layers_iv.toInt();
        edge_idx++;
    }

    // double dropout
    double dropout = lstm1_node.getDropout();
    if (nncir::isDefaultValue(dropout)) {
        auto& dropout_edge = cast<nncir::DataEdge>(lstm1_node.getInEdge(edge_idx));
        int dropout_blob_id = dropout_edge.getBlobId();
        auto dropout_iv = stream_executor.findBlob(dropout_blob_id).second;
        assert(dropout_iv.isDouble());
        dropout = dropout_iv.toDouble();
        edge_idx++;
    }

    // bool train
    int train = lstm1_node.getTrain();
    if (nncir::isDefaultValue(train)) {
        auto& train_edge = cast<nncir::DataEdge>(lstm1_node.getInEdge(edge_idx));
        int train_blob_id = train_edge.getBlobId();
        auto train_iv = stream_executor.findBlob(train_blob_id).second;
        assert(train_iv.isInt());
        train = train_iv.toInt();
        edge_idx++;
    }

    // bool bidirectional
    int bidirectional = lstm1_node.getBidirectional();
    if (nncir::isDefaultValue(bidirectional)) {
        auto& bidirectional_edge = cast<nncir::DataEdge>(lstm1_node.getInEdge(edge_idx));
        int bidirectional_blob_id = bidirectional_edge.getBlobId();
        auto bidirectional_iv = stream_executor.findBlob(bidirectional_blob_id).second;
        assert(bidirectional_iv.isInt());
        bidirectional = bidirectional_iv.toInt();
        edge_idx++;
    }

    // bool batch_first
    int batch_first = lstm1_node.getBatchFirst();
    if (nncir::isDefaultValue(batch_first)) {
        auto& batch_first_edge = cast<nncir::DataEdge>(lstm1_node.getInEdge(edge_idx));
        int batch_first_blob_id = batch_first_edge.getBlobId();
        auto batch_first_iv = stream_executor.findBlob(batch_first_blob_id).second;
        assert(batch_first_iv.isInt());
        batch_first = batch_first_iv.toInt();
        edge_idx++;
    }

    // at::TensorList params
    // param layerout                --> (fw_w_ih, fw_w_hh, fw_b_ih?, fw_b_hh?) * layers
    // param layerout (bidirctional) --> (fw_w_ih, fw_w_hh, fw_b_ih?, fw_b_hh?, bw_w_ih, bw_w_hh, bw_b_ih?, bw_b_hh?) *
    // layers

    auto weight_blob_ids = lstm1_node.getWeightBlobId();
    auto bias_blob_ids = lstm1_node.getBiasBlobId();
    std::vector<at::Tensor> param_vector;
    assert((bidirectional == 0 || bidirectional == 1));
    for (int i = 0; i < num_layers * (bidirectional + 1); i++) {
        // w_ih
        auto w_ih_iv = stream_executor.findBlob(weight_blob_ids[i * 2]).second;
        if (w_ih_iv.isTensor()) {
            param_vector.push_back(w_ih_iv.toTensor());
        }
        // w_hh
        auto w_hh_iv = stream_executor.findBlob(weight_blob_ids[i * 2 + 1]).second;
        if (w_hh_iv.isTensor()) {
            param_vector.push_back(w_hh_iv.toTensor());
        }
        if (has_biases) {
            // b_ih? (optional)
            auto b_ih_iv = stream_executor.findBlob(bias_blob_ids[i * 2]).second;
            if (b_ih_iv.isTensor()) {
                param_vector.push_back(b_ih_iv.toTensor());
            }
            // b_hh? (optional)
            auto b_hh_iv = stream_executor.findBlob(bias_blob_ids[i * 2 + 1]).second;
            if (b_hh_iv.isTensor()) {
                param_vector.push_back(b_hh_iv.toTensor());
            }
        }
    }
    at::TensorList params(param_vector);
    auto output =
        nnrt::atenLstm1(input, hx, params, static_cast<bool>(has_biases), num_layers, dropout, static_cast<bool>(train),
                       static_cast<bool>(bidirectional), static_cast<bool>(batch_first));

    auto out_blob_ids = getOutBlobIds(op_node);
    auto pos = std::unique(out_blob_ids.begin(), out_blob_ids.end());
    out_blob_ids.erase(pos, out_blob_ids.end());
    assert(out_blob_ids.size() == 3);
    auto xxx = std::get<0>(output);
    stream_executor.updateBlob(out_blob_ids[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
    stream_executor.updateBlob(out_blob_ids[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
    stream_executor.updateBlob(out_blob_ids[2], DataType::TENSOR, tensorToIValue(std::get<2>(output)));

}

void executorAtenLSTM2(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten LSTM2 node";

    auto lstm2_node = cast<nncir::AtenLSTM2Node>(op_node);
    int edge_idx = 0;

    // const at::Tensor &input
    auto& input_edge = cast<nncir::DataEdge>(lstm2_node.getInEdge(edge_idx));
    int input_blob_id = input_edge.getBlobId();
    auto input_iv = stream_executor.findBlob(input_blob_id).second;
    assert(input_iv.isTensor());
    auto input = input_iv.toTensor();
    edge_idx++;

    // const at::Tensor &batch_sizes,
    at::Tensor batch_sizes;
    auto& batch_sizes_edge = cast<nncir::DataEdge>(lstm2_node.getInEdge(edge_idx));
    int batch_sizes_blob_id = batch_sizes_edge.getBlobId();
    auto batch_sizes_iv = stream_executor.findBlob(batch_sizes_blob_id).second;
    assert(batch_sizes_iv.isTensor());
    batch_sizes = batch_sizes_iv.toTensor();
    edge_idx++;

    // at::TensorList hx
    auto& hx_edge = cast<nncir::DataEdge>(lstm2_node.getInEdge(edge_idx));
    int hx_blob_id = hx_edge.getBlobId();
    auto hx_iv = stream_executor.findBlob(hx_blob_id).second;
    assert(hx_iv.isTensorList());
    auto hx_list_tensor = hx_iv.toTensorList();
    std::vector<at::Tensor> hx_list_tensor_vector;
    for (auto tensor : hx_list_tensor) {
        hx_list_tensor_vector.push_back(tensor);
    }
    at::TensorList hx(hx_list_tensor_vector);
    edge_idx++;

    // at::TensorList params
    // Check and skip, will handle params after getting all arguments
    if (lstm2_node.getNumInputs() > edge_idx) {
        auto& params_edge = cast<nncir::DataEdge>(lstm2_node.getInEdge(edge_idx));
        int params_blob_id = params_edge.getBlobId();
        auto params_iv = stream_executor.findBlob(params_blob_id).second;
        if (params_iv.isTensorList()) {
            edge_idx++;
        }
    }

    // bool has_biases
    int has_biases = lstm2_node.getHasBiases();
    if (nncir::isDefaultValue(has_biases)) {
        auto& has_biases_edge = cast<nncir::DataEdge>(lstm2_node.getInEdge(edge_idx));
        int has_biases_blob_id = has_biases_edge.getBlobId();
        auto has_biases_iv = stream_executor.findBlob(has_biases_blob_id).second;
        assert(has_biases_iv.isInt());
        has_biases = has_biases_iv.toInt();
        edge_idx++;
    }

    // int64_t num_layers
    int64_t num_layers = lstm2_node.getNumLayers();
    if (nncir::isDefaultValue(num_layers)) {
        auto& num_layers_edge = cast<nncir::DataEdge>(lstm2_node.getInEdge(edge_idx));
        int num_layers_blob_id = num_layers_edge.getBlobId();
        auto num_layers_iv = stream_executor.findBlob(num_layers_blob_id).second;
        assert(num_layers_iv.isInt());
        num_layers = num_layers_iv.toInt();
        edge_idx++;
    }

    // double dropout
    double dropout = lstm2_node.getDropout();
    if (nncir::isDefaultValue(dropout)) {
        auto& dropout_edge = cast<nncir::DataEdge>(lstm2_node.getInEdge(edge_idx));
        int dropout_blob_id = dropout_edge.getBlobId();
        auto dropout_iv = stream_executor.findBlob(dropout_blob_id).second;
        assert(dropout_iv.isDouble());
        dropout = dropout_iv.toDouble();
        edge_idx++;
    }

    // bool train
    int train = lstm2_node.getTrain();
    if (nncir::isDefaultValue(train)) {
        auto& train_edge = cast<nncir::DataEdge>(lstm2_node.getInEdge(edge_idx));
        int train_blob_id = train_edge.getBlobId();
        auto train_iv = stream_executor.findBlob(train_blob_id).second;
        assert(train_iv.isInt());
        train = train_iv.toInt();
        edge_idx++;
    }

    // bool bidirectional
    int bidirectional = lstm2_node.getBidirectional();
    if (nncir::isDefaultValue(bidirectional)) {
        auto& bidirectional_edge = cast<nncir::DataEdge>(lstm2_node.getInEdge(edge_idx));
        int bidirectional_blob_id = bidirectional_edge.getBlobId();
        auto bidirectional_iv = stream_executor.findBlob(bidirectional_blob_id).second;
        assert(bidirectional_iv.isInt());
        bidirectional = bidirectional_iv.toInt();
        edge_idx++;
    }

    // at::TensorList params
    // param layerout                --> (fw_w_ih, fw_w_hh, fw_b_ih?, fw_b_hh?) * layers
    // param layerout (bidirctional) --> (fw_w_ih, fw_w_hh, fw_b_ih?, fw_b_hh?, bw_w_ih, bw_w_hh, bw_b_ih?, bw_b_hh?) *
    // layers

    auto weight_blob_ids = lstm2_node.getWeightBlobId();
    auto bias_blob_ids = lstm2_node.getBiasBlobId();
    std::vector<at::Tensor> param_vector;
    assert((bidirectional == 0 || bidirectional == 1));
    for (int i = 0; i < num_layers * (bidirectional + 1); i++) {
        // w_ih
        auto w_ih_iv = stream_executor.findBlob(weight_blob_ids[i * 2]).second;
        if (w_ih_iv.isTensor()) {
            param_vector.push_back(w_ih_iv.toTensor());
        }
        // w_hh
        auto w_hh_iv = stream_executor.findBlob(weight_blob_ids[i * 2 + 1]).second;
        if (w_hh_iv.isTensor()) {
            param_vector.push_back(w_hh_iv.toTensor());
        }
        if (has_biases) {
            // b_ih? (optional)
            auto b_ih_iv = stream_executor.findBlob(bias_blob_ids[i * 2]).second;
            if (b_ih_iv.isTensor()) {
                param_vector.push_back(b_ih_iv.toTensor());
            }
            // b_hh? (optional)
            auto b_hh_iv = stream_executor.findBlob(bias_blob_ids[i * 2 + 1]).second;
            if (b_hh_iv.isTensor()) {
                param_vector.push_back(b_hh_iv.toTensor());
            }
        }
    }
    at::TensorList params(param_vector);

    auto output = nnrt::atenLstm2(input, batch_sizes, hx, params, static_cast<bool>(has_biases), num_layers, dropout,
                                static_cast<bool>(train), static_cast<bool>(bidirectional));

    auto out_blob_ids = getOutBlobIds(op_node);
    auto pos = std::unique(out_blob_ids.begin(), out_blob_ids.end());
    out_blob_ids.erase(pos, out_blob_ids.end());
    assert(out_blob_ids.size() == 3);
    stream_executor.updateBlob(out_blob_ids[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
    stream_executor.updateBlob(out_blob_ids[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
    stream_executor.updateBlob(out_blob_ids[2], DataType::TENSOR, tensorToIValue(std::get<2>(output)));
}

void executorAtenLt(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Lt node";
    auto node = cast<nncir::AtenLtNode>(op_node);

    auto in_blob_ids = getInBlobIds(op_node);
    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(in_blob_ids.at(0)).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(in_blob_ids.at(1)).second;

    // Find output blob
    auto out_blob_ids = getUniqueOutBlobIds(op_node);
    assert(out_blob_ids.size() == 1);

    if (iv_self.isTensor() && iv_other.isTensor()) {
        // tensor = Lt(tensor, tensor)
        auto output = nnrt::atenLt(iv_self.toTensor(), iv_other.toTensor());
        stream_executor.updateBlob(out_blob_ids.at(0), DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isTensor() && iv_other.isScalar()) {
        // tensor = Lt(tensor, scalar)
        auto output = nnrt::atenLt(iv_self.toTensor(), iv_other.toScalar());
        stream_executor.updateBlob(out_blob_ids.at(0), DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isScalar() && iv_other.isInt()) {
        // int/bool Lt(scalar, int)
        int64_t output = iv_self.toScalar().toInt() < iv_other.toInt();
        stream_executor.updateBlob(out_blob_ids.at(0), DataType::BOOL, scalarToIValue<int64_t>(output));
    } else if (iv_self.isInt() && iv_other.isInt()) {
        // int/bool = Lt(int, int)
        int64_t output = iv_self.toInt() < iv_other.toInt();
        stream_executor.updateBlob(out_blob_ids.at(0), DataType::BOOL, scalarToIValue<int64_t>(output));
    }
}

void executorAtenMaskedFill(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten MaskedFill node";

    auto node = cast<nncir::AtenMaskedFillNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(1));
    auto& input_value = cast<nncir::DataEdge>(node.getInEdge(2));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();
    int input_value_blob_id = input_value.getBlobId();

    // Find the input blob
    auto iv_self = stream_executor.findBlob(input_self_blob_id).second;
    auto iv_other = stream_executor.findBlob(input_other_blob_id).second;
    auto iv_value = stream_executor.findBlob(input_value_blob_id).second;
    assert(iv_self.isTensor() && iv_other.isTensor());
    auto self_tensor = iv_self.toTensor();
    auto other_tensor = iv_other.toTensor();

    if (iv_value.isTensor()) {
        auto value_tensor = iv_value.toTensor();
        auto output = nnrt::atenMaskedFill(self_tensor, other_tensor, value_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (iv_value.isScalar()) {
        at::Scalar value_scalar = iv_value.toScalar();
        auto output = nnrt::atenMaskedFill(self_tensor, other_tensor, value_scalar);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::masked_fill";
    }
}

void executorAtenMatmul(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Matmul node";

    auto node = cast<nncir::AtenMatmulNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    assert(iv_self.isTensor());
    assert(iv_other.isTensor());

    auto output = nnrt::atenMatmul(iv_self.toTensor(), iv_other.toTensor());
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenMax(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Max node";

    auto max_node = cast<nncir::AtenMaxNode>(op_node);
    int edge_id = 0;

    // Get first input
    auto& input_self = cast<nncir::DataEdge>(max_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto dim = max_node.getDim();
    int keep_dim = max_node.getKeepDim();

    if (max_node.getInEdgeIds().size() == 1) {
        if (nncir::isDefaultValue(dim) && nncir::isDefaultValue(keep_dim)) {
            // aten::max(Tensor)
            auto output = nnrt::atenMax(self_tensor);
            // update output
            auto& out_edge = cast<nncir::DataEdge>(max_node.getFirstOutEdge());
            stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
        } else {
            // aten::max(Tensor, dim, keepdim)
            auto output = nnrt::atenMax(self_tensor, dim, static_cast<bool>(keep_dim));
            // update output
            auto& out_edge = cast<nncir::DataEdge>(max_node.getFirstOutEdge());
            auto out_blob_ids = getOutBlobIds(op_node);
            auto pos = std::unique(out_blob_ids.begin(), out_blob_ids.end());
            out_blob_ids.erase(pos, out_blob_ids.end());
            stream_executor.updateBlob(out_blob_ids[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
            stream_executor.updateBlob(out_blob_ids[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
        }
        return;
    }

    // Get second input
    auto& input_other = cast<nncir::DataEdge>(max_node.getInEdge(1));
    int input_other_blob_id = input_other.getBlobId();
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;

    auto dtype = stream_executor.findBlob(input_other_blob_id).first;
    if (dtype == DataType::TENSOR) {
        // aten::max(Tensor, Tensor)
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenMax(self_tensor, other_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(max_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (isScalarType(dtype)) {
        // aten::max(Tensor, dim, keepdim)
        edge_id++;
        auto dim = max_node.getDim();
        if (nncir::isDefaultValue(dim)) {
            auto& dim_edge = cast<nncir::DataEdge>(max_node.getInEdge(edge_id++));
            int dim_blob_id = dim_edge.getBlobId();
            auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
            assert(dim_iv.isInt());
            dim = dim_iv.toInt();
        }
        int keep_dim = max_node.getKeepDim();
        if (nncir::isDefaultValue(keep_dim)) {
            auto& keep_dim_edge = cast<nncir::DataEdge>(max_node.getInEdge(edge_id++));
            int keep_dim_blob_id = keep_dim_edge.getBlobId();
            auto keep_dim_iv = stream_executor.findBlob(keep_dim_blob_id).second;
            assert(keep_dim_iv.isBool());
            keep_dim = keep_dim_iv.toBool();
        }

        auto output = nnrt::atenMax(self_tensor, dim, static_cast<bool>(keep_dim));
        // update output
        auto& out_edge = cast<nncir::DataEdge>(max_node.getFirstOutEdge());
        auto out_blob_ids = getOutBlobIds(op_node);
        auto pos = std::unique(out_blob_ids.begin(), out_blob_ids.end());
        out_blob_ids.erase(pos, out_blob_ids.end());
        stream_executor.updateBlob(out_blob_ids[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
        stream_executor.updateBlob(out_blob_ids[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));

    } else {
        DLOG(ERROR) << "Unsupported input type for aten::max";
    }
}

void executorAtenMaxPool2d(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten MaxPool2d node";

    auto node = cast<nncir::AtenMaxPool2dNode>(op_node);
    int edge_id = 0;

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    int input_self_blob_id = input_self.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    auto self_tensor = iv_self.toTensor();
    edge_id++;

    auto kernel_size = node.getKernelSize();
    std::vector<int64_t> kernel_size_vec;
    if (kernel_size.h == INT64_MIN && kernel_size.w == INT64_MIN) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isList());
        auto data_list = data_iv.toListRef();
        kernel_size_vec = parseIValueVector<int64_t>(data_list);
    } else {
        kernel_size_vec.push_back(kernel_size.h);
        kernel_size_vec.push_back(kernel_size.w);
    }

    auto stride = node.getStride();
    std::vector<int64_t> stride_vec;
    if (stride.h == INT64_MIN && stride.w == INT64_MIN) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isList());
        auto data_list = data_iv.toListRef();
        stride_vec = parseIValueVector<int64_t>(data_list);
    } else {
        stride_vec.push_back(stride.h);
        stride_vec.push_back(stride.w);
    }

    // In PyTorch, Pad is a tuple(int, int)
    auto padding = node.getPad();
    std::vector<int64_t> padding_vec;
    if (padding.l == INT64_MIN && padding.r == INT64_MIN) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isList());
        auto data_list = data_iv.toListRef();
        padding_vec = parseIValueVector<int64_t>(data_list);
    } else {
        padding_vec.push_back(padding.l);
        padding_vec.push_back(padding.r);
    }

    auto dilation = node.getDilation();
    std::vector<int64_t> dilation_vec;
    if (dilation.h == INT64_MIN && dilation.w == INT64_MIN) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isList());
        auto data_list = data_iv.toListRef();
        dilation_vec = parseIValueVector<int64_t>(data_list);
    } else {
        dilation_vec.push_back(dilation.h);
        dilation_vec.push_back(dilation.w);
    }

    auto ceil_mode = node.getCeilMode();
    if (nncir::isDefaultValue(ceil_mode)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        ceil_mode = data_iv.toInt();
    }

    auto output = nnrt::atenMaxPool2d(self_tensor, at::ArrayRef<int64_t>(kernel_size_vec),
                                      at::ArrayRef<int64_t>(stride_vec), at::ArrayRef<int64_t>(padding_vec),
                                      at::ArrayRef<int64_t>(dilation_vec), static_cast<bool>(ceil_mode));
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenMin(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Min node";

    auto node = cast<nncir::AtenMinNode>(op_node);
    int edge_id = 0;

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    int input_self_blob_id = input_self.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto dim = node.getDimOrY();
    if (node.getNumInputs() == 1 && nncir::isDefaultValue(dim)) {
        auto output = nnrt::atenMin(self_tensor);
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
        return;
    }

    if (node.getNumInputs() == 2 && nncir::isDefaultValue(dim)) {
        auto& input_other = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int input_other_blob_id = input_other.getBlobId();
        torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenMin(self_tensor, other_tensor);
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
        return;
    }

    if (nncir::isDefaultValue(dim)) {
        auto& dim_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int dim_blob_id = dim_edge.getBlobId();
        auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }

    auto keepdim = node.getKeepDim();
    if (nncir::isDefaultValue(keepdim)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        keepdim = static_cast<int>(data_iv.toInt());
    }

    auto output = nnrt::atenMin(self_tensor, dim, static_cast<bool>(keepdim));
    // update output
    auto out_blob_ids = getOutBlobIds(op_node);
    auto pos = std::unique(out_blob_ids.begin(), out_blob_ids.end());
    out_blob_ids.erase(pos, out_blob_ids.end());
    stream_executor.updateBlob(out_blob_ids[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
    stream_executor.updateBlob(out_blob_ids[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
}

void executorAtenMul(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Mul node";

    auto node = cast<nncir::AtenMulNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;

    if (iv_self.isInt()) {
        assert(iv_other.isInt());
        auto self_int = iv_self.toInt();
        auto other_int = iv_other.toInt();
        auto output = nnrt::atenMul(self_int, other_int);
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, intToIValue(output));
    } else if (iv_self.isDouble()) {
        assert(iv_other.isDouble());
        auto self_double = iv_self.toDouble();
        auto other_double = iv_other.toDouble();
        auto output = nnrt::atenMul(self_double, other_double);
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::FLOAT64, doubleToIValue(output));
    } else if (iv_self.isTensor()) {
        at::Tensor self_tensor = iv_self.toTensor();
        if (iv_other.isTensor()) {
            at::Tensor other_tensor = iv_other.toTensor();
            auto output = nnrt::atenMul(self_tensor, other_tensor);
            auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
            stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
        } else if (iv_other.isScalar()) {
            at::Scalar other_scalar = iv_other.toScalar();
            auto output = nnrt::atenMul(self_tensor, other_scalar);
            auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
            stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
        } else {
            DLOG(ERROR) << "Unsupported input type for aten::mul";
        }
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::mul";
    }
}

void executorAtenNe(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Ne node";

    auto ne_node = cast<nncir::AtenNeNode>(op_node);
    assert(ne_node.getNumInputs() == 2);

    auto& input_self = cast<nncir::DataEdge>(ne_node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(ne_node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;

    if (iv_self.isIntList() && iv_other.isIntList()) {
        c10::List<int64_t> la = iv_self.toIntList();
        c10::List<int64_t> lb = iv_other.toIntList();
        auto output = nnrt::atenNe(la, lb);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(ne_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::BOOL, boolToIValue(output));
    } else if (iv_self.isTensor()) {
        assert(iv_other.isTensor());
        at::Tensor self_tensor = iv_self.toTensor();
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenNe(self_tensor, other_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(ne_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isScalar()) {
        assert(iv_other.isScalar());
        at::Scalar self_scalar = iv_self.toScalar();
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = nnrt::atenNe(self_scalar, other_scalar);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(ne_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::BOOL, boolToIValue(output));
    } else if (iv_self.isString()) {
        assert(iv_other.isString());
        auto self_scalar = iv_self.toString()->string();
        auto other_scalar = iv_other.toString()->string();
        auto output = nnrt::atenNe(self_scalar, other_scalar);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(ne_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::BOOL, boolToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::ne";
    }
}

void executorAtenNeg(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Neg node";
    auto neg_node = cast<nncir::AtenNegNode>(op_node);
    assert(neg_node.getNumInputs() == 1);

    auto& input_tensor = cast<nncir::DataEdge>(neg_node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv = stream_executor.findBlob(input_tensor_blob_id).second;
    auto dtype = stream_executor.findBlob(input_tensor_blob_id).first;
    auto& out_edge = cast<nncir::DataEdge>(neg_node.getFirstOutEdge());

    if (isScalarType(dtype)) {
        if (iv.isInt()) {
            int out = iv.toInt() * -1;
            stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, scalarToIValue<int>(out));
        } else if (iv.isDouble()) {
            double out = iv.toDouble() * -1;
            stream_executor.updateBlob(out_edge.getBlobId(), DataType::FLOAT64, scalarToIValue<double>(out));
        }
    } else if (iv.isTensor()) {
        // assert(iv.isTensor());
        at::Tensor tensor = iv.toTensor();
        auto output = nnrt::atenNeg(tensor);
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));

    } else {
        DLOG(ERROR) << "AtenNeg: unsupported dtype!";
    }
}

void executorAtenNot(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Not node";
    auto node = cast<nncir::AtenNotNode>(op_node);

    auto& input_tensor = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    auto iv = stream_executor.findBlob(input_tensor_blob_id).second;
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    if (iv.isTensor()) {
        auto tensor = iv.toTensor();
        auto output = nnrt::atenNot(tensor);
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (iv.isBool()) {
        auto input = iv.toBool();
        auto output = nnrt::atenNot(input);
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::BOOL, boolToIValue(output));
    } else {
        DLOG(FATAL) << "Aten not op's data type do not support!";
    }
}

void executorAtenOnes(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Ones node";
    auto node = cast<nncir::AtenOnesNode>(op_node);
    int edge_id = 0;

    auto& input_tensor = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    int input_tensor_blob_id = input_tensor.getBlobId();
    auto iv_self = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_self.isList());
    auto self_list = iv_self.toListRef();
    auto array_ref = parseIValueVector<int64_t>(self_list);

    at::TensorOptions options;
    auto dtype = node.getDtype();
    if (nncir::isDefaultValue(dtype)) {
        auto& edge_dtype = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto dtype_id = edge_dtype.getBlobId();
        auto iv_dtype = stream_executor.findBlob(dtype_id).second;
        if (!iv_dtype.isNone()) {
            options = options.dtype(iv_dtype.toScalarType());
        }
    } else {
        options = options.dtype(at::ScalarType(dtype));
    }

    auto layout = node.getLayout();
    if (nncir::isDefaultValue(layout)) {
        auto& edge_layout = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto layout_id = edge_layout.getBlobId();
        auto iv_layout = stream_executor.findBlob(layout_id).second;
        if (!iv_layout.isNone()) {
            options = options.layout(iv_layout.toLayout());
        }
    } else {
        options = options.layout(at::Layout(layout));
    }

    auto device = node.getDevice();
    if (nncir::isDefaultValue(device)) {
        auto& edge_device = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto device_id = edge_device.getBlobId();
        auto iv_device = stream_executor.findBlob(device_id).second;
        if (!iv_device.isNone()) {
            options = options.device(iv_device.toDevice());
        }
    } else {
        options = options.device(device);
    }

    auto pin_memory = node.getPinMemory();
    if (nncir::isDefaultValue(pin_memory)) {
        auto& edge_pin_memory = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        auto pin_memory_id = edge_pin_memory.getBlobId();
        auto iv_pin_memory = stream_executor.findBlob(pin_memory_id).second;
        if (!iv_pin_memory.isNone()) {
            options = options.pinned_memory(iv_pin_memory.toBool());
        }
    } else {
        options = options.pinned_memory(static_cast<bool>(pin_memory));
    }

    auto output = nnrt::atenOnes(at::ArrayRef<int64_t>(array_ref), options);
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenPackPaddedSequence(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten PackPaddedSequence node";

    auto node = cast<nncir::AtenPackPaddedSequenceNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    assert(iv_self.isTensor() && iv_other.isTensor());
    auto self_tensor = iv_self.toTensor();
    auto other_tensor = iv_other.toTensor();

    auto batch_first = node.getBatchFirst();
    if (nncir::isDefaultValue(batch_first)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(2));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        batch_first = static_cast<int>(data_iv.toInt());
    }

    auto output = nnrt::atenPackPaddedSequence(self_tensor, other_tensor, static_cast<bool>(batch_first));
    auto out_blob_ids = getOutBlobIds(op_node);
    auto pos = std::unique(out_blob_ids.begin(), out_blob_ids.end());
    out_blob_ids.erase(pos, out_blob_ids.end());
    stream_executor.updateBlob(out_blob_ids[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
    stream_executor.updateBlob(out_blob_ids[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
}

void executorAtenPadPackedSequence(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten PadPackedSequence node";

    auto node = cast<nncir::AtenPadPackedSequenceNode>(op_node);
    int edge_id = 0;

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(edge_id++));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    assert(iv_self.isTensor() && iv_other.isTensor());
    auto self_tensor = iv_self.toTensor();
    auto other_tensor = iv_other.toTensor();

    auto batch_first = node.getBatchFirst();
    if (nncir::isDefaultValue(batch_first)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        batch_first = static_cast<int>(data_iv.toInt());
    }

    auto padding_value = node.getPaddingValue();
    if (nncir::isDefaultValue(padding_value)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isDouble());
        padding_value = static_cast<float>(data_iv.toDouble());
    }

    auto total_length = node.getTotalLength();
    if (nncir::isDefaultValue(total_length)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        total_length = data_iv.toInt();
    }

    auto output = nnrt::atenPadPackedSequence(self_tensor, other_tensor, static_cast<bool>(batch_first), padding_value,
                                              total_length);
    auto out_blob_ids = getOutBlobIds(op_node);
    auto pos = std::unique(out_blob_ids.begin(), out_blob_ids.end());
    out_blob_ids.erase(pos, out_blob_ids.end());
    stream_executor.updateBlob(out_blob_ids[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
    stream_executor.updateBlob(out_blob_ids[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
}

void executorAtenPow(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Pow node";

    auto node = cast<nncir::AtenPowNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(node.getInEdge(1));

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;

    if (iv_self.isTensor() && iv_other.isTensor()) {
        auto self_tensor = iv_self.toTensor();
        auto other_tensor = iv_other.toTensor();
        auto output = nnrt::atenPow(self_tensor, other_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isTensor() && iv_other.isScalar()) {
        auto self_tensor = iv_self.toTensor();
        auto other_scalar = iv_other.toScalar();
        auto output = nnrt::atenPow(self_tensor, other_scalar);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (iv_self.isScalar() && iv_other.isTensor()) {
        auto self_scalar = iv_self.toScalar();
        auto other_tensor = iv_other.toTensor();
        auto output = nnrt::atenPow(self_scalar, other_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::pow";
    }
}

void executorAtenRelu(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Relu node";
    auto relu_node = cast<nncir::AtenReluNode>(op_node);
    assert(relu_node.getNumInputs() == 1);

    auto& input_tensor = cast<nncir::DataEdge>(relu_node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    auto output = nnrt::atenRelu(tensor);
    auto& out_edge = cast<nncir::DataEdge>(relu_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenSelect(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Select node";

    auto select_node = cast<nncir::AtenSelectNode>(op_node);
    int edge_id = 0;
    auto& input_self = cast<nncir::DataEdge>(select_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();

    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    edge_id++;

    auto dim = select_node.getDim();
    if (nncir::isDefaultValue(dim)) {
        auto& dim_edge = cast<nncir::DataEdge>(select_node.getInEdge(edge_id++));
        int dim_blob_id = dim_edge.getBlobId();
        auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }
    auto index = select_node.getIndex();
    if (nncir::isDefaultValue(index)) {
        auto& index_edge = cast<nncir::DataEdge>(select_node.getInEdge(edge_id++));
        int index_blob_id = index_edge.getBlobId();
        auto index_iv = stream_executor.findBlob(index_blob_id).second;
        assert(index_iv.isInt());
        index = index_iv.toInt();
    }

    auto output = nnrt::atenSelect(self_tensor, dim, index);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(select_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenSetItem(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten SetItem node";

    auto node = cast<nncir::AtenSetItemNode>(op_node);
    int edge_id = 0;
    auto& input_self = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    int input_self_blob_id = input_self.getBlobId();

    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isList());
    auto self_list = iv_self.toList();

    auto indice = node.getIndices();
    if (nncir::isDefaultValue(indice)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        indice = data_iv.toInt();
    }

    auto& input_item = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    int input_item_blob_id = input_item.getBlobId();
    torch::jit::IValue iv_item = stream_executor.findBlob(input_item_blob_id).second;

    auto output = nnrt::atenSetItem(self_list, indice, iv_item);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::LIST, listToIValue(output));
    stream_executor.updateBlob(input_self_blob_id, DataType::LIST, listToIValue(output));
}

void executorAtenSize(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Size node";

    auto size_node = cast<nncir::AtenSizeNode>(op_node);

    auto& input_tensor = cast<nncir::DataEdge>(size_node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    int inedges_cnt = size_node.getInEdgeIds().size();
    auto dim = size_node.getDim();
    if (inedges_cnt == 1 && nncir::isDefaultValue(dim)) {
        auto output = nnrt::atenSize(tensor);
        auto& out_edge = cast<nncir::DataEdge>(size_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::LIST, torch::jit::IValue(output));
    } else {
        if (nncir::isDefaultValue(dim)) {
            auto& dim_edge = cast<nncir::DataEdge>(size_node.getInEdge(1));
            int dim_blob_id = dim_edge.getBlobId();
            auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
            assert(dim_iv.isInt());
            dim = dim_iv.toInt();
        }
        int64_t output = nnrt::atenSize(tensor, dim);
        auto& out_edge = cast<nncir::DataEdge>(size_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, intToIValue(output));
    }
}

void executorAtenSlice(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Slice node";

    auto slice_node = cast<nncir::AtenSliceNode>(op_node);
    int edge_id = 0;

    auto& input_tensor = cast<nncir::DataEdge>(slice_node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    edge_id++;

    auto input_cnts = slice_node.getNumInputs();

    auto dim = slice_node.getDim();
    if (nncir::isDefaultValue(dim)) {
        auto& dim_edge = cast<nncir::DataEdge>(slice_node.getInEdge(edge_id++));
        int dim_blob_id = dim_edge.getBlobId();
        auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }
    auto start = slice_node.getStart();
    if (nncir::isDefaultValue(start)) {
        auto& start_edge = cast<nncir::DataEdge>(slice_node.getInEdge(edge_id++));
        int start_blob_id = start_edge.getBlobId();
        auto start_iv = stream_executor.findBlob(start_blob_id).second;
        assert(start_iv.isInt());
        start = start_iv.toInt();
    }
    auto end = slice_node.getEnd();
    if (nncir::isDefaultValue(end)) {
        auto& end_edge = cast<nncir::DataEdge>(slice_node.getInEdge(edge_id++));
        int end_blob_id = end_edge.getBlobId();
        auto end_iv = stream_executor.findBlob(end_blob_id).second;
        assert(end_iv.isInt());
        end = end_iv.toInt();
    }
    auto step = slice_node.getStep();
    if (nncir::isDefaultValue(step)) {
        auto& step_edge = cast<nncir::DataEdge>(slice_node.getInEdge(edge_id++));
        int step_blob_id = step_edge.getBlobId();
        auto step_iv = stream_executor.findBlob(step_blob_id).second;
        assert(step_iv.isInt());
        step = step_iv.toInt();
    }

    auto output = nnrt::atenSlice(iv_tensor.toTensor(), dim, start, end, step);
    auto& out_edge = cast<nncir::DataEdge>(slice_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenSoftmax(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Softmax node";

    auto node = cast<nncir::AtenSoftmaxNode>(op_node);
    int edge_id = 0;

    auto& input_tensor = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    auto self_tensor = iv_tensor.toTensor();

    auto dim = node.getDim();
    if (nncir::isDefaultValue(dim)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        dim = data_iv.toInt();
    }

    auto dtype = node.getDtype();
    at::Tensor output;
    if (nncir::isDefaultValue(dtype)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        if (!data_iv.isNone()) {
            dtype = data_iv.toInt();
            output = nnrt::atenSoftmax(self_tensor, dim, at::ScalarType(dtype));
        } else {
            output = nnrt::atenSoftmax(self_tensor, dim);
        }
    }

    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenSqueeze(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Squeeze node";

    auto node = cast<nncir::AtenSqueezeNode>(op_node);
    int edge_id = 0;

    auto& input_tensor = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    auto self_tensor = iv_tensor.toTensor();

    auto dim = node.getDim();
    if (nncir::isDefaultValue(dim)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        dim = data_iv.toInt();
    }

    auto output = nnrt::atenSqueeze(self_tensor, dim);
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenSub(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Sub node";

    auto sub_node = cast<nncir::AtenSubNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(sub_node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(sub_node.getInEdge(1));
    int64_t alpha = sub_node.getAlpha();
    if (nncir::isDefaultValue(alpha)) {
        auto& alpha_edge = cast<nncir::DataEdge>(sub_node.getInEdge(2));
        int alpha_blob_id = alpha_edge.getBlobId();
        auto alpha_iv = stream_executor.findBlob(alpha_blob_id).second;
        assert(alpha_iv.isInt());
        alpha = alpha_iv.toInt();
    }

    // Get input blob
    int input_self_blob_id = input_self.getBlobId();
    int input_other_blob_id = input_other.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_other = stream_executor.findBlob(input_other_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    auto dtype = stream_executor.findBlob(input_other_blob_id).first;
    if (dtype == DataType::TENSOR) {
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenSub(self_tensor, other_tensor, alpha);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(sub_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (isScalarType(dtype)) {
        assert(iv_other.isScalar());
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = nnrt::atenSub(self_tensor, other_scalar, alpha);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(sub_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::sub";
    }
}

void executorAtenSum(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Sum node";

    auto node = cast<nncir::AtenSumNode>(op_node);
    int edge_id = 0;

    auto& input_tensor = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    auto self_tensor = iv_tensor.toTensor();

    auto dims = node.getDim();
    if (dims.size() == 0) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isList());
        auto dims_list = data_iv.toListRef();
        dims = parseIValueVector<int64_t>(dims_list);
    }

    auto keepdim = node.getKeepdim();
    if (nncir::isDefaultValue(keepdim)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        keepdim = static_cast<int>(data_iv.toInt());
    }

    auto dtype = node.getDtype();
    if (nncir::isDefaultValue(dtype)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        dtype = data_iv.toInt();
    }

    auto output =
        nnrt::atenSum(self_tensor, at::ArrayRef<int64_t>(dims), static_cast<bool>(keepdim), at::ScalarType(dtype));
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenTanh(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Tanh node";

    auto node = cast<nncir::AtenTanhNode>(op_node);
    int edge_id = 0;

    auto& input_tensor = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    auto self_tensor = iv_tensor.toTensor();

    auto output = nnrt::atenTanh(self_tensor);
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenTensor(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Tensor node";

    auto tensor_node = cast<nncir::AtenTensorNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(tensor_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;

    at::TensorOptions options;
    auto& edge_dtype = cast<nncir::DataEdge>(tensor_node.getInEdge(1));
    auto& edge_device = cast<nncir::DataEdge>(tensor_node.getInEdge(2));
    // Fixme(SRCX): device may also be an input for aten::tensor
    // auto& edge_device     = cast<nncir::DataEdge>(tensor_node.getInEdge(3));
    auto& edge_pin_memory = cast<nncir::DataEdge>(tensor_node.getInEdge(3));
    auto dtype_id = edge_dtype.getBlobId();
    auto device_id = edge_device.getBlobId();
    auto pin_memory_id = edge_pin_memory.getBlobId();
    auto iv_dtype = stream_executor.findBlob(dtype_id).second;
    auto iv_device = stream_executor.findBlob(device_id).second;
    auto iv_pin_memory = stream_executor.findBlob(pin_memory_id).second;

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
            DLOG(ERROR) << "Unsupported data type to parse iv_pin_memory.";
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
        DLOG(ERROR) << "Unsupported data type to IValue.";
    }

    at::Tensor output;
    if (value_item.isInt()) {
        std::vector<int64_t> value_vec;
        std::vector<int64_t> dim = {1};
        parseIValueList<int64_t>(stream_executor.findBlob(input_self_blob_id).second, value_vec, dim, 1);
        output = nnrt::atenTensor(at::ArrayRef<int64_t>(value_vec), options).reshape(at::ArrayRef<int64_t>(dim));
    } else if (value_item.isDouble()) {
        std::vector<double> value_vec;
        std::vector<int64_t> dim = {1};
        parseIValueList<double>(stream_executor.findBlob(input_self_blob_id).second, value_vec, dim, 1);
        output = nnrt::atenTensor(at::ArrayRef<double>(value_vec), options).reshape(at::ArrayRef<int64_t>(dim));
    } else {
        DLOG(ERROR) << "Unsupported data type to parse IValue list.";
    }
    auto& out_edge = cast<nncir::DataEdge>(tensor_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenTo(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten To node";

    auto to_node = cast<nncir::AtenToNode>(op_node);
    int edge_id = 0;

    auto& input_self = cast<nncir::DataEdge>(to_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();

    auto data_type = stream_executor.findBlob(input_self_blob_id).first;
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    edge_id++;

    auto ori_dtype = to_node.getDType();
    if (nncir::isDefaultValue(ori_dtype)) {
        auto& ori_dtype_edge = cast<nncir::DataEdge>(to_node.getInEdge(edge_id++));
        int ori_dtype_blob_id = ori_dtype_edge.getBlobId();
        auto ori_dtype_iv = stream_executor.findBlob(ori_dtype_blob_id).second;
        assert(ori_dtype_iv.isInt());
        ori_dtype = ori_dtype_iv.toInt();
    }
    auto dtype = at::ScalarType(ori_dtype);

    int non_blocking_val = to_node.getNonBlocking();
    if (nncir::isDefaultValue(non_blocking_val)) {
        auto& non_blocking_edge = cast<nncir::DataEdge>(to_node.getInEdge(edge_id++));
        int non_blocking_blob_id = non_blocking_edge.getBlobId();
        auto non_blocking_iv = stream_executor.findBlob(non_blocking_blob_id).second;
        // assert(non_blocking_iv.isBool());
        non_blocking_val = non_blocking_iv.toInt();
    }
    bool non_blocking = static_cast<bool>(non_blocking_val);

    int copy_val = to_node.getCopy();
    if (nncir::isDefaultValue(copy_val)) {
        auto& copy_edge = cast<nncir::DataEdge>(to_node.getInEdge(edge_id++));
        int copy_blob_id = copy_edge.getBlobId();
        auto copy_iv = stream_executor.findBlob(copy_blob_id).second;
        assert(copy_iv.isBool());
        copy_val = copy_iv.toBool();
    }
    bool copy = static_cast<bool>(copy_val);

    auto optional_memory_format = to_node.getOptionalMemoryFormat();
    if (nncir::isDefaultValue(optional_memory_format)) {
        auto& optional_memory_format_edge = cast<nncir::DataEdge>(to_node.getInEdge(edge_id++));
        int optional_memory_format_blob_id = optional_memory_format_edge.getBlobId();
        auto optional_memory_format_iv = stream_executor.findBlob(optional_memory_format_blob_id).second;
        assert(optional_memory_format_iv.isInt());
        optional_memory_format = optional_memory_format_iv.toInt();
    }

    if (optional_memory_format == -1) {  // optional_memory_format = NONE
        auto output = nnrt::atenTo(self_tensor, dtype, non_blocking, copy);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(to_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else {
        auto memory_format = getMemoryFormat(optional_memory_format);
        auto output = nnrt::atenTo(self_tensor, dtype, non_blocking, copy, memory_format);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(to_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    }
}

void executorAtenTopk(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Topk node";

    auto node = cast<nncir::AtenTopkNode>(op_node);
    int edge_id = 0;

    auto& input_tensor = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    auto self_tensor = iv_tensor.toTensor();

    auto k = node.getK();
    if (nncir::isDefaultValue(k)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        k = data_iv.toInt();
    }

    auto dim = node.getDim();
    if (nncir::isDefaultValue(dim)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        dim = data_iv.toInt();
    }

    auto largest = node.getLargest();
    if (nncir::isDefaultValue(largest)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        largest = static_cast<int>(data_iv.toInt());
    }

    auto sorted = node.getSorted();
    if (nncir::isDefaultValue(sorted)) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int data_blob_id = data_edge.getBlobId();
        auto data_iv = stream_executor.findBlob(data_blob_id).second;
        assert(data_iv.isInt());
        sorted = static_cast<int>(data_iv.toInt());
    }

    auto output = nnrt::atenTopk(self_tensor, k, dim, static_cast<bool>(largest), static_cast<bool>(sorted));
    auto out_blob_ids = getOutBlobIds(op_node);
    auto pos = std::unique(out_blob_ids.begin(), out_blob_ids.end());
    out_blob_ids.erase(pos, out_blob_ids.end());
    stream_executor.updateBlob(out_blob_ids[0], DataType::TENSOR, tensorToIValue(std::get<0>(output)));
    stream_executor.updateBlob(out_blob_ids[1], DataType::TENSOR, tensorToIValue(std::get<1>(output)));
}

void executorAtenTranspose(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Transpose node";

    auto transpose_node = cast<nncir::AtenTransposeNode>(op_node);
    int edge_id = 0;

    auto& input_self = cast<nncir::DataEdge>(transpose_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    edge_id++;

    auto dim0 = transpose_node.getDim0();
    if (nncir::isDefaultValue(dim0)) {
        auto& dim0_edge = cast<nncir::DataEdge>(transpose_node.getInEdge(edge_id++));
        int dim0_blob_id = dim0_edge.getBlobId();
        auto dim0_iv = stream_executor.findBlob(dim0_blob_id).second;
        assert(dim0_iv.isInt());
        dim0 = dim0_iv.toInt();
    }
    auto dim1 = transpose_node.getDim1();
    if (nncir::isDefaultValue(dim1)) {
        auto& dim1_edge = cast<nncir::DataEdge>(transpose_node.getInEdge(edge_id++));
        int dim1_blob_id = dim1_edge.getBlobId();
        auto dim1_iv = stream_executor.findBlob(dim1_blob_id).second;
        assert(dim1_iv.isInt());
        dim1 = dim1_iv.toInt();
    }

    auto output = nnrt::atenTranspose(self_tensor, dim0, dim1);
    auto& out_edge = cast<nncir::DataEdge>(transpose_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenUnsqueeze(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Unsqueeze node";

    auto unsqueeze_node = cast<nncir::AtenUnsqueezeNode>(op_node);
    // assert(unsqueeze_node.getNumInputs() == 2);
    auto& input_tensor = cast<nncir::DataEdge>(unsqueeze_node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    auto dim = unsqueeze_node.getDim();
    if (nncir::isDefaultValue(dim)) {
        auto& dim0_edge = cast<nncir::DataEdge>(unsqueeze_node.getInEdge(1));
        int dim_blob_id = dim0_edge.getBlobId();
        auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }
    auto is_inplace = unsqueeze_node.getIsInplace();
    at::Tensor output = nnrt::atenUnsqueeze(tensor, dim);
    // If Unsqueeze op is in-place op, it need change origin data
    if (is_inplace) {
        auto releation_blob_id = stream_executor.releation_blob_ids_map_.find(input_tensor_blob_id);
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
        auto& out_edge = cast<nncir::DataEdge>(unsqueeze_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    }
}

void executorAtenView(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten View node";

    auto node = cast<nncir::AtenViewNode>(op_node);

    auto& input_tensor = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    auto& input_size = cast<nncir::DataEdge>(node.getInEdge(1));
    int input_size_blob_id = input_size.getBlobId();
    torch::jit::IValue iv_size = stream_executor.findBlob(input_size_blob_id).second;
    assert(iv_size.isList());
    auto size_list = iv_size.toListRef();
    auto size_array = parseIValueVector<int64_t>(size_list);

    at::Tensor output = nnrt::atenView(tensor, at::ArrayRef<int64_t>(size_array));
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenWarn(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute AtenWarn node";

    auto node = cast<nncir::AtenWarnNode>(op_node);
    auto& data_edge = cast<nncir::DataEdge>(node.getFirstInEdge());
    int input_blob_id = data_edge.getBlobId();
    torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;

    atenWarn(iv.toString()->string());
}

void executorAtenZeros(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Zeros node";

    auto zeros_node = cast<nncir::AtenZerosNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(zeros_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isList());

    auto self_list = iv_self.toListRef();
    // input list -> at::IntArrayRef size, so datatype of elements in list must be int.
    auto array_ref = parseIValueVector<int64_t>(self_list);

    at::TensorOptions options;
    auto& edge_dtype = cast<nncir::DataEdge>(zeros_node.getInEdge(1));
    auto& edge_layout = cast<nncir::DataEdge>(zeros_node.getInEdge(2));
    auto& edge_device = cast<nncir::DataEdge>(zeros_node.getInEdge(3));
    auto& edge_pin_memory = cast<nncir::DataEdge>(zeros_node.getInEdge(4));
    auto dtype_id = edge_dtype.getBlobId();
    auto layout_id = edge_layout.getBlobId();
    auto device_id = edge_device.getBlobId();
    auto pin_memory_id = edge_pin_memory.getBlobId();
    auto iv_dtype = stream_executor.findBlob(dtype_id).second;
    auto iv_layout = stream_executor.findBlob(layout_id).second;
    auto iv_device = stream_executor.findBlob(device_id).second;
    auto iv_pin_memory = stream_executor.findBlob(pin_memory_id).second;

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

    auto output = nnrt::atenZeros(at::ArrayRef<int64_t>(array_ref), options);
    auto& out_edge = cast<nncir::DataEdge>(zeros_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenZerosLike(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten ZerosLike node";
    auto zeros_like_node = cast<nncir::AtenZerosLikeNode>(op_node);
    assert(zeros_like_node.getNumInputs() == 6);

    auto& input_tensor = cast<nncir::DataEdge>(zeros_like_node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    at::Tensor tensor = iv_tensor.toTensor();

    at::TensorOptions options;
    auto& edge_dtype = cast<nncir::DataEdge>(zeros_like_node.getInEdge(1));
    auto& edge_layout = cast<nncir::DataEdge>(zeros_like_node.getInEdge(2));
    auto& edge_device = cast<nncir::DataEdge>(zeros_like_node.getInEdge(3));
    auto& edge_pin_memory = cast<nncir::DataEdge>(zeros_like_node.getInEdge(4));
    auto dtype_id = edge_dtype.getBlobId();
    auto layout_id = edge_layout.getBlobId();
    auto device_id = edge_device.getBlobId();
    auto pin_memory_id = edge_pin_memory.getBlobId();
    auto iv_dtype = stream_executor.findBlob(dtype_id).second;
    auto iv_layout = stream_executor.findBlob(layout_id).second;
    auto iv_device = stream_executor.findBlob(device_id).second;
    auto iv_pin_memory = stream_executor.findBlob(pin_memory_id).second;

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

    auto& input_memory_format = cast<nncir::DataEdge>(zeros_like_node.getInEdge(5));
    auto memory_format_id = input_memory_format.getBlobId();
    auto iv_memory_format = stream_executor.findBlob(memory_format_id).second;
    at::Tensor output;
    if (iv_memory_format.isNone()) {
        output = nnrt::atenZeroslike(tensor, options);
    } else {
        output = nnrt::atenZeroslike(tensor, options, iv_memory_format.toMemoryFormat());
    }

    // update output
    auto& out_edge = cast<nncir::DataEdge>(zeros_like_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenBatchNorm2d(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten BN2d node";
    auto bn_node = cast<nncir::AtenBatchNorm2dNode>(op_node);
    auto in_blob_ids = getInBlobIds(op_node);
    auto get_tensor = [&stream_executor](int id) {
        auto blob = stream_executor.findBlob(id);
        assert(blob.second.isTensor());
        return blob.second.toTensor();
    };
    at::Tensor input = get_tensor(in_blob_ids[0]);
    at::Tensor running_mean = get_tensor(in_blob_ids[1]);
    at::Tensor running_var = get_tensor(in_blob_ids[2]);

    auto weight_id = bn_node.getWeightBlobId();
    auto bias_id = bn_node.getBiasBlobId();
    assert(weight_id.size() == 1 && bias_id.size() == 1);
    at::Tensor weight = get_tensor(weight_id[0]);
    at::Tensor bias = get_tensor(bias_id[0]);

    // Get input attrs
    int training = bn_node.getTraining();
    double monentum = bn_node.getMomentum();
    double eps = bn_node.getEps();
    int cudnn_enabled = bn_node.getCudnnEnable();

    int offest = 3;
    if (nncir::isDefaultValue(training)) {
        auto iv = stream_executor.findBlob(in_blob_ids[offest++]).second;
        assert(iv.isInt());
        training = static_cast<int>(iv.toInt());
    }
    if (nncir::isDefaultValue(monentum)) {
        auto iv = stream_executor.findBlob(in_blob_ids[offest++]).second;
        assert(iv.isDouble());
        monentum = iv.toDouble();
    }
    if (nncir::isDefaultValue(eps)) {
        auto iv = stream_executor.findBlob(in_blob_ids[offest++]).second;
        assert(iv.isDouble());
        eps = iv.toDouble();
    }
    if (nncir::isDefaultValue(cudnn_enabled)) {
        auto iv = stream_executor.findBlob(in_blob_ids[offest++]).second;
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
    auto out_blob_id = getUniqueOutBlobIds(op_node)[0];
    stream_executor.updateBlob(out_blob_id, DataType::TENSOR, tensorToIValue(output));

}

void executorAtenReshape(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Reshape node";
    auto bn_node = cast<nncir::AtenReshapeNode>(op_node);
    auto in_blob_ids = getInBlobIds(op_node);
    auto get_tensor = [&stream_executor](int id) {
        auto blob = stream_executor.findBlob(id);
        assert(blob.second.isTensor());
        return blob.second.toTensor();
    };
    at::Tensor input_tensor = get_tensor(in_blob_ids[0]);

    // Get shape
    auto iv = stream_executor.findBlob(in_blob_ids[1]).second;
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
    auto out_blob_id = getUniqueOutBlobIds(op_node)[0];
    stream_executor.updateBlob(out_blob_id, DataType::TENSOR, tensorToIValue(output_tensor));
}

}  // namespace nnrt
