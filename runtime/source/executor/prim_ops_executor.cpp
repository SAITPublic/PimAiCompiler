#include "common/include/cast.hpp"
#include "executor/prim_ops.h"
#include "executor/prim_ops_executor.h"
#include "executor/prim_utils.h"
#include "executor/stream_executor.h"
#include "executor/utils.h"
#include "glog/logging.h"

#include "ir/include/control_nodes/prim_constant_node.hpp"
#include "ir/include/control_nodes/prim_data_node.hpp"
#include "ir/include/control_nodes/prim_device_node.hpp"
#include "ir/include/control_nodes/prim_dtype_node.hpp"
#include "ir/include/control_nodes/prim_end_loop_node.hpp"
#include "ir/include/control_nodes/prim_if_node.hpp"
#include "ir/include/control_nodes/prim_list_construct_node.hpp"
#include "ir/include/control_nodes/prim_tuple_construct_node.hpp"
#include "ir/include/control_nodes/prim_tuple_index_node.hpp"
#include "ir/include/control_nodes/prim_tuple_unpack_node.hpp"
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/nn_ir.hpp"

namespace nnrt {

void executePrimConstant(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "executePrimConstant";

    auto constant_node = cast<nncir::PrimConstantNode>(op_node);
    assert(constant_node.getNumInputs() == 0);  // Constant Op

    // Create IValue
    torch::jit::IValue iv;

    // Ntype is from GraphGen: Scalar(float, int32, int64...), String, Tensor, Device
    auto ntype = constant_node.getNtype();

    // dtype: {float, int32, int64, ... String, Tensor, Device}
    DataType dtype = DataType::NONE;

    // Get the data_ptr
    uint8_t* ptr = const_cast<uint8_t*>(constant_node.getData().data());

    // dont del these 2 line, otherwise the data will be error
    std::vector<uint8_t> tmp_data = constant_node.getData();
    int len = tmp_data.size();

    if (ntype == "str") {
        std::string str = primStrConstsnt((void*)ptr);
        iv = strToIValue(str);
        dtype = DataType::STRING;

    } else if (ntype == "Device") {
        std::string dev_str = primStrConstsnt((void*)ptr);
        iv = strToIValue(dev_str);
        dtype = DataType::DEVICE;

    } else if (ntype == "int" || ntype == "bool") {
        // set bool as int64
        int64_t tmp = 0;
        tmp = *(int64_t*)ptr;
        LOG(INFO) << "Set Data: " << tmp;
        iv = scalarToIValue(tmp);
        LOG(INFO) << "IV: " << iv;
        dtype = DataType::INT64;

    } else if (ntype == "float") {
        // float64
        iv = scalarToIValue<double>(*(double*)ptr);
        dtype = DataType::FLOAT64;

    } else if (ntype == "Tensor") {
        auto shape_ = constant_node.getShape();
        auto bit_width = constant_node.getBitWidth();
        auto scalar_type = DataType::NONE;

        if (bit_width == 16) {
            scalar_type = DataType::FLOAT16;
        } else if (bit_width == 32) {
            scalar_type = DataType::FLOAT32;
        } else {
            DLOG(ERROR) << "PrimConstant Error, unsupport data type when create Tensor!";
        }

        uint8_t* ptr = const_cast<uint8_t*>(constant_node.getData().data());
        std::vector<int64_t> input_shape = {static_cast<int64_t>(shape_.n), static_cast<int64_t>(shape_.c),
                                            static_cast<int64_t>(shape_.h), static_cast<int64_t>(shape_.w)};
        auto tensor = primTensorConstant((void*)ptr, input_shape, scalar_type);
        iv = tensorToIValue(tensor);
        dtype = DataType::TENSOR;

    } else {
        DLOG(ERROR) << "PrimConstant Error, unsupport data type!";
    }

    // save to global_blobs_ table
    // the Blob of all output edge is same in Contant Op
    auto& out_edge = cast<nncir::DataEdge>(constant_node.getOutEdge(0));
    DLOG(INFO) << "BlobId:" << out_edge.getBlobId();
    stream_executor.updateBlob(out_edge.getBlobId(), dtype, iv);
}

void executePrimDtype(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "executePrimDtype";

    // cast Node -> PrimDtypeNode
    auto dtype_node = cast<nncir::PrimDtypeNode>(op_node);
    assert(dtype_node.getNumInputs() == 1);

    // Find input edge, primDtype only have one input & output
    auto& data_edge = cast<nncir::DataEdge>(dtype_node.getFirstInEdge());
    // Get input blob
    int input_blob_id = data_edge.getBlobId();
    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;
    assert(iv.isTensor());
    torch::Tensor tensor = iv.toTensor();

    // Call OpKernel
    int64_t scalar_type = primDtype(tensor);

    // update output
    auto& out_edge = cast<nncir::DataEdge>(dtype_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, scalarToIValue(scalar_type));
}

void executePrimEndLoop(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "executePrimEndLoop";

    // cast Node -> PrimEndLoopNode
    auto end_loop_node = cast<nncir::PrimEndLoopNode>(op_node);
    auto inedges = end_loop_node.getInEdgeIds();
    auto outedges = end_loop_node.getOutEdgeIds();
    std::vector<torch::Tensor> input_tensor;
    for (uint32_t id = 0; id < inedges.size(); id++) {
        auto& in_edge = cast<nncir::DataEdge>(end_loop_node.getInEdge(inedges.at(id)));
        auto input_blob_id = in_edge.getBlobId();
        auto value_map = stream_executor.findBlob(input_blob_id);
        torch::jit::IValue iv = value_map.second;
        auto type = value_map.first;
        auto& out_edge = cast<nncir::DataEdge>(end_loop_node.getOutEdge(outedges.at(id)));
        stream_executor.updateBlob(out_edge.getBlobId(), type, iv);
    }
}

void executePrimData(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "executePrimData";

    // cast Node -> PrimDataNode
    auto data_node = cast<nncir::PrimDataNode>(op_node);
    auto& data_edge = cast<nncir::DataEdge>(data_node.getFirstInEdge());
    int input_blob_id = data_edge.getBlobId();
    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;
    assert(iv.isTensor());
    torch::Tensor tensor = iv.toTensor();

    // Call OpKernel
    torch::Tensor tensor_value = primData(tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(data_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(tensor_value));
}

void executePrimDevice(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "executePrimDevice";

    // cast Node -> PrimDeviceNode
    auto device_node = cast<nncir::PrimDeviceNode>(op_node);
    auto& data_edge = cast<nncir::DataEdge>(device_node.getFirstInEdge());
    int input_blob_id = data_edge.getBlobId();
    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;
    assert(iv.isTensor());
    torch::Tensor tensor = iv.toTensor();

    // Call OpKernel
    c10::Device device_value = primDevice(tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(device_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::STRING, strToIValue(device_value.str()));
}

void executePrimTupleConstruct(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "executePrimTupleConstruct";

    // cast Node -> PrimTupleConstructNode
    auto tuple_construct_node = cast<nncir::PrimTupleConstructNode>(op_node);
    auto inedges = tuple_construct_node.getInEdgeIds();

    std::vector<torch::IValue> inputs;
    for (auto edge_id : inedges) {
        auto& data_edge = cast<nncir::DataEdge>(tuple_construct_node.getInEdge(edge_id));
        int input_blob_id = data_edge.getBlobId();
        // Find the input blob
        torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;
        inputs.push_back(iv);
    }
    // Call OpKernel
    primTupleConstruct(inputs, inputs.size());

    // update output
    auto& out_edge = cast<nncir::DataEdge>(tuple_construct_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TUPLE, scalarToIValue(inputs.at(0)));
}

void executePrimTupleIndex(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "executePrimTupleIndex";

    // cast Node -> PrimTupleIndexNode
    auto tuple_index_node = cast<nncir::PrimTupleIndexNode>(op_node);
    auto inedges = tuple_index_node.getInEdgeIds();
    auto& data_edge = cast<nncir::DataEdge>(tuple_index_node.getInEdge(inedges.at(0)));
    int input_data_blob_id = data_edge.getBlobId();

    // Find the input data blob
    torch::jit::IValue data_iv = stream_executor.findBlob(input_data_blob_id).second;
    std::vector<torch::Tensor> inputs;
    auto tensors = data_iv.toTuple()->elements();
    for (auto tensor : tensors) {
        inputs.push_back(tensor.toTensor());
    }

    auto& index_edge = cast<nncir::DataEdge>(tuple_index_node.getInEdge(inedges.at(1)));
    int input_index_blob_id = index_edge.getBlobId();
    // Find the input index blob
    torch::jit::IValue index_iv = stream_executor.findBlob(input_index_blob_id).second;
    int64_t index = index_iv.toInt();
    // Call OpKernel
    auto output = primTupleIndex(inputs, index);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(tuple_index_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executePrimTupleUnpack(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "executePrimTupleUnpack";

    // cast Node -> PrimTupleUnpackNode
    auto tuple_unpack_node = cast<nncir::PrimTupleUnpackNode>(op_node);
    auto& data_edge = cast<nncir::DataEdge>(tuple_unpack_node.getFirstInEdge());
    int input_blob_id = data_edge.getBlobId();
    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;

    // Call OpKernel
    auto output = primTupleUnpack(iv.toTuple());
    // update output
    auto outedges = tuple_unpack_node.getOutEdgeIds();
    for (uint32_t idx = 0; idx < outedges.size(); idx++) {
        auto& out_edge = cast<nncir::DataEdge>(tuple_unpack_node.getOutEdge(outedges.at(idx)));
        auto type = inferDataType(output.at(idx));
        stream_executor.updateBlob(out_edge.getBlobId(), type, output.at(idx));
    }
}

void executePrimListConstruct(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "executePrimListConstruct";

    // cast Node -> PrimListConstructNode
    auto list_construct_node = cast<nncir::PrimListConstructNode>(op_node);
    auto inedges = list_construct_node.getInEdgeIds();

    std::vector<torch::IValue> inputs;
    for (auto edge_id : inedges) {
        auto& data_edge = cast<nncir::DataEdge>(list_construct_node.getInEdge(edge_id));
        int input_blob_id = data_edge.getBlobId();
        // Find the input blob
        torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;
        inputs.push_back(iv);
    }
    // Call OpKernel
    primListConstruct(inputs, inputs.size(), inferTypeFromDataType(type));

    // update output
    auto& out_edge = cast<nncir::DataEdge>(list_construct_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::LIST, scalarToIValue(inputs.at(0)));
}

}  // namespace nnrt
