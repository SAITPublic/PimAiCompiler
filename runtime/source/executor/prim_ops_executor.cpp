#include "executor/prim_ops_executor.h"
#include "common/include/cast.hpp"
#include "executor/prim_ops.h"
#include "executor/prim_utils.h"
#include "executor/stream_executor.h"
#include "executor/utils.h"
#include "glog/logging.h"
#include "ir/include/all_nodes.hpp"
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/nn_ir.hpp"

namespace nnrt
{
void executePrimConstant(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimConstant node";

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

void executePrimData(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimData node";

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

void executePrimDevice(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimDevice node";

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

void executePrimDtype(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimDtype node";

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




void executePrimListConstruct(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimListConstruct node";

    // cast Node -> PrimListConstructNode
    auto list_construct_node = cast<nncir::PrimListConstructNode>(op_node);
    auto inedges = list_construct_node.getInEdgeIds();

    std::vector<torch::IValue> inputs;
    DataType type = DataType::NONE;
    for (auto edge_id : inedges) {
        auto& data_edge = cast<nncir::DataEdge>(list_construct_node.getInEdge(edge_id));
        int input_blob_id = data_edge.getBlobId();
        // Find the input blob
        auto value_map = stream_executor.findBlob(input_blob_id);
        torch::jit::IValue iv = value_map.second;
        type = value_map.first;
        inputs.push_back(iv);
    }
    // Call OpKernel
    primListConstruct(inputs, inputs.size(), inferTypeFromDataType(type));

    // update output
    auto& out_edge = cast<nncir::DataEdge>(list_construct_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::LIST, scalarToIValue(inputs.at(0)));
}

void executePrimListUnpack(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimListUnpack node";

    // cast Node -> PrimListUnpackNode
    auto list_unpack_node = cast<nncir::PrimListUnpackNode>(op_node);
    auto& data_edge = cast<nncir::DataEdge>(list_unpack_node.getFirstInEdge());
    int input_blob_id = data_edge.getBlobId();
    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;
    std::vector<torch::IValue> inputs;
    for (auto item : iv.toTuple()->elements()) {
        inputs.push_back(item);
    }
    // Call OpKernel
    primListUnpack(inputs, inputs.size());
    // update output
    auto outedges = list_unpack_node.getOutEdgeIds();
    for (uint32_t idx = 0; idx < outedges.size(); idx++) {
        auto& out_edge = cast<nncir::DataEdge>(list_unpack_node.getOutEdge(outedges.at(idx)));
        auto type = inferDataType(inputs.at(idx));
        stream_executor.updateBlob(out_edge.getBlobId(), type, inputs.at(idx));
    }
}

void executePrimRaiseException(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimRaiseException node";

    // cast Node -> PrimRaiseExceptionNode
    auto raise_exception_node = cast<nncir::PrimRaiseExceptionNode>(op_node);
    auto& data_edge = cast<nncir::DataEdge>(raise_exception_node.getFirstInEdge());
    int input_blob_id = data_edge.getBlobId();
    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;
    // Call OpKernel
    primRaiseException(iv.toString()->string());
}

void executePrimTupleConstruct(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimTupleConstruct node";

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

void executePrimTupleIndex(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimTupleIndex node";

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

void executePrimTupleUnpack(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimTupleUnpack node";

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

void executePrimUncheckedCast(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimUncheckedCast node";

    // cast Node -> PrimUncheckedCastNode
    auto unchecked_cast_node = cast<nncir::PrimUncheckedCastNode>(op_node);
    auto& data_edge = cast<nncir::DataEdge>(unchecked_cast_node.getFirstInEdge());
    int input_blob_id = data_edge.getBlobId();
    auto map_value = stream_executor.findBlob(input_blob_id);
    auto type = map_value.first;
    // Find the input blob
    torch::jit::IValue iv = map_value.second;
    // Call OpKernel
    auto output = primUncheckedCast(iv);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(unchecked_cast_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), type, output);
}

void executePrimUninitialized(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimUninitialized node";

    // cast Node -> PrimUninitializedNode
    auto uninitialized_node = cast<nncir::PrimUninitializedNode>(op_node);
    // Call OpKernel
    auto output = primUninitialized();
    // update output
    auto& out_edge = cast<nncir::DataEdge>(uninitialized_node.getFirstOutEdge());
    auto type = inferDataType(output);
    stream_executor.updateBlob(out_edge.getBlobId(), type, output);
}

void executePrimVariable(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimVariable node";

    // cast Node -> PrimVariableNode
    auto variable_node = cast<nncir::PrimVariableNode>(op_node);
    auto data_arr = variable_node.getData();
    auto tensor_shape = variable_node.getShape();
    auto node_data_type = variable_node.getDataType();
    auto tensor_data_type = variable_node.getTensorDataType();
    uint8_t* ptr = const_cast<uint8_t*>(data_arr.data());

    // list[scalar,scalar] list[tensor,tensor]
    if (node_data_type.find("List") != std::string::npos) {
        int size = tensor_shape[0].n * tensor_shape[0].c * tensor_shape[0].h * tensor_shape[0].w;
        std::vector<torch::IValue> inputs;
        int total_size = 0;
        auto scalar_type = DataType::NONE;

        for (uint32_t idx = 0; idx < tensor_shape.size(); idx++) {
            std::vector<int64_t> input_shape = {
                static_cast<int64_t>(tensor_shape.at(idx).n), static_cast<int64_t>(tensor_shape.at(idx).c),
                static_cast<int64_t>(tensor_shape.at(idx).h), static_cast<int64_t>(tensor_shape.at(idx).w)};
            torch::jit::IValue iv;
            scalar_type = DataType::NONE;
            auto sizeofnum = 0;
            if (tensor_data_type.at(idx) == "int64") {
                scalar_type = DataType::INT64;
                sizeofnum = sizeof(int64_t);
                iv = scalarToIValue<int64_t>(*(int64_t*)(ptr + total_size * sizeofnum));
            } else if (tensor_data_type.at(idx) == "int32") {
                scalar_type = DataType::INT32;
                sizeofnum = sizeof(int32_t);
                iv = scalarToIValue<int32_t>(*(int32_t*)(ptr + total_size * sizeofnum));
            } else if (tensor_data_type.at(idx) == "int16") {
                scalar_type = DataType::INT16;
                sizeofnum = sizeof(int16_t);
                iv = scalarToIValue<int16_t>(*(int16_t*)(ptr + total_size * sizeofnum));
            } else if (tensor_data_type.at(idx) == "uint16") {
                scalar_type = DataType::UINT16;
                sizeofnum = sizeof(uint16_t);
                iv = scalarToIValue<uint16_t>(*(uint16_t*)(ptr + total_size * sizeofnum));
            } else if (tensor_data_type.at(idx) == "int8") {
                scalar_type = DataType::INT8;
                sizeofnum = sizeof(int8_t);
                iv = scalarToIValue<int8_t>(*(int8_t*)(ptr + total_size * sizeofnum));
            } else if (tensor_data_type.at(idx) == "uint8") {
                scalar_type = DataType::UINT8;
                sizeofnum = sizeof(uint8_t);
                iv = scalarToIValue<uint8_t>(*(uint8_t*)(ptr + total_size * sizeofnum));
            } else if (tensor_data_type.at(idx) == "float32") {
                scalar_type = DataType::FLOAT32;
                sizeofnum = sizeof(float);
                iv = scalarToIValue<float>(*(float*)(ptr + total_size * sizeofnum));
            } else if (tensor_data_type.at(idx) == "float64") {
                scalar_type = DataType::FLOAT64;
                sizeofnum = sizeof(float) * 2;
                iv = scalarToIValue<double>(*(double*)(ptr + total_size * sizeofnum));
            } else if (tensor_data_type.at(idx) == "bool") {
                scalar_type = DataType::BOOL;
                sizeofnum = sizeof(int64_t);
                iv = scalarToIValue<int64_t>(*(int64_t*)(ptr + total_size * sizeofnum));
            } else {
                DLOG(ERROR) << "Element type do not support! ";
            }
            // is tensor type
            if (size != 1 && (node_data_type.find("int") != std::string::npos ||
                              node_data_type.find("float") != std::string::npos)) {
                auto tensor = primTensorConstant((void*)(ptr + total_size * sizeofnum), input_shape, scalar_type);
                iv = tensorToIValue(tensor);
            }
            inputs.push_back(iv);
            total_size +=
                tensor_shape.at(idx).n * tensor_shape.at(idx).c * tensor_shape.at(idx).h * tensor_shape.at(idx).w;
        }

        primListConstruct(inputs, inputs.size(), inferTypeFromDataType(scalar_type));
        auto& out_edge = cast<nncir::DataEdge>(variable_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::LIST, inputs.at(0));
    }
}

void executePrimIf(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "executePrimIf";

    // cast Node
    auto if_node = cast_if<nncir::PrimIfNode>(op_node);
    assert(if_node->getNumInputs() == 1);

    DLOG(INFO) << "PrimIfNode.if" << if_node->getId();

    // Find input edge, primIf only have one input
    auto& data_edge = cast<nncir::DataEdge>(if_node->getFirstInEdge());
    // Get input blob
    int input_blob_id = data_edge.getBlobId();
    // Find the input blob, named condition, it is a int64/bool value
    torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;
    assert(iv.isInt());
    int64_t condiction = iv.toInt();
    assert(condiction == 0 || condiction == 1);

    int64_t next_node_id = -1;
    nncir::Node* next_node = nullptr;

    // ref: https://github.sec.samsung.net/PIM/NNCompiler/pull/74/files
    if (condiction == 1) {
        DLOG(INFO) << "PrimIf(True branch)";
        // choose then_net
        // next_node = if_node.id + 1
        next_node_id = if_node->getId() + 1;
    } else {
        DLOG(INFO) << "PrimIf(False branch)";
        // getElseNetStartNode
        next_node_id = if_node->getElseNetStartNode();
    }

    DLOG(INFO) << "PrimIf_next_node_id:" << next_node_id;
    stream_executor.setCursor(next_node_id);
}

void executePrimEndIf(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "executePrimEndIf";
    DLOG(INFO) << "Node_id:" << op_node.getId();

    // cast Node
    auto end_if_node = cast_if<nncir::PrimEndIfNode>(op_node);
    int if_id = end_if_node->getIfNodeId();

    DLOG(INFO) << "end_if_id: " << end_if_node->getId();
    DLOG(INFO) << "if_id: " << if_id;

    int64_t next_node_id = -1;
    nncir::Node* next_node = nullptr;

    // if --- endif -(else body)--- end
    // EndIfNode.kind == IF
    // need to skip else_body
    if (!end_if_node->getIsElseNet()) {
        next_node_id = end_if_node->getGotoNode();
        DLOG(INFO) << "EndIfNode.kind==IF";
    } else {
        DLOG(INFO) << "EndIfNode.kind==ELSE";

        // EndIfNode.kind == ELSE
        // next_node_id = end_if_node.getId() + 1;
        next_node_id = op_node.getId() + 1;
        DLOG(INFO) << "EndIfNode.next_node_id:" << next_node_id;

        // Only elseKind EndIf has inputs & outputs
        if (end_if_node->getNumInputs() > 0) {
            int if_id = end_if_node->getIfNodeId();
            nncir::Node* if_node = stream_executor.getGraph()->getNode(if_id);
            // get the condition: bool/int64

            // Find input edge, primIf only have one input
            auto& data_edge = cast<nncir::DataEdge>(if_node->getFirstInEdge());
            // Get input blob
            int input_blob_id = data_edge.getBlobId();
            // Find the input blob, named condition, it is a int64/bool value
            torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;
            assert(iv.isInt());
            int64_t condiction = iv.toInt();

            std::vector<int64_t> in_blob_ids;
            assert(end_if_node->getNumInputs() % 2 == 0);
            int num_input = end_if_node->getNumInputs();
            DLOG(INFO) << "Num Input: " << num_input;
            if (condiction == 1) {
                // Then Net ---> Endif
                // Get the TenNet's output
                for (int i = 0; i < end_if_node->getNumInputs() / 2; i++) {
                    int in_blob_id = cast<nncir::DataEdge>(end_if_node->getInEdge(i)).getBlobId();
                    in_blob_ids.push_back(in_blob_id);
                }
            } else {
                // Else Net --> Endif
                for (int i = end_if_node->getNumInputs() / 2; i < end_if_node->getNumInputs(); i++) {
                    int in_blob_id = cast<nncir::DataEdge>(end_if_node->getInEdge(i)).getBlobId();
                    in_blob_ids.push_back(in_blob_id);
                }
            }

            // Set inputs --> output
            // multi-inputs --> multi-outputs
            assert(in_blob_ids.size() == end_if_node->getOutEdgeIds().size());
            auto out_blob_ids = getOutBlobIds(op_node);
            for (int i = 0; i < in_blob_ids.size(); i++) {
                auto in_data = stream_executor.findBlob(in_blob_ids[i]);
                stream_executor.updateBlob(out_blob_ids[i], in_data.first, in_data.second);
            }
        }
    }

    // update the output Blobs
    stream_executor.setCursor(next_node_id);
}

void executePrimLoopIndex(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "executePrimLoopIndex";
    SHOW_OP_INFO(op_node);
    
    // cast Node
    auto loop_index_node = cast_if<nncir::PrimLoopIndexNode>(op_node);
    int64_t loop_index = loop_index_node->getIndex();
    if(loop_index < 0) {
        DLOG(INFO) <<"Invalid value for LoopIndex! set default loopIndex = 0!";
        loop_index = 0;
    }

    // LoopIndexNode only has one blob
    int out_blob_id = getOutBlobIds(*loop_index_node)[0];
    stream_executor.updateBlob(out_blob_id, DataType::INT64, scalarToIValue<int64_t>(loop_index));

}


std::unordered_map<std::string, int64_t> getMatchedLoopInfo(int64_t loop_block_id, StreamExecutor& stream_executor) {
    // Get LoopNode
    int64_t loop_id = loop_block_id - 2;
    assert(loop_id >=0 );
    auto graph = stream_executor.getGraph();
    auto loop_node = cast_if<nncir::PrimLoopNode>(*graph->getNode(loop_id));

    // max_cnt & cond
    int64_t max_trip_cnt = loop_node->getTripCount();
    int64_t cond = loop_node->getCond();
    
    // get LoopIndex
    int64_t loop_index_id = loop_block_id -1;
    int blob_id = getOutBlobIds(*graph->getNode(loop_index_id))[0];
    auto blob = stream_executor.findBlob(blob_id);
    assert(blob.first == DataType::INT64);
    int64_t loop_index = blob.second.toInt();

    // get loopEndNode.id + 1
    int end_loop_next_id = loop_node->getGotoNode();

    std::unordered_map<std::string, int64_t> umap;
    umap.insert({{"max_trip_cnt", max_trip_cnt}, {"cond", cond}, {"loop_index", loop_index}, {"end_loop_next_id", end_loop_next_id}});
    return umap;
}


void executePrimBlock(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "executePrimBlock";    

    // Prim Blcok only transfer input -- block --> outputs
    // Block( loopIndex: int,  x1, x2.....xr : IValue)
    auto block_node = cast_if<nncir::PrimBlockNode>(op_node);
    auto in_blob_ids = getInBlobIds(*block_node);
    auto out_blob_ids = getOutBlobIds(*block_node);

    auto umap = getMatchedLoopInfo(block_node->getId(), stream_executor);
    int64_t max_trip_cnt = umap["max_trip_cnt"];
    int64_t cond = umap["cond"];
    int64_t loop_index = umap["loop_index"];
    int64_t end_loop_next_id = umap["end_loop_next_id"];

    if(loop_index >= max_trip_cnt || cond == 0) {
        // Get EndLoop's input
        // transfer EndLoop's input --> output
        auto end_loop_node = stream_executor.getGraph()->getNode(end_loop_next_id - 1);
        auto end_loop_in_blobs = getInBlobIds(*end_loop_node);
        auto end_loop_out_blobs = getOutBlobIds(*end_loop_node);

        /**
         * %4 : bool = prim::Constant[value=1]()
         * block0(%i.1 : int, %y1.10 : Tensor):
         * %y1.2 : Tensor = aten::add(%y1.10, %x2.1, %3) 
         * %y1.5 : Tensor = aten::add(%y1.2, %i.1, %3) 
         * -> (%4, %y1.5) --> PrimEndLoop
         * 
         * Here is an example of PrimEndLoop, actually the output is %y1.5, exclude %4  in GraphIR
         * 
         */
    
        assert(end_loop_in_blobs.size() == end_loop_out_blobs.size() + 1);
        for(int i=0;i<end_loop_out_blobs.size();i++){
            auto in_blob = stream_executor.findBlob(end_loop_in_blobs.at(i + 1));
             stream_executor.updateBlob(end_loop_out_blobs.at(i), in_blob.first, in_blob.second);
        }

        // Jump to End Loop'next
        stream_executor.setCursor(end_loop_next_id);
    } else {
        // currently, only assume in_blob_ids.size == out_blob_ids.size
        assert(in_blob_ids.size() == out_blob_ids.size());
        for(int i = 0; i<in_blob_ids.size();i++) {
            auto in_blob = stream_executor.findBlob(in_blob_ids.at(i));
            stream_executor.updateBlob(out_blob_ids.at(i), in_blob.first, in_blob.second);
        }
        stream_executor.setCursor(block_node->getId() + 1);
    }
}

void executePrimLoop(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "executePrimLoop";
    SHOW_OP_INFO(op_node);

    auto loop_node = cast_if<nncir::PrimLoopNode>(op_node);
    int64_t loop_node_id = loop_node->getId();
    
    int64_t loop_cond = loop_node->getCond();
    int64_t max_trip_cnt = loop_node->getTripCount();

    // loop_node.id + 1 ---> loopIndex_node
    // loop_node.id + 2 ---> PrimBlockNode

    // Get LoopIndex
    auto loop_index_node = stream_executor.getGraph()->getNode(loop_node_id + 1);
    int64_t loop_index_blob_id = getOutBlobIds(*loop_index_node)[0];
    // 
    executePrimLoopIndex(*loop_index_node, stream_executor);   // BUG: this will reset loop_index

    DLOG(INFO) <<"loop_index_blob_id: " << loop_index_blob_id;
    int64_t loop_index = stream_executor.findBlob(getOutBlobIds(*loop_index_node)[0]).second.toInt();

    // Get LoopBlockNode.id
    int loop_block_id = loop_node_id + 2;
    auto loop_block_node = stream_executor.getGraph()->getNode(loop_block_id);

    // Passed the Input Tensors to PrimBlock
    auto in_blob_ids = getInBlobIds(op_node);
    auto out_blob_ids = getOutBlobIds(op_node);

    assert(in_blob_ids.size() == out_blob_ids.size() + 1);
    for(int i=0;i<out_blob_ids.size();i++) {
        int64_t in_id = in_blob_ids.at(i+1);
        int64_t out_id = out_blob_ids.at(i);
        auto in_blob = stream_executor.findBlob(in_id);
        stream_executor.updateBlob(out_id, in_blob.first, in_blob.second);
    }

    executePrimBlock(*loop_block_node, stream_executor);
    DLOG(INFO) <<"PrimLoop: loop_index = "<< loop_index;

    if(loop_index < max_trip_cnt && loop_cond == 1) {
        // Set to Block's next node
        stream_executor.setCursor(loop_block_id + 1);
    } else {
        // jump to matched EndLoop
    }
}

void executePrimEndLoop(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "executePrimEndLoop";
    SHOW_OP_INFO(op_node);

    auto end_loop_node = cast_if<nncir::PrimEndLoopNode>(op_node);

    // get the matched StartLoopNode
    int64_t loop_start_node_id =  end_loop_node->getGotoNode();
    auto loop_index_node = stream_executor.getGraph()->getNode(loop_start_node_id + 1);

    // Get newset LoopIndex
    int loop_index_blob_id = getOutBlobIds(*loop_index_node)[0];
    DLOG(INFO) <<"LoopIndexBlobId:" << loop_index_blob_id;
    int64_t loop_index = stream_executor.findBlob(loop_index_blob_id).second.toInt();

    // loop_index ++
    loop_index++;

    // update loop_index blob
    stream_executor.updateBlob(loop_index_blob_id, DataType::INT64, scalarToIValue<int64_t>(loop_index));

    DLOG(INFO) <<"PrimEndLoop: loop_index="<<loop_index;

    // jump to loopStart
    // LoopBlock
    stream_executor.setCursor(loop_start_node_id + 2);

    // update EndLoop's input Blobs --> PrimBlock's input
    auto end_loop_input_blob_ids = getInBlobIds(op_node);
    auto prim_block_input_blob_ids = getInBlobIds(*stream_executor.getGraph()->getNode(loop_start_node_id + 2));

    assert(end_loop_input_blob_ids.size() == prim_block_input_blob_ids.size());
    // skip 0 element
    for(int i=1; i<end_loop_input_blob_ids.size(); i++) {
        auto end_loop_input_blob = stream_executor.findBlob(end_loop_input_blob_ids.at(i));
        stream_executor.updateBlob(prim_block_input_blob_ids.at(i), end_loop_input_blob.first, end_loop_input_blob.second);
    }
}

}  // namespace nnrt
