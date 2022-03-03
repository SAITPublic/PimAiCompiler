#include "new_runtime/include/executor/prim_ops_executor.h"
#include "new_ir/include/layers/all_layers.h"

namespace nn_compiler
{
namespace runtime
{
void executePrimConstant(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute PrimConstant";
    auto constant_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(layer);
    assert(layer->getInSTensorID().size() == 0);  // Constant Op

    // Create IValue
    torch::jit::IValue iv;

    // Ntype: Scalar(float, int32, int64...), String, Tensor, Device
    auto ntype = constant_layer->getNType();

    // dtype: {float, int32, int64, ... String, Tensor, Device}
    DataType dtype = DataType::NONE;

    // Get the data_ptr
    auto constant_layer_attr = constant_layer->getAttr();
    auto constant_data = constant_layer_attr->getData<uint8_t>();
    uint8_t* ptr = const_cast<uint8_t*>(constant_data->data());

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
        Log::RT::D() << "Set Data: " << tmp;
        iv = scalarToIValue(tmp);
        Log::RT::D() << "IV: " << iv;
        dtype = DataType::INT64;
    } else if (ntype == "float") {
        // float64
        iv = scalarToIValue<double>(*(double*)ptr);
        dtype = DataType::FLOAT64;
    } else if (ntype == "Tensor") {
        auto shape_ = constant_layer_attr->getTensorShape();
        auto bit_width = constant_layer_attr->getBitWidth();
        auto stride_ = constant_layer_attr->getStride();
        auto scalar_type = DataType::NONE;
        if (bit_width == 16) {
            scalar_type = DataType::FLOAT16;
        } else if (bit_width == 32) {
            scalar_type = DataType::FLOAT32;
        } else {
            Log::RT::E() << "PrimConstant Error, unsupport data type when create Tensor!";
        }

        std::vector<int64_t> input_shape = getDataShapeFromSTensor(shape_);
        std::vector<int64_t> stride = getDataShapeFromVector(stride_);

        // TODO(SRCX): add gnmt optimization

        auto tensor = createPtTensor((void*)ptr, input_shape, scalar_type, stride).cuda();
        iv = tensorToIValue(tensor);
        dtype = DataType::TENSOR;
    } else if (ntype.find("None") != std::string::npos) {
        dtype = DataType::NONE;
    } else if (ntype.find("(") != std::string::npos && ntype.find(")") != std::string::npos) {  // tuple type
        auto ntypeParser_ = parseNtype(ntype);
        std::vector<torch::IValue> inputs;
        for (uint32_t i = 0; i < ntypeParser_.first; i++) {
            torch::IValue iv_tmp;
            if (ntypeParser_.second = DataType::INT64) {
                int64_t tmp = *(int64_t*)(ptr + sizeof(int64_t) * i);
                iv_tmp = scalarToIValue<int64_t>(tmp);
            } else if (ntypeParser_.second = DataType::FLOAT64) {
                double tmp = *(int64_t*)(ptr + sizeof(int64_t) * i);
                iv_tmp = scalarToIValue<double>(tmp);
            } else {
                Log::RT::E() << "unspported datatype for tuple.";
            }
            inputs.push_back(iv_tmp);
        }
        primTupleConstruct(inputs, inputs.size());
        iv = inputs.at(0);
        dtype = DataType::TUPLE;
    } else {
        Log::RT::E() << "PrimConstant Error, ntype: " << ntype << "unsupport!";
    }

    // the stensor id of all output edge is same in Contant Op
    auto out_stensor_id = layer->getOutSTensorID()[0];
    stream_executor.updateBlob(out_stensor_id, dtype, iv);
}

void executePrimData(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute PrimData";

    auto in_stensor_ids = layer->getInSTensorID();
    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_ids[0]).second;
    assert(iv.isTensor());
    torch::Tensor tensor = iv.toTensor();

    // Call OpKernel
    torch::Tensor tensor_value = primData(tensor);
    // update output
    auto out_stensor_id = layer->getOutSTensorID()[0];
    stream_executor.releation_blob_ids_map_.insert({out_stensor_id, {in_stensor_ids[0], -1}});
    stream_executor.updateBlob(out_stensor_id, DataType::TENSOR, tensorToIValue(tensor_value));
}

void executePrimDevice(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute PrimDevice";

    auto in_stensor_ids = layer->getInSTensorID();
    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_ids[0]).second;
    assert(iv.isTensor());
    torch::Tensor tensor = iv.toTensor();

    // Call OpKernel
    c10::Device device_value = primDevice(tensor);
    // update output
    auto out_stensor_id = layer->getOutSTensorID()[0];
    stream_executor.updateBlob(out_stensor_id, DataType::DEVICE, deviceToIValue(device_value));
}

void executePrimDtype(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute PrimDtype";

    // Find input edge, primDtype only have one input & output
    auto in_stensor_ids = layer->getInSTensorID();
    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_ids[0]).second;
    assert(iv.isTensor());
    torch::Tensor tensor = iv.toTensor();

    // Call OpKernel
    int64_t scalar_type = primDtype(tensor);

    // update output
    auto out_stensor_id = layer->getOutSTensorID()[0];
    stream_executor.updateBlob(out_stensor_id, DataType::INT64, scalarToIValue(scalar_type));
}

void executePrimEndIf(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute PrimEndIf";

    auto end_if_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimEndIfLayer>(layer);
    int if_id = end_if_layer->getIfLayerId();

    Log::RT::D() << "end_if_id: " << layer->getID();
    Log::RT::D() << "if_id: " << if_id;

    int64_t next_layer_id = -1;
    nn_compiler::ir::NNLayer* next_layer = nullptr;

    // if --- endif -(else body)--- end
    if (!end_if_layer->getIsElseNet()) {
        next_layer_id = end_if_layer->getGotoLayer();
        Log::RT::D() << "If layer run then branch!";
    } else {
        Log::RT::D() << "If layer run else branch!";
        next_layer_id = layer->getID() + 1;
        Log::RT::D() << "EndIfLayer.next_layer_id:" << next_layer_id;

        auto in_stensor_ids = layer->getInSTensorID();
        if (in_stensor_ids.size() > 0) {
            auto if_layer = stream_executor.getGraph()->getLayer(if_id);
            auto if_layer_in_stensor_ids = if_layer->getInSTensorID();

            // Find the input blob, named condition, it is a int64/bool value
            torch::jit::IValue iv = stream_executor.findBlob(if_layer_in_stensor_ids[0]).second;
            int64_t condiction;
            if (iv.isInt()) {
                condiction = iv.toInt();
            } else if (iv.isBool()) {
                condiction = iv.toBool();
            } else {
                Log::RT::E() << "PrimEndIf Error, unsupport data type!";
            }
            assert(in_stensor_ids.size() % 2 == 0);
            Log::RT::D() << "Num Input: " << in_stensor_ids.size();

            std::vector<int64_t> in_blob_ids;
            if (condiction == 1) {
                // Then Net ---> Endif
                // Get the TenNet's output
                for (int i = 0; i < in_stensor_ids.size() / 2; i++) {
                    in_blob_ids.push_back(in_stensor_ids[i]);
                }
            } else {
                // Else Net --> Endif
                for (int i = in_stensor_ids.size() / 2; i < in_stensor_ids.size(); i++) {
                    in_blob_ids.push_back(in_stensor_ids[i]);
                }
            }

            // Set inputs --> output
            // multi-inputs --> multi-outputs
            auto out_stensor_ids = getUniqueOutStensorIds(layer);
            for (int i = 0; i < in_blob_ids.size(); i++) {
                auto in_data = stream_executor.findBlob(in_blob_ids[i]);
                stream_executor.updateBlob(out_stensor_ids[i], in_data.first, in_data.second);
            }
        }
    }
    // update the output Blobs
    stream_executor.setCursor(next_layer_id);
}

void executePrimGetAttr(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute PrimGetAttr";

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = getUniqueOutStensorIds(layer);

    // update output
    for (int i = 0; i < out_stensor_ids.size(); i++) {
        auto map_value = stream_executor.findBlob(in_stensor_ids.at(i));
        torch::jit::IValue iv = map_value.second;
        auto type = map_value.first;
        stream_executor.updateBlob(out_stensor_ids.at(i), type, iv);
    }
}

void executePrimIf(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute PrimIf";
    auto if_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimIfLayer>(layer);
    auto in_stensor_ids = layer->getInSTensorID();
    assert(in_stensor_ids.size() == 1);

    Log::RT::D() << "PrimIfLayer.if" << layer->getID();

    // Find input edge, primIf only have one input
    // Find the input blob, named condition, it is a int64/bool value
    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_ids.at(0)).second;
    int64_t condiction;
    if (iv.isInt()) {
        condiction = iv.toInt();
    } else if (iv.isBool()) {
        condiction = iv.toBool();
    } else {
        Log::RT::E() << "PrimIf Error, unsupport data type!";
    }
    assert(condiction == 0 || condiction == 1);

    int64_t next_layer_id = -1;
    nn_compiler::ir::NNLayer* next_layer = nullptr;

    // ref: https://github.sec.samsung.net/PIM/NNCompiler/pull/74/files
    if (condiction == 1) {
        Log::RT::D() << "PrimIf(True branch)";
        // choose then_net
        // next_node = if_node.id + 1
        next_layer_id = layer->getID() + 1;
    } else {
         Log::RT::D() << "PrimIf(False branch)";
        // getElseNetStartNode
        next_layer_id = if_layer->getElseNetStartLayer();
    }

    Log::RT::D() << "PrimIf_next_layer_id:" << next_layer_id;
    stream_executor.setCursor(next_layer_id);
}

void executePrimListConstruct(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute PrimListConstruct";
    auto in_stensor_ids = layer->getInSTensorID();
    std::vector<torch::IValue> inputs;
    DataType type = DataType::NONE;
    for (unsigned int i = 0; i < in_stensor_ids.size(); i++) {
        // Find the input blob
        auto value_map = stream_executor.findBlob(in_stensor_ids.at(i));
        torch::jit::IValue iv = value_map.second;
        type = value_map.first;
        inputs.push_back(iv);
    }
    // Call OpKernel
    primListConstruct(inputs, inputs.size(), inferTypeFromDataType(type));

    // update output
    auto out_stensor_id = layer->getOutSTensorID()[0];

    stream_executor.updateBlob(out_stensor_id, DataType::LIST, inputs.at(0));
}

void executePrimListUnpack(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute PrimListUnpack";

    auto in_stensor_ids = layer->getInSTensorID();
    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_ids[0]).second;

    // Don't need to call nnrt::primListUnpack anymore
    std::vector<torch::IValue> results;
    for (auto item : iv.toListRef()) {
        results.push_back(item);
    }

    auto out_stensor_ids = getUniqueOutStensorIds(layer);
    for (int i = 0; i < results.size(); i++) {
        auto type = inferDataType(results.at(i));
        stream_executor.updateBlob(out_stensor_ids.at(i), type, results.at(i));
    }
}

void executePrimLoopIndex(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "executePrimLoopIndex";

    // cast Node
    auto loop_index_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimLoopIndexLayer>(layer);
    int64_t loop_index = loop_index_layer->getIndex();
    if (loop_index < 0) {
        Log::RT::D() << "Invalid value for LoopIndex! set default loopIndex = 0!";
        loop_index = 0;
    }

    // check LoopIndex != INT64_MAX,  INT64_MAX is a default value, so LoopIndex need to be re-initialize
    if (nn_compiler::ir::isDefaultValue(loop_index)) {
        loop_index = 0;
    }

    // LoopIndexNode only has one blob
    auto out_stensor_id = layer->getOutSTensorID()[0];
    stream_executor.updateBlob(out_stensor_id, DataType::INT64, scalarToIValue<int64_t>(loop_index));
}

void executePrimTupleConstruct(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute PrimTupleConstruct";

    auto in_stensor_ids = layer->getInSTensorID();

    std::vector<torch::IValue> inputs;
    for (unsigned int i = 0; i < in_stensor_ids.size(); i++) {
        torch::jit::IValue iv = stream_executor.findBlob(in_stensor_ids[i]).second;
        inputs.push_back(iv);
    }
    // Call OpKernel
    primTupleConstruct(inputs, inputs.size());

    // update output
    auto out_stensor_id = layer->getOutSTensorID()[0];
    stream_executor.updateBlob(out_stensor_id, DataType::TUPLE, scalarToIValue(inputs.at(0)));
}

void executePrimVariable(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{

}

}  // namespace runtime
}  // namespace nn_compiler
