#include "new_runtime/include/executor/prim_ops_executor.h"
#include "new_ir/include/layers/all_layers.h"

namespace nn_compiler
{
namespace runtime
{
void executePrimConstant(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimConstant";
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
        DLOG(INFO) << "Set Data: " << tmp;
        iv = scalarToIValue(tmp);
        DLOG(INFO) << "IV: " << iv;
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
            DLOG(FATAL) << "PrimConstant Error, unsupport data type when create Tensor!";
        }

        std::vector<int64_t> input_shape = getDataShapeFromSTensor(shape_);
        std::vector<int64_t> stride = getDataShapeFromVector(stride_);

        // TODO(SRCX): add gnmt optimization
        if (stream_executor.model_type_ == "GNMT" && constant_layer->getOutSTensorID()[0] == 3) {
            std::vector<int64_t> reorder_shape(input_shape);

            int align_m = 32;
            int align_k = 16;

            int m_align = (input_shape[0] + align_m - 1) / align_m * align_m;
            int k_align = (input_shape[1] + align_k - 1) / align_k * align_k;
            reorder_shape[0] = m_align;
            reorder_shape[1] = k_align;

            _Float16* ptr_origin = (_Float16*)ptr;
            auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCPU);
            auto reorder = at::zeros(reorder_shape, options);
            _Float16* y = (_Float16*)reorder.data_ptr();

            for (int i = 0; i < k_align / align_k; ++i) {
                for (int j = 0; j < m_align; ++j) {
                    for (int n = 0; n < align_k; ++n) {
                        int dst = n + j * align_k + i * align_k * m_align;
                        int src = j * input_shape[1] + (i * align_k + n);
                        if (j >= input_shape[0] || (i * align_k + n) >= input_shape[1]) {
                            y[dst] = 0;
                        } else {
                            y[dst] = ptr_origin[src];
                        }
                    }
                }
            }
            auto tensor = createPtTensor((void*)y, input_shape, scalar_type, stride).cuda();
            iv = tensorToIValue(tensor);
            dtype = DataType::TENSOR;
        } else {
            auto tensor = createPtTensor((void*)ptr, input_shape, scalar_type, stride).cuda();
            iv = tensorToIValue(tensor);
            dtype = DataType::TENSOR;
        }
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
                DLOG(FATAL) << "unspported datatype for tuple.";
            }
            inputs.push_back(iv_tmp);
        }
        primTupleConstruct(inputs, inputs.size());
        iv = inputs.at(0);
        dtype = DataType::TUPLE;
    } else if (ntype.find("[") != std::string::npos && ntype.find("]") != std::string::npos) {  // list type
        std::vector<torch::IValue> inputs;
        if (ntype.find("int") != std::string::npos) {
            auto data = constant_layer_attr->getData<int64_t>();
            for (auto item : *data) {
                inputs.push_back(intToIValue(item));
            }
            // Call OpKernel
            primListConstruct(inputs, inputs.size(), inferTypeFromDataType(DataType::INT64));
        } else if (ntype.find("float") != std::string::npos) {
            auto data = constant_layer_attr->getData<double>();
            for (auto item : *data) {
                inputs.push_back(doubleToIValue(item));
            }
            // Call OpKernel
            primListConstruct(inputs, inputs.size(), inferTypeFromDataType(DataType::FLOAT64));
        }
        iv = inputs.at(0);
        dtype = DataType::LIST;
    } else {
        DLOG(FATAL) << "PrimConstant Error, ntype: " << ntype << "unsupport!";
    }

    // the stensor id of all output edge is same in Contant Op
    auto out_stensor_id = layer->getOutSTensorID()[0];
    stream_executor.updateBlob(out_stensor_id, dtype, iv);
}

void executePrimData(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimData";

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
    DLOG(INFO) << "execute PrimDevice";

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
    DLOG(INFO) << "execute PrimDtype";

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
    DLOG(INFO) << "execute PrimEndIf";

    auto end_if_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimEndIfLayer>(layer);
    int if_id = end_if_layer->getIfLayerId();

    DLOG(INFO) << "end_if_id: " << layer->getID();
    DLOG(INFO) << "if_id: " << if_id;

    int64_t next_layer_id = -1;
    nn_compiler::ir::NNLayer* next_layer = nullptr;

    // if --- endif -(else body)--- end
    if (!end_if_layer->getIsElseNet()) {
        next_layer_id = end_if_layer->getGotoLayer();
        DLOG(INFO) << "If layer run then branch!";
    } else {
        DLOG(INFO) << "If layer run else branch!";
        next_layer_id = layer->getID() + 1;
        DLOG(INFO) << "EndIfLayer.next_layer_id:" << next_layer_id;

        auto in_stensor_ids = layer->getInSTensorID();
        if (in_stensor_ids.size() > 0) {
            auto if_layer = stream_executor.getGraph()->getLayerByPosition(if_id);
            auto if_layer_in_stensor_ids = if_layer->getInSTensorID();

            // Find the input blob, named condition, it is a int64/bool value
            torch::jit::IValue iv = stream_executor.findBlob(if_layer_in_stensor_ids[0]).second;
            int64_t condiction;
            if (iv.isInt()) {
                condiction = iv.toInt();
            } else if (iv.isBool()) {
                condiction = iv.toBool();
            } else {
                DLOG(FATAL) << "PrimEndIf Error, unsupport data type!";
            }
            assert(in_stensor_ids.size() % 2 == 0);
            DLOG(INFO) << "Num Input: " << in_stensor_ids.size();

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
    DLOG(INFO) << "execute PrimGetAttr";

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
    DLOG(INFO) << "execute PrimIf";
    auto if_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimIfLayer>(layer);
    auto in_stensor_ids = layer->getInSTensorID();
    assert(in_stensor_ids.size() == 1);

    DLOG(INFO) << "PrimIfLayer.if" << layer->getID();

    // Find input edge, primIf only have one input
    // Find the input blob, named condition, it is a int64/bool value
    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_ids.at(0)).second;
    int64_t condiction;
    if (iv.isInt()) {
        condiction = iv.toInt();
    } else if (iv.isBool()) {
        condiction = iv.toBool();
    } else {
        DLOG(FATAL) << "PrimIf Error, unsupport data type!";
    }
    assert(condiction == 0 || condiction == 1);

    int64_t next_layer_id = -1;
    nn_compiler::ir::NNLayer* next_layer = nullptr;

    // ref: https://github.sec.samsung.net/PIM/NNCompiler/pull/74/files
    if (condiction == 1) {
        DLOG(INFO) << "PrimIf(True branch)";
        // choose then_net
        // next_node = if_node.id + 1
        next_layer_id = layer->getID() + 1;
    } else {
        DLOG(INFO) << "PrimIf(False branch)";
        // getElseNetStartNode
        next_layer_id = if_layer->getElseNetStartLayer();
    }

    DLOG(INFO) << "PrimIf_next_layer_id:" << next_layer_id;
    stream_executor.setCursor(next_layer_id);
}

void executePrimListConstruct(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimListConstruct";
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
    DLOG(INFO) << "execute PrimListUnpack";

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
    DLOG(INFO) << "executePrimLoopIndex";

    // cast Layer
    auto loop_index_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimLoopIndexLayer>(layer);
    int64_t loop_index = loop_index_layer->getIndex();
    if (loop_index < 0) {
        DLOG(INFO) << "Invalid value for LoopIndex! set default loopIndex = 0!";
        loop_index = 0;
    }

    // check LoopIndex != INT64_MAX,  INT64_MAX is a default value, so LoopIndex need to be re-initialize
    if (nn_compiler::ir::isDefaultValue(loop_index)) {
        loop_index = 0;
    }

    // LoopIndexLayer only has one blob
    auto out_stensor_id = layer->getOutSTensorID()[0];
    stream_executor.updateBlob(out_stensor_id, DataType::INT64, scalarToIValue<int64_t>(loop_index));
}

void executePrimRaiseException(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimRaiseException";

    auto in_stensor_ids = layer->getInSTensorID();
    // Find the input blob
    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_ids[0]).second;
    // Call OpKernel
    primRaiseException(iv.toString()->string());
}

void executePrimSetAttr(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimSetAttr";

    auto in_stensor_ids = layer->getInSTensorID();
    // first edge is variable node, second edge is the data saved to the variable node.
    auto type = stream_executor.findBlob(in_stensor_ids[1]).first;
    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_ids[1]).second;
    stream_executor.updateBlob(in_stensor_ids[0], type, iv);
}

void executePrimTupleConstruct(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimTupleConstruct";

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

void executePrimTupleIndex(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimTupleIndex";

    auto tuple_index_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimTupleIndexLayer>(layer);
    auto in_stensor_ids = layer->getInSTensorID();
    // Find the input data blob
    torch::jit::IValue data_iv = stream_executor.findBlob(in_stensor_ids[0]).second;

    auto out_stensor_id = layer->getOutSTensorID()[0];
    std::vector<torch::IValue> inputs;
    if (data_iv.isNone()) {
        torch::jit::IValue iv;
        stream_executor.updateBlob(out_stensor_id, DataType::NONE, iv);
    } else {
        auto ivalues = data_iv.toTuple()->elements();
        for (auto iv : ivalues) {
            inputs.push_back(iv);
        }
        int64_t index = tuple_index_layer->getIndex();
        if (nn_compiler::ir::isDefaultValue(index)) {
            // Find the input index blob
            torch::jit::IValue index_iv = stream_executor.findBlob(in_stensor_ids[1]).second;
            index = index_iv.toInt();
        }
        // Call OpKernel
        auto output_iv = primTupleIndex(inputs, index);
        // update output
        if (output_iv.isTensor()) {
            stream_executor.updateBlob(out_stensor_id, DataType::TENSOR, output_iv);
        } else if (output_iv.isTuple()) {
            stream_executor.updateBlob(out_stensor_id, DataType::TUPLE, output_iv);
        } else {
            DLOG(FATAL) << "Unsupported input for PrimTupleIndex.";
        }
    }
}

void executePrimTupleUnpack(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimTupleUnpack";
    auto in_stensor_ids = layer->getInSTensorID();
    torch::jit::IValue iv = stream_executor.findBlob(in_stensor_ids[0]).second;

    // Call OpKernel
    auto output = primTupleUnpack(iv.toTuple());

    // update output
    auto out_stensor_ids = getUniqueOutStensorIds(layer);
    for (int i = 0; i < output.size(); i++) {
        auto type = inferDataType(output.at(i));
        stream_executor.updateBlob(out_stensor_ids.at(i), type, output.at(i));
    }
}

void executePrimType(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimType";

    auto in_stensor_ids = layer->getInSTensorID();
    auto map_value = stream_executor.findBlob(in_stensor_ids[0]);
    torch::jit::IValue iv = map_value.second;
    std::string device_type = "cpu";
    // FIXME: need to ensure PrimType op how to executor
    if (iv.isDevice()) {
        device_type = iv.toDevice().str();
    } else {
        DLOG(FATAL) << "PrimType op's input data is incorrect!";
    }

    // update output
    auto out_stensor_id = layer->getOutSTensorID()[0];
    stream_executor.updateBlob(out_stensor_id, DataType::STRING, strToIValue(device_type));
}

void executePrimUncheckedCast(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimUncheckedCast";

    auto in_stensor_ids = layer->getInSTensorID();
    auto map_value = stream_executor.findBlob(in_stensor_ids[0]);
    auto type = map_value.first;
    // Find the input blob
    torch::jit::IValue iv = map_value.second;
    // Call OpKernel
    auto output = primUncheckedCast(iv);
    // update output
    auto out_stensor_id = layer->getOutSTensorID()[0];
    stream_executor.updateBlob(out_stensor_id, type, output);
}

void executePrimUninitialized(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimUninitialized";

    // Call OpKernel
    auto output = primUninitialized();
    // update output
    auto out_stensor_id = layer->getOutSTensorID()[0];
    auto type = inferDataType(output);
    stream_executor.updateBlob(out_stensor_id, type, output);
}

void executePrimVariable(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimVariable";

    auto variable_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimVariableLayer>(layer);
    auto ntype = variable_layer->getNType();
    auto variable_attrs = variable_layer->getAttr();
    at::ListTypePtr list_type = at::ListType::ofTensors();

    if (ntype.find("List") != std::string::npos) {  // list type  list[scalar,scalar] list[tensor,tensor]
        auto d_type = variable_attrs.at(0)->getDataType();
        auto tensor_shape = variable_attrs.at(0)->getTensorShape();
        auto shape = getDataShapeFromSTensor(tensor_shape);
        int size = 1;
        for (auto item : shape) {
            size *= item;
        }
        std::vector<torch::IValue> inputs;
        torch::jit::IValue iv;

        for (unsigned int i = 0; i < variable_attrs.size(); i++) {
            auto t_dtensor = variable_attrs.at(i);
            auto d_type = t_dtensor->getDataType();
            auto tensor_shape = t_dtensor->getTensorShape();
            auto stride_ = t_dtensor->getStride();
            auto shape = getDataShapeFromSTensor(tensor_shape);
            auto stride = getDataShapeFromVector(stride_);
            auto cur_data = *(t_dtensor->getData<uint8_t>());
            uint8_t* ptr = const_cast<uint8_t*>(cur_data.data());
            if (size != 1) {
                auto tensor = createPtTensor((void*)(ptr), shape, d_type, stride).cuda();
                iv = tensorToIValue(tensor);
                list_type = at::ListType::ofTensors();
            } else {
                iv = convertVaraibleData2IValve(ptr, d_type);
                list_type = inferTypeFromDataType(d_type);
            }
            inputs.push_back(iv);
        }
        auto out_stensor_id = layer->getOutSTensorID()[0];

        // ntype = "List[Tuple[Tensor,Tensor]]" for variable op created by getattr or set attr;
        if (ntype.length() > 4) {
            torch::jit::IValue iv = primVariable(ntype, inputs);
            stream_executor.updateBlob(out_stensor_id, DataType::LIST, iv);
        } else {
            primListConstruct(inputs, inputs.size(), list_type);
            stream_executor.updateBlob(out_stensor_id, DataType::LIST, inputs.at(0));
        }
    } else if (ntype.find("bool") != std::string::npos) {  // bool type
        auto data_ptr = variable_attrs.at(0)->getData<uint8_t>();
        auto d_type = variable_attrs.at(0)->getDataType();
        uint8_t* ptr = const_cast<uint8_t*>(data_ptr->data());
        torch::jit::IValue iv = convertVaraibleData2IValve(ptr, d_type);

        auto out_stensor_id = layer->getOutSTensorID()[0];
        stream_executor.updateBlob(out_stensor_id, DataType::BOOL, iv);
    } else if (ntype.find("Tensor") != std::string::npos) {  // tesnor type
        auto t_dtensor = variable_attrs.at(0);
        auto d_type = t_dtensor->getDataType();
        auto tensor_shape = t_dtensor->getTensorShape();
        auto stride_ = t_dtensor->getStride();
        auto shape = getDataShapeFromSTensor(tensor_shape);
        auto stride = getDataShapeFromVector(stride_);

        auto cur_data = *(t_dtensor->getData<uint8_t>());
        uint8_t* ptr = const_cast<uint8_t*>(cur_data.data());
        auto tensor = createPtTensor((void*)(ptr), shape, d_type, stride).cuda();
        torch::jit::IValue iv = tensorToIValue(tensor);
        auto out_stensor_id = layer->getOutSTensorID()[0];
        stream_executor.updateBlob(out_stensor_id, DataType::TENSOR, iv);
    } else {
        DLOG(FATAL) << "Variable op data type: " << ntype << "do not support! ";
    }
}

/**
 * @brief check Loop has (x1, x2, ...xr) inputs
 * according to torch::loop, %y_1, ..., %y_r = prim::Loop(%max_trip_count, %initial_condition, %x_1, ..., %x_r)
 * https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#loops
 * @param loop_layer
 * @return true
 * @return false
 */
static bool loopHasExtraInputs(std::shared_ptr<nn_compiler::ir::PrimLoopLayer>& loop_layer)
{
    int64_t loop_cond = loop_layer->getCond();
    int64_t max_trip_cnt = loop_layer->getTripCount();
    bool trip_cnt_invalid = nn_compiler::ir::isDefaultValue(max_trip_cnt);
    bool loop_cond_invalid = nn_compiler::ir::isDefaultValue(loop_cond);

    int64_t num_inputs = loop_layer->getInSTensorID().size();

    if (trip_cnt_invalid && loop_cond_invalid) {
        if (num_inputs == 2) return false;
    } else if (!trip_cnt_invalid && loop_cond_invalid) {
        if (num_inputs == 1) return false;
    } else if (trip_cnt_invalid && !loop_cond_invalid) {
        if (num_inputs == 1) return false;
    } else if (!trip_cnt_invalid && !loop_cond_invalid) {
        if (num_inputs == 0) return false;
    }

    return true;
}

std::unordered_map<std::string, int64_t> getMatchedLoopInfo(int64_t loop_block_id, StreamExecutor& stream_executor)
{
    // Get LoopNode
    int64_t loop_id = loop_block_id - 2;
    assert(loop_id >= 0);
    auto graph = stream_executor.getGraph();
    auto layer = graph->getLayerByPosition(loop_id);
    auto loop_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimLoopLayer>(layer);

    // max_cnt & cond
    int64_t max_trip_cnt = loop_layer->getTripCount();
    int64_t cond = loop_layer->getCond();

    // check default
    auto in_stensor_ids = layer->getInSTensorID();
    if (nn_compiler::ir::isDefaultValue(max_trip_cnt)) {
        // Get from loop's input[0]
        max_trip_cnt = stream_executor.findBlob(in_stensor_ids[0]).second.toInt();
    }
    if (nn_compiler::ir::isDefaultValue(cond)) {
        // Get from loop's input[1]
        cond = stream_executor.findBlob(in_stensor_ids[1]).second.toInt();
    }

    int64_t loop_index_id = loop_block_id - 1;
    auto loop_index_layer_ = graph->getLayerByPosition(loop_index_id);
    int blob_id = loop_index_layer_->getOutSTensorID()[0];
    auto blob = stream_executor.findBlob(blob_id);
    assert(blob.first == DataType::INT64);
    int64_t loop_index = blob.second.toInt();

    // get loopEndNode.id + 1
    int end_loop_next_id = loop_layer->getGotoLayer();

    std::unordered_map<std::string, int64_t> umap;
    umap.insert({{"max_trip_cnt", max_trip_cnt},
                 {"cond", cond},
                 {"loop_index", loop_index},
                 {"end_loop_next_id", end_loop_next_id}});
    umap.insert({"loop_id", loop_layer->getID()});
    return umap;
}

void executePrimBlock(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimBlock";

    // Prim Blcok only transfer input -- block --> outputs
    // Block( loopIndex: int,  x1, x2.....xr : IValue)

    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();

    auto umap = getMatchedLoopInfo(layer->getID(), stream_executor);
    int64_t max_trip_cnt = umap["max_trip_cnt"];
    int64_t cond = umap["cond"];
    int64_t loop_index = umap["loop_index"];
    int64_t end_loop_next_id = umap["end_loop_next_id"];
    int64_t loop_layer_id = umap["loop_id"];

    if (loop_index >= max_trip_cnt || cond == 0) {
        // Get EndLoop's input
        // transfer EndLoop's input --> output
        auto end_loop_layer = stream_executor.getGraph()->getLayerByPosition(end_loop_next_id - 1);
        auto end_loop_in_stensor_ids = end_loop_layer->getInSTensorID();
        auto end_loop_out_stensor_ids = end_loop_layer->getOutSTensorID();

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

        assert(end_loop_in_stensor_ids.size() == end_loop_out_stensor_ids.size() + 1);
        for (int i = 0; i < end_loop_out_stensor_ids.size(); i++) {
            auto in_blob = stream_executor.findBlob(end_loop_in_stensor_ids.at(i + 1));
            stream_executor.updateBlob(end_loop_out_stensor_ids.at(i), in_blob.first, in_blob.second);
        }
        // Jump to End Loop'next
        int64_t temp_cond = stream_executor.loop_condition_stack_.top();
        stream_executor.loop_condition_stack_.pop();
        auto loop_op_layer = stream_executor.getGraph()->getLayerByPosition(loop_layer_id);
        auto loop_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimLoopLayer>(loop_op_layer);
        loop_layer->setCond(temp_cond);
        stream_executor.setCursor(end_loop_next_id);
    } else {
        auto graph = stream_executor.getGraph();
        auto loop_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimLoopLayer>(graph->getLayerByPosition(loop_layer_id));
        if (!loopHasExtraInputs(loop_layer)) {
            // the in_stensor_ids[0] is invalid, only is maintain linkage
            // only need to pass LoopIndex into Block's inner
            // auto in_blob = stream_executor.findBlob(in_stensor_ids.at(1));
            // stream_executor.updateBlob(out_blob_ids.at(0), DataType::INT64, loop_index);
        } else {
            std::vector<int64_t> temp_out_blob_ids(in_stensor_ids.size(), -1);
            int out_id = 0;
            for (int i = 0; i < out_stensor_ids.size(); i++) {
                if (std::find(temp_out_blob_ids.begin(), temp_out_blob_ids.end(), out_stensor_ids.at(i)) ==
                    temp_out_blob_ids.end()) {
                    temp_out_blob_ids[out_id++] = out_stensor_ids.at(i);
                }
            }

            for (int i = 0; i < in_stensor_ids.size(); i++) {
                auto in_blob = stream_executor.findBlob(in_stensor_ids.at(i));
                stream_executor.updateBlob(temp_out_blob_ids.at(i), in_blob.first, in_blob.second);
            }
        }
        stream_executor.updateBlob(out_stensor_ids.at(0), DataType::INT64, intToIValue(loop_index));
        stream_executor.setCursor(layer->getID() + 1);
    }
}

void executePrimLoop(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimLoop";

    auto loop_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimLoopLayer>(layer);
    int64_t loop_layer_id = layer->getID();

    // ref: torch_jit Loop: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#loops
    int64_t loop_cond = loop_layer->getCond();
    stream_executor.loop_condition_stack_.push(loop_cond);
    int64_t max_trip_cnt = loop_layer->getTripCount();

    int edge_id = 0;
    auto in_stensor_ids = layer->getInSTensorID();
    auto out_stensor_ids = layer->getOutSTensorID();
    // check default
    if (nn_compiler::ir::isDefaultValue(max_trip_cnt)) {
        // Get from loop's input[0]
        max_trip_cnt = stream_executor.findBlob(in_stensor_ids[edge_id++]).second.toInt();
    }
    if (nn_compiler::ir::isDefaultValue(loop_cond)) {
        // Get from loop's input[1]
        loop_cond = stream_executor.findBlob(in_stensor_ids[edge_id++]).second.toInt();
    }

    // loop_layer.id + 1 ---> loopIndex_layer
    // loop_layer.id + 2 ---> PrimBlockLayer

    // Get LoopIndex
    auto loop_index_layer_ = stream_executor.getGraph()->getLayerByPosition(loop_layer_id + 1);
    int64_t loop_index_blob_id = loop_index_layer_->getOutSTensorID()[0];
    //
    executePrimLoopIndex(loop_index_layer_, stream_executor);  // BUG: this will reset loop_index

    DLOG(INFO) << "loop_index_blob_id: " << loop_index_blob_id;
    int64_t loop_index = stream_executor.findBlob(loop_index_layer_->getOutSTensorID()[0]).second.toInt();

    // Get LoopBlockNode.id
    int loop_block_id = loop_layer_id + 2;
    auto loop_block_layer_ = stream_executor.getGraph()->getLayerByPosition(loop_block_id);

    // the Loop's input maybe empty
    // loopHasExtraInputs(loop_node)
    if (loopHasExtraInputs(loop_layer)) {
        // assert(in_stensor_ids.size() == out_stensor_ids.size() + 1);
        for (int i = 0; i < out_stensor_ids.size(); i++) {
            int64_t in_id = in_stensor_ids.at(i + edge_id);
            int64_t out_id = out_stensor_ids.at(i);
            auto in_blob = stream_executor.findBlob(in_id);
            stream_executor.updateBlob(out_id, in_blob.first, in_blob.second);
        }
    }

    executePrimBlock(loop_block_layer_, stream_executor);
    DLOG(INFO) << "PrimLoop: loop_index = " << loop_index;
}

void executePrimEndLoop(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute PrimEndLoop";

    auto end_loop_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimEndLoopLayer>(layer);

    // get the matched StartLoopNode
    int64_t loop_start_layer_id = end_loop_layer->getGotoLayer();
    auto loop_index_layer_ = stream_executor.getGraph()->getLayerByPosition(loop_start_layer_id + 1);

    // Get newset LoopIndex
    int loop_index_blob_id = loop_index_layer_->getOutSTensorID()[0];
    DLOG(INFO) << "LoopIndexBlobId:" << loop_index_blob_id;
    int64_t loop_index = stream_executor.findBlob(loop_index_blob_id).second.toInt();

    // loop_index ++
    loop_index++;

    // update loop_index blob
    stream_executor.updateBlob(loop_index_blob_id, DataType::INT64, scalarToIValue<int64_t>(loop_index));

    DLOG(INFO) << "PrimEndLoop: loop_index=" << loop_index;

    // jump to loopStart
    // LoopBlock
    stream_executor.setCursor(loop_start_layer_id + 2);

    // update EndLoop's input Blobs --> PrimBlock's input
    auto end_loop_input_blob_ids = layer->getInSTensorID();
    auto prim_block_layer_ = stream_executor.getGraph()->getLayerByPosition(loop_start_layer_id + 2);
    auto prim_block_input_blob_ids = prim_block_layer_->getInSTensorID();

    // the 0 element is condition
    auto prim_loop_layer_ = stream_executor.getGraph()->getLayerByPosition(loop_start_layer_id);
    auto prim_loop_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimLoopLayer>(prim_loop_layer_);
    auto cond_iv = stream_executor.findBlob(end_loop_input_blob_ids.at(0)).second;
    if (cond_iv.isInt()) {
        prim_loop_layer->setCond(cond_iv.toInt());
    } else {
        prim_loop_layer->setCond(cond_iv.toBool());
    }

    for (int i = 1; i < end_loop_input_blob_ids.size(); i++) {
        auto end_loop_input_blob = stream_executor.findBlob(end_loop_input_blob_ids.at(i));
        stream_executor.updateBlob(prim_block_input_blob_ids.at(i), end_loop_input_blob.first,
                                   end_loop_input_blob.second);
    }
}

}  // namespace runtime
}  // namespace nn_compiler
