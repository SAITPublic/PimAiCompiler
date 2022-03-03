#include "new_runtime/include/executor/prim_ops_executor.h"
#include "new_ir/include/layers/all_layers.h"

namespace nn_compiler
{
namespace runtime
{
void executePrimConstant(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute PrimConstant node";
    auto constant_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(layer);
    assert(layer->getInSTensorID() == 0);  // Constant Op

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
}

void executePrimTupleConstruct(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    Log::RT::D() << "execute PrimTupleConstruct node";

    auto in_stensor_ids = layer->getInSTensorID();

    std::vector<torch::IValue> inputs;
    for (unsigned int i = 0; i < in_stensor_ids.size(); i++) {
        torch::jit::IValue iv = stream_executor.findBlob(in_stensor_ids[i]).second;
        inputs.push_back(iv);
    }
    // Call OpKernel
    primTupleConstruct(inputs, inputs.size());

    // update output
    auto out_blob_id = getUniqueOutStensorIds(layer)[0];
    stream_executor.updateBlob(out_blob_id, DataType::TUPLE, scalarToIValue(inputs.at(0)));
}

}  // namespace runtime
}  // namespace nn_compiler
