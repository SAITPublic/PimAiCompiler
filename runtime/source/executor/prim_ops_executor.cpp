#include "glog/logging.h"
#include "executor/prim_ops_executor.h"
#include "common/include/cast.hpp"
#include "executor/prim_ops.h"
#include "executor/stream_executor.h"
#include "executor/utils.h"
#include "ir/include/control_nodes/prim_constant_node.hpp"
#include "ir/include/control_nodes/prim_dtype_node.hpp"
#include "ir/include/control_nodes/prim_if_node.hpp"
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/nn_ir.hpp"

namespace nnrt
{
void executePrimConstant(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
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

        if (bit_width == 16)
            scalar_type = DataType::FLOAT16;
        else if (bit_width == 32)
            scalar_type = DataType::FLOAT32;
        else {
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

void executePrimDtype(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
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
    int scalar_type = primDtype(tensor);

    // update output
    auto& out_edge = cast<nncir::DataEdge>(dtype_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, scalarToIValue(scalar_type));
}

}  // namespace nnrt
