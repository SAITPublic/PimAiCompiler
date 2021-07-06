#include "common/include/cast.hpp"
#include "executor/aten_ops.h"
#include "executor/stream_executor.h"
#include "executor/utils.h"
#include "glog/logging.h"
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/ir_types.hpp"
#include "ir/include/nn_ir.hpp"
#include "ir/include/nn_nodes/aten_add_node.hpp"

namespace nnrt
{
void executorAtenAdd(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Add node";

    auto add_node = cast<nncir::AtenAddNode>(op_node);
    // addOp has 3 inputs
    assert(add_node.getNumInputs() == 3);   

    auto& input_self = cast<nncir::DataEdge>(add_node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(add_node.getInEdge(1));

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
        auto output = nnrt::atenAdd(self_tensor, other_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(add_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
               dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::INT64 ||
               dtype == DataType::FLOAT32 || dtype == DataType::FLOAT64 || dtype == DataType::BOOL) {
        assert(iv_other.isScalar());
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = nnrt::atenAdd(self_tensor, other_scalar);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(add_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    }
}  // executorAtenAdd
}  // namespace nnrt
