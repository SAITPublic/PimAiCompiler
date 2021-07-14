#include "common/include/cast.hpp"
#include "executor/aten_ops.h"
#include "executor/stream_executor.h"
#include "executor/utils.h"
#include "glog/logging.h"
#include "ir/include/all_nodes.hpp"
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/ir_types.hpp"
#include "ir/include/nn_ir.hpp"

namespace nnrt
{
bool isScalarType(DataType dtype)
{
    return dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
           dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::INT64 ||
           dtype == DataType::FLOAT32 || dtype == DataType::FLOAT64 || dtype == DataType::BOOL;
}

void executorAtenDeriveIndex(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten DeriveIndex node";
    // TODO need new changes from npu_ir repo
}

void executorAtenGetItem(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten GetItem node";
    // TODO need new changes from npu_ir repo
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
void executorAtenAdd(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Add node";

    auto add_node = cast<nncir::AtenAddNode>(op_node);
    int64_t alpha = add_node.getAlpha();

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

void executorAtenAddMM(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Addmm node";
    // TODO need new changes from npu_ir repo
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

void executorAtenCat(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Cat node";

    auto cat_node = cast<nncir::AtenCatNode>(op_node);
    auto dim = cat_node.getDim();
    std::vector<at::Tensor> tensor_vec;

    auto& input_list_edge = cast<nncir::DataEdge>(cat_node.getInEdge(0));
    auto input_blob_id = input_list_edge.getBlobId();

    auto dtype = stream_executor.findBlob(input_blob_id).first;
    auto ivalue = stream_executor.findBlob(input_blob_id).second;
    assert(ivalue.isTensorList());

    auto c10_tensor_list = ivalue.toTensorList();
    for (auto tensor : c10_tensor_list) {
        tensor_vec.push_back(tensor);
    }
    at::TensorList tensor_list(tensor_vec);

    auto output = nnrt::atenCat(tensor_list, dim);
    auto& out_edge = cast<nncir::DataEdge>(cat_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorListToIValue(output));
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
    } else if (dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
               dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::INT64 ||
               dtype == DataType::FLOAT32 || dtype == DataType::FLOAT64 || dtype == DataType::BOOL) {
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
    auto& input_tensor = cast<nncir::DataEdge>(dropout_node.getInEdge(0));
    // Get input blob
    int input_tensor_blob_id = input_tensor.getBlobId();
    // Find the input blob
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    double proportion = (double)dropout_node.getProportion();
    bool train = dropout_node.getTrain();
    at::Tensor output = nnrt::atenDropout(tensor, proportion, train);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(dropout_node.getFirstOutEdge());

    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenEmbedding(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Embedding node";

    auto node = cast<nncir::AtenEmbeddingNode>(op_node);
    assert(node.getNumInputs() == 2);

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

    int64_t padding_idx = node.getPaddingIdx();
    bool scale_grad_by_freq = node.getScaleGradByFreq();
    bool sparse = node.getSparse();
    auto output = nnrt::atenEmbedding(iv_weights.toTensor(), iv_indices.toTensor(), padding_idx, scale_grad_by_freq, sparse);

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
        assert(iv_other.isTensor());
        at::Tensor self_tensor = iv_self.toTensor();
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenEq(self_tensor, other_tensor);
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

void executorAtenFormat(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "execute Aten Format node";

    auto format_node = cast<nncir::AtenFormatNode>(op_node);
    auto assembly_format = format_node.getAssemblyFormat();

    auto &input1 = cast<nncir::DataEdge>(format_node.getInEdge(1));
    auto &input2 = cast<nncir::DataEdge>(format_node.getInEdge(2));

    // Get input blob
    int input1_blob_id = input1.getBlobId();
    int input2_blob_id = input2.getBlobId();

    // Find the input blob
    auto i_value1 = stream_executor.findBlob(input1_blob_id).second;
    auto i_value2 = stream_executor.findBlob(input2_blob_id).second;

    auto dtype = stream_executor.findBlob(input1_blob_id).first;
    if (dtype == DataType::TUPLE) {
        // aten::format(string, tuple(int,..., int), list(int,..., int))
        c10::intrusive_ptr <c10::ivalue::Tuple> i_tuple_values = i_value1.toTuple();
        auto i_list_values = i_value2.toList();
        std::vector<std::string> value1;
        std::vector<std::string> value2;
        for (auto &item : i_tuple_values->elements()) {
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
        auto &out_edge = cast<nncir::DataEdge>(format_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::STRING, strToIValue(output));
    } else if (dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
               dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::INT64) {
        // aten::format(string, int, int)
        std::string str1 = std::to_string(i_value1.toInt());
        std::string str2 = std::to_string(i_value2.toInt());

        auto output = nnrt::atenFormat(assembly_format, str1, str2);
        // update output
        auto &out_edge = cast<nncir::DataEdge>(format_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::STRING, strToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::format";
    }
}

void executorAtenGt(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Gt node";

    auto node = cast<nncir::AtenGtNode>(op_node);
    assert(node.getNumInputs() == 2);

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
    auto dtype = stream_executor.findBlob(input_other_blob_id).first;
    if (dtype == DataType::TENSOR) {
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenGt(self_tensor, other_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (isScalarType(dtype)) {
        assert(iv_other.isScalar());
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = nnrt::atenGt(self_tensor, other_scalar);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    }
}

void executorAtenInt(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Int node";

    auto int_node = cast<nncir::AtenIntNode>(op_node);
    assert(int_node.getNumInputs() == 1);

    auto& input_self = cast<nncir::DataEdge>(int_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();

    auto dtype = stream_executor.findBlob(input_self_blob_id).first;
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isScalar());
    auto self_scalar = iv_self.toScalar();

    auto output = nnrt::atenInt(self_scalar);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(int_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, scalarToIValue(output));
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
    auto output = nnrt::atenItem(self_tensor);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::UNDEFINED, torch::jit::IValue(output));
}

void executorAtenLen(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Len node";

    auto node = cast<nncir::AtenLenNode>(op_node);
    assert(node.getNumInputs() == 1);

    auto& input_list = cast<nncir::DataEdge>(node.getInEdge(0));
    int input_list_blob_id = input_list.getBlobId();

    // Find the input blob
    torch::jit::IValue iv_list = stream_executor.findBlob(input_list_blob_id).second;

    assert(iv_list.isList());

    auto output = nnrt::atenLen(iv_list.toList());
    // update output
    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, intToIValue(output));
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

void executorAtenLt(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Lt node";

    auto node = cast<nncir::AtenLtNode>(op_node);
    assert(node.getNumInputs() == 2);

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
    auto dtype = stream_executor.findBlob(input_other_blob_id).first;
    if (dtype == DataType::TENSOR) {
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenLt(self_tensor, other_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (isScalarType(dtype)) {
        assert(iv_other.isScalar());
        at::Scalar other_scalar = iv_other.toScalar();
        auto output = nnrt::atenLt(self_tensor, other_scalar);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    }
}

void executorAtenMatmul(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Matmul node";

    auto node = cast<nncir::AtenMatmulNode>(op_node);
    assert(node.getNumInputs() == 2);

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

    auto& input_self = cast<nncir::DataEdge>(max_node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(max_node.getInEdge(1));

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
        // aten::max(Tensor, Tensor)
        assert(iv_other.isTensor());
        at::Tensor other_tensor = iv_other.toTensor();
        auto output = nnrt::atenMax(self_tensor, other_tensor);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(max_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
    } else if (dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
               dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::INT64 ||
               dtype == DataType::FLOAT32 || dtype == DataType::FLOAT64 || dtype == DataType::BOOL) {
        // aten::max(Tensor, dim, keepdim)
        auto dim = max_node.getDim();
        auto keep_dim = max_node.getKeepDim();

        auto output = nnrt::atenMax(self_tensor, dim, keep_dim);
        // update output
        auto& out_edge = cast<nncir::DataEdge>(max_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TUPLE, tupleToIValue(output));
    } else {
        DLOG(ERROR) << "Unsupported input type for aten::max";
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

    if (iv_self.isTensor()) {
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
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::BOOL, scalarToIValue(output));
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
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    auto output = nnrt::atenNeg(tensor);
    auto& out_edge = cast<nncir::DataEdge>(neg_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
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
    auto dim = select_node.getDim();
    auto index = select_node.getIndex();

    auto& input_self = cast<nncir::DataEdge>(select_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();

    auto dtype = stream_executor.findBlob(input_self_blob_id).first;
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto output = nnrt::atenSelect(self_tensor, dim, index);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(select_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenSize(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Size node";

    auto size_node = cast<nncir::AtenSizeNode>(op_node);
    assert(size_node.getNumInputs() == 1);

    auto& input_tensor = cast<nncir::DataEdge>(size_node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    int64_t output = nnrt::atenSize(tensor, size_node.getDim());
    auto& out_edge = cast<nncir::DataEdge>(size_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, intToIValue(output));
}

void executorAtenSlice(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Slice node";

    auto slice_node = cast<nncir::AtenSliceNode>(op_node);
    assert(slice_node.getNumInputs() == 1);
    auto& input_tensor = cast<nncir::DataEdge>(slice_node.getInEdge(0));
    ;
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());

    auto output = nnrt::atenSlice(iv_tensor.toTensor(), slice_node.getDim(), slice_node.getStart(), slice_node.getEnd(),
                                  slice_node.getStep());
    auto& out_edge = cast<nncir::DataEdge>(slice_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenSub(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Sub node";
    // TODO need new changes from npu_ir repo
}

void executorAtenTensor(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "execute Aten Tensor node";

    auto tensor_node = cast<nncir::AtenTensorNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(tensor_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isList());

    at::TensorOptions options;
    auto& edge_dtype      = cast<nncir::DataEdge>(tensor_node.getInEdge(1));
    auto& edge_layout     = cast<nncir::DataEdge>(tensor_node.getInEdge(2));
    // Fixme(SRCX): device may also be an input for aten::tensor
    //auto& edge_device     = cast<nncir::DataEdge>(tensor_node.getInEdge(3));
    auto& edge_pin_memory = cast<nncir::DataEdge>(tensor_node.getInEdge(3));
    auto dtype_id      = edge_dtype.getBlobId();
    auto layout_id     = edge_layout.getBlobId();
    auto pin_memory_id = edge_pin_memory.getBlobId();
    auto iv_dtype      = stream_executor.findBlob(dtype_id).second;
    auto iv_layout     = stream_executor.findBlob(layout_id).second;
    auto iv_pin_memory = stream_executor.findBlob(pin_memory_id).second;
    options = options.dtype(iv_dtype.toScalarType());
    options = options.layout(iv_layout.toLayout());
    options = options.pinned_memory(iv_pin_memory.toBool());

    auto self_list = iv_self.toListRef();
    assert(self_list.size() > 0);
    at::Tensor output;
    if (self_list[0].isInt()) {
        auto array_ref = parseIValueArrayRef<int64_t>(self_list);
        output = nnrt::atenTensor(array_ref, options);
    } else if (self_list[0].isDouble()) {
        auto array_ref = parseIValueArrayRef<double>(self_list);
        output = nnrt::atenTensor(array_ref, options);
    } else {
        DLOG(ERROR) << "Unsupported data type to parse ArrayRef.";
    }

    auto& out_edge = cast<nncir::DataEdge>(tensor_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenTo(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten To node";

    auto to_node = cast<nncir::AtenToNode>(op_node);

    auto dtype = convertDTypeToATScalarType(to_node.getDType());
    auto non_blocking = to_node.getNonBlocking();
    auto copy = to_node.getCopy();
    auto optional_memory_format = to_node.getOptionalMemoryFormat();

    auto& input_self = cast<nncir::DataEdge>(to_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();

    auto data_type = stream_executor.findBlob(input_self_blob_id).first;
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

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

void executorAtenTranspose(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Transpose node";

    auto transpose_node = cast<nncir::AtenTransposeNode>(op_node);
    auto dim0 = transpose_node.getDim0();
    auto dim1 = transpose_node.getDim1();

    auto& input_self = cast<nncir::DataEdge>(transpose_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();
    auto dtype = stream_executor.findBlob(input_self_blob_id).first;
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();

    auto output = nnrt::atenTranspose(self_tensor, dim0, dim1);
    auto& out_edge = cast<nncir::DataEdge>(transpose_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenUnsqueeze(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Unsqueeze node";

    auto unsqueeze_node = cast<nncir::AtenUnsqueezeNode>(op_node);
    assert(unsqueeze_node.getNumInputs() == 2);
    auto& input_tensor = cast<nncir::DataEdge>(unsqueeze_node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    at::Tensor output = nnrt::atenUnsqueeze(tensor, unsqueeze_node.getDim());
    auto& out_edge = cast<nncir::DataEdge>(unsqueeze_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenZeros(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "execute Aten Zeros node";

    auto zeros_node = cast<nncir::AtenZerosNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(zeros_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isList());

    auto self_list = iv_self.toListRef();
    // input list -> at::IntArrayRef size, so datatype of elements in list must be int.
    auto array_ref = parseIValueArrayRef<int64_t>(self_list);

    at::TensorOptions options;
    auto& edge_dtype      = cast<nncir::DataEdge>(zeros_node.getInEdge(1));
    auto& edge_layout     = cast<nncir::DataEdge>(zeros_node.getInEdge(2));
    auto& edge_device     = cast<nncir::DataEdge>(zeros_node.getInEdge(3));
    auto& edge_pin_memory = cast<nncir::DataEdge>(zeros_node.getInEdge(4));
    auto dtype_id      = edge_dtype.getBlobId();
    auto layout_id     = edge_layout.getBlobId();
    auto device_id     = edge_device.getBlobId();
    auto pin_memory_id = edge_pin_memory.getBlobId();
    auto iv_dtype      = stream_executor.findBlob(dtype_id).second;
    auto iv_layout     = stream_executor.findBlob(layout_id).second;
    auto iv_device     = stream_executor.findBlob(device_id).second;
    auto iv_pin_memory = stream_executor.findBlob(pin_memory_id).second;
    options = options.dtype(iv_dtype.toScalarType());
    options = options.layout(iv_layout.toLayout());
    options = options.device(iv_device.toDevice());
    options = options.pinned_memory(iv_pin_memory.toBool());

    auto output = nnrt::atenZeros(array_ref, options);
    auto& out_edge = cast<nncir::DataEdge>(zeros_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenZerosLike(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    // TODO to be implemented after fixing converting from at::IValue to TensorOption
}
}  // namespace nnrt
