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
#include "ir/include/common/utils.hpp"


namespace nnrt
{
void executorAtenAdd(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Add node";

    auto add_node = cast<nncir::AtenAddNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(add_node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(add_node.getInEdge(1));
    int64_t alpha = add_node.getAlpha();
    if (nncir::isDefaultValue<int64_t>(alpha)) {
        auto& alpha_edge = cast<nncir::DataEdge>(add_node.getInEdge(2));
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

void executorAtenAddmm(const nncir::Node& op_node, StreamExecutor& stream_executor) {
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

    auto output = nnrt::atenAddmm(iv_self.toTensor(), iv_mat1.toTensor(), iv_mat2.toTensor(),
                                  iv_beta.toScalar(), iv_alpha.toScalar());
    // update output
    auto& out_edge = cast<nncir::DataEdge>(addmm_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
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
    if (nncir::isDefaultValue<int64_t>(dim)) {
        auto& dim_edge = cast<nncir::DataEdge>(cat_node.getInEdge(1));
        int dim_blob_id = dim_edge.getBlobId();
        auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }

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

void executorAtenDeriveIndex(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "execute Aten derive_index node";

    auto derive_index_node = cast<nncir::AtenDeriveIndexNode>(op_node);
    auto index   = derive_index_node.getIndex();
    auto step    = derive_index_node.getStep();
    auto inedges = derive_index_node.getInEdgeIds();
    int edge_id = 0;
    auto& data_edge = cast<nncir::DataEdge>(derive_index_node.getInEdge(edge_id));
    auto input_blob_id    = data_edge.getBlobId();
    torch::jit::IValue iv = stream_executor.findBlob(input_blob_id).second;
    auto start = iv.toInt();
    edge_id++;

    if (nncir::isDefaultValue<int64_t>(index)) {
        auto& index_data_edge = cast<nncir::DataEdge>(derive_index_node.getInEdge(edge_id));
        input_blob_id = index_data_edge.getBlobId();
        iv = stream_executor.findBlob(input_blob_id).second;
        index = iv.toInt();
        edge_id++;
    }
    if (nncir::isDefaultValue<int64_t>(step)) {
        auto& step_data_edge = cast<nncir::DataEdge>(derive_index_node.getInEdge(edge_id));
        input_blob_id = step_data_edge.getBlobId();
        iv = stream_executor.findBlob(input_blob_id).second;
        step = iv.toInt();
        edge_id++;
    }

    auto output    = nnrt::atenDeriveIndex(start, index, step);
    auto& out_edge = cast<nncir::DataEdge>(derive_index_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, scalarToIValue(output));
}


void executorAtenCopy(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "execute Aten Copy node";
    auto copy_node = cast<nncir::AtenCopyNode>(op_node);
    // assert(copy_node.getNumInputs() == 2);

    auto &input_self = cast<nncir::DataEdge>(copy_node.getInEdge(0));
    auto &input_src = cast<nncir::DataEdge>(copy_node.getInEdge(1));
    int input_self_blob_id = input_self.getBlobId();
    int input_src_blob_id = input_src.getBlobId();

    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    torch::jit::IValue iv_src = stream_executor.findBlob(input_src_blob_id).second;
    at::Tensor self_tensor = iv_self.toTensor();
    at::Tensor src_tensor = iv_self.toTensor();

    int non_blocking = copy_node.getNonBlocking();
    if (nncir::isDefaultValue<int>(non_blocking)) {
        auto& non_blocking_data_edge = cast<nncir::DataEdge>(copy_node.getInEdge(2));
        int non_blocking_blob_id = non_blocking_data_edge.getBlobId();
        auto non_blocking_iv = stream_executor.findBlob(non_blocking_blob_id).second;
        non_blocking = non_blocking_iv.toInt();
    }

    auto output = nnrt::atenCopy_(self_tensor, src_tensor, static_cast<bool>(non_blocking));
    // update output
    auto &out_edge = cast<nncir::DataEdge>(copy_node.getFirstOutEdge());
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
    if (nncir::isDefaultValue<double>(proportion)) {
        auto& proportion_edge = cast<nncir::DataEdge>(dropout_node.getInEdge(edge_id++));
        int proportion_blob_id = proportion_edge.getBlobId();
        auto proportion_iv = stream_executor.findBlob(proportion_blob_id).second;
        assert(proportion_iv.isDouble());
        proportion = proportion_iv.toDouble();
    }
    int train_val = dropout_node.getTrain();
    if (nncir::isDefaultValue<int>(train_val)) {
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
    if (nncir::isDefaultValue<int64_t>(padding_idx)) {
        auto& padding_idx_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int padding_idx_blob_id = padding_idx_edge.getBlobId();
        auto padding_idx_iv = stream_executor.findBlob(padding_idx_blob_id).second;
        assert(padding_idx_iv.isInt());
        padding_idx = padding_idx_iv.toInt();
    }

    int scale_grad_by_freq_val = node.getScaleGradByFreq();
    if (nncir::isDefaultValue<int>(scale_grad_by_freq_val)) {
        auto& scale_grad_by_freq_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int scale_grad_by_freq_blob_id = scale_grad_by_freq_edge.getBlobId();
        auto scale_grad_by_freq_iv = stream_executor.findBlob(scale_grad_by_freq_blob_id).second;
        assert(scale_grad_by_freq_iv.isBool());
        scale_grad_by_freq_val = scale_grad_by_freq_iv.toBool();
    }
    bool scale_grad_by_freq = static_cast<bool>(scale_grad_by_freq_val);

    int sparse_val = node.getSparse();
    if (nncir::isDefaultValue<int>(sparse_val)) {
        auto& sparse_edge = cast<nncir::DataEdge>(node.getInEdge(edge_id++));
        int sparse_blob_id = sparse_edge.getBlobId();
        auto sparse_iv = stream_executor.findBlob(sparse_blob_id).second;
        assert(sparse_iv.isBool());
        sparse_val = sparse_iv.toBool();
    }
    bool sparse = static_cast<bool>(sparse_val);

    auto output = nnrt::atenEmbedding(iv_weights.toTensor(), iv_indices.toTensor(),
                                      padding_idx, scale_grad_by_freq, sparse);

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
    auto size_list = parseIValueArrayRef<int64_t>(size_ivalue_list);

    int implicit = expand_node.getImplicit();
    if (nncir::isDefaultValue<int>(implicit)) {
        auto& implicit_edge = cast<nncir::DataEdge>(expand_node.getInEdge(2));
        int implicit_blob_id = implicit_edge.getBlobId();
        auto implicit_iv = stream_executor.findBlob(implicit_blob_id).second;
        assert(implicit_iv.isInt());
        implicit = implicit_iv.toInt();
    }

    auto output = nnrt::atenExpand(self_tensor, size_list, static_cast<bool>(implicit));
    // update output
    auto& out_edge = cast<nncir::DataEdge>(expand_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));
}

void executorAtenFormat(const nncir::Node& op_node, StreamExecutor& stream_executor) {
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

    auto &input1 = cast<nncir::DataEdge>(format_node.getInEdge(edge_id++));
    auto &input2 = cast<nncir::DataEdge>(format_node.getInEdge(edge_id++));

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

    int tmp_self = iv_self.toScalar().toInt();
    // DLOG(INFO) <<"AtenGt: "<<"iv_self:"<<iv_self.toScalar();
    DLOG(INFO) <<"AtenGt: "<<"iv_other:"<<iv_other.toInt();
    DLOG(INFO) <<"AtenGt:" <<"tmp_self:"<<tmp_self;

    int64_t output = tmp_self > iv_other.toInt();
    DLOG(INFO) <<"AtenGt:" <<"output:"<<output;

    auto& out_edge = cast<nncir::DataEdge>(node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, scalarToIValue<int64_t>(output));

#if 0
    // assert(iv_self.isTensor());
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

#endif // 0
}

void executorAtenGetItem(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "execute Aten GetItem node";

    auto get_item_node = cast<nncir::AtenGetItemNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(get_item_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();

    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isList());
    auto self_list = iv_self.toList();

    int idx = get_item_node.getIdx();
    if (nncir::isDefaultValue<int>(idx)) {
        auto& idx_edge = cast<nncir::DataEdge>(get_item_node.getInEdge(1));
        int idx_blob_id = idx_edge.getBlobId();
        auto idx_iv = stream_executor.findBlob(idx_blob_id).second;
        assert(idx_iv.isInt());
        idx = idx_iv.toInt();
    }

    auto output = nnrt::atenGetItem(self_list, idx);
    // update output
    auto& out_edge = cast<nncir::DataEdge>(get_item_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::IVALUE, output);
}

void executorAtenInt(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Int node";

    auto int_node = cast<nncir::AtenIntNode>(op_node);
    assert(int_node.getNumInputs() == 1);

    auto& input_self = cast<nncir::DataEdge>(int_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();

    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isScalar());
    auto self_scalar = iv_self.toScalar();

    auto output = nnrt::atenInt(self_scalar);
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

void executorAtenLSTM(const nncir::Node& op_node, StreamExecutor& stream_executor) {
    DLOG(INFO) << "execute Aten LSTM node";

    auto lstm_node = cast<nncir::AtenLSTMNode>(op_node);
    int edge_idx = 0;

    // const at::Tensor &input
    auto& input_edge = cast<nncir::DataEdge>(lstm_node.getInEdge(0));
    int input_blob_id = input_edge.getBlobId();
    auto input_iv = stream_executor.findBlob(input_blob_id).second;
    assert(input_iv.isTensor());
    auto input = input_iv.toTensor();
    edge_idx++;

    // at::TensorList hx
    auto& hx_edge = cast<nncir::DataEdge>(lstm_node.getInEdge(1));
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
    // Skip for now, will handle params after getting all arguments
    edge_idx++;

    // bool has_biases
    int has_biases = lstm_node.getHasBiases();
    if (nncir::isDefaultValue<int>(has_biases)) {
        auto& has_biases_edge = cast<nncir::DataEdge>(lstm_node.getInEdge(edge_idx));
        int has_biases_blob_id = has_biases_edge.getBlobId();
        auto has_biases_iv = stream_executor.findBlob(has_biases_blob_id).second;
        assert(has_biases_iv.isInt());
        has_biases = has_biases_iv.toInt();
        edge_idx++;
    }

    // int64_t num_layers
    int64_t num_layers = lstm_node.getNumLayers();
    if (nncir::isDefaultValue<int64_t>(num_layers)) {
        auto& num_layers_edge = cast<nncir::DataEdge>(lstm_node.getInEdge(edge_idx));
        int num_layers_blob_id = num_layers_edge.getBlobId();
        auto num_layers_iv = stream_executor.findBlob(num_layers_blob_id).second;
        assert(num_layers_iv.isInt());
        num_layers = num_layers_iv.toInt();
        edge_idx++;
    }

    // double dropout
    double dropout = lstm_node.getDropout();
    if (nncir::isDefaultValue<double>(dropout)) {
        auto& dropout_edge = cast<nncir::DataEdge>(lstm_node.getInEdge(edge_idx));
        int dropout_blob_id = dropout_edge.getBlobId();
        auto dropout_iv = stream_executor.findBlob(dropout_blob_id).second;
        assert(dropout_iv.isDouble());
        dropout = dropout_iv.toDouble();
        edge_idx++;
    }

    // bool train
    int train = lstm_node.getTrain();
    if (nncir::isDefaultValue<int>(train)) {
        auto& train_edge = cast<nncir::DataEdge>(lstm_node.getInEdge(edge_idx));
        int train_blob_id = train_edge.getBlobId();
        auto train_iv = stream_executor.findBlob(train_blob_id).second;
        assert(train_iv.isInt());
        train = train_iv.toInt();
        edge_idx++;
    }

    // bool bidirectional
    int bidirectional = lstm_node.getBidirectional();
    if (nncir::isDefaultValue<int>(bidirectional)) {
        auto& bidirectional_edge = cast<nncir::DataEdge>(lstm_node.getInEdge(edge_idx));
        int bidirectional_blob_id = bidirectional_edge.getBlobId();
        auto bidirectional_iv = stream_executor.findBlob(bidirectional_blob_id).second;
        assert(bidirectional_iv.isInt());
        bidirectional = bidirectional_iv.toInt();
        edge_idx++;
    }

    // bool batch_first
    int batch_first = lstm_node.getBatchFirst();
    if (nncir::isDefaultValue<int>(batch_first)) {
        auto& batch_first_edge = cast<nncir::DataEdge>(lstm_node.getInEdge(edge_idx));
        int batch_first_blob_id = batch_first_edge.getBlobId();
        auto batch_first_iv = stream_executor.findBlob(batch_first_blob_id).second;
        assert(batch_first_iv.isInt());
        batch_first = batch_first_iv.toInt();
        edge_idx++;
    }

    // at::TensorList params
    // param layerout --> (w_ih, w_hh, b_ih?, b_hh?) * layers
    auto weight_blob_ids = lstm_node.getWeightBlobId();
    auto bias_blob_ids = lstm_node.getBiasBlobId();
    std::vector<at::Tensor> param_vector;
    for (int i = 0; i < num_layers; i++) {
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
        nnrt::atenLstm(input, hx, params, static_cast<bool>(has_biases), num_layers, dropout, static_cast<bool>(train),
                       static_cast<bool>(bidirectional), static_cast<bool>(batch_first));
    // update output
    auto& out_edge = cast<nncir::DataEdge>(lstm_node.getFirstOutEdge());
    stream_executor.updateBlob(out_edge.getBlobId(), DataType::TUPLE, tupleToIValue(output));
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
    int edge_id = 0;

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
    } else if (isScalarType(dtype)) {
        // aten::max(Tensor, dim, keepdim)
        edge_id++;
        auto dim = max_node.getDim();
        if (nncir::isDefaultValue<int64_t>(dim)) {
            auto& dim_edge = cast<nncir::DataEdge>(max_node.getInEdge(edge_id++));
            int dim_blob_id = dim_edge.getBlobId();
            auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
            assert(dim_iv.isInt());
            dim = dim_iv.toInt();
        }
        int keep_dim = max_node.getKeepDim();
        if (nncir::isDefaultValue<int>(keep_dim)) {
            auto& keep_dim_edge = cast<nncir::DataEdge>(max_node.getInEdge(edge_id++));
            int keep_dim_blob_id = keep_dim_edge.getBlobId();
            auto keep_dim_iv = stream_executor.findBlob(keep_dim_blob_id).second;
            assert(keep_dim_iv.isBool());
            keep_dim = keep_dim_iv.toBool();
        }

        auto output = nnrt::atenMax(self_tensor, dim, static_cast<bool>(keep_dim));
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

    if (iv_self.isIntList() && iv_other.isIntList()){
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
    torch::jit::IValue iv = stream_executor.findBlob(input_tensor_blob_id).second;
    auto dtype = stream_executor.findBlob(input_tensor_blob_id).first;
    auto& out_edge = cast<nncir::DataEdge>(neg_node.getFirstOutEdge());

    if(isScalarType(dtype)) {
        if(iv.isInt()) {
            int out = iv.toInt() * -1;
            stream_executor.updateBlob(out_edge.getBlobId(), DataType::INT64, scalarToIValue<int>(out));
        }else if(iv.isDouble()) {
            double out = iv.toDouble() * -1;
            stream_executor.updateBlob(out_edge.getBlobId(), DataType::FLOAT64, scalarToIValue<double>(out));
        }
    } else if (iv.isTensor()){
        // assert(iv.isTensor());
        at::Tensor tensor = iv.toTensor();
        auto output = nnrt::atenNeg(tensor);
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::TENSOR, tensorToIValue(output));

    } else {
        DLOG(ERROR) <<"AtenNeg: unsupported dtype!";
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
    if (nncir::isDefaultValue<int64_t>(dim)) {
        auto& dim_edge = cast<nncir::DataEdge>(select_node.getInEdge(edge_id++));
        int dim_blob_id = dim_edge.getBlobId();
        auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }
    auto index = select_node.getIndex();
    if (nncir::isDefaultValue<int64_t>(index)) {
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
    if (inedges_cnt == 1 && nncir::isDefaultValue<int64_t>(dim)) {
        auto output = nnrt::atenSize(tensor);
        auto& out_edge = cast<nncir::DataEdge>(size_node.getFirstOutEdge());
        stream_executor.updateBlob(out_edge.getBlobId(), DataType::LIST, torch::jit::IValue(output));
    } else {
        if (nncir::isDefaultValue<int64_t>(dim)) {
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
    if (nncir::isDefaultValue<int64_t>(dim)) {
        auto& dim_edge = cast<nncir::DataEdge>(slice_node.getInEdge(edge_id++));
        int dim_blob_id = dim_edge.getBlobId();
        auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }
    auto start = slice_node.getStart();
    if (nncir::isDefaultValue<int64_t>(start)) {
        auto& start_edge = cast<nncir::DataEdge>(slice_node.getInEdge(edge_id++));
        int start_blob_id = start_edge.getBlobId();
        auto start_iv = stream_executor.findBlob(start_blob_id).second;
        assert(start_iv.isInt());
        start = start_iv.toInt();
    }
    auto end = slice_node.getEnd();
    if (nncir::isDefaultValue<int64_t>(end)) {
        auto& end_edge = cast<nncir::DataEdge>(slice_node.getInEdge(edge_id++));
        int end_blob_id = end_edge.getBlobId();
        auto end_iv = stream_executor.findBlob(end_blob_id).second;
        assert(end_iv.isInt());
        end = end_iv.toInt();
    }
    auto step = slice_node.getStep();
    if (nncir::isDefaultValue<int64_t>(step)) {
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

void executorAtenSub(const nncir::Node& op_node, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute Aten Sub node";

    auto sub_node = cast<nncir::AtenSubNode>(op_node);

    auto& input_self = cast<nncir::DataEdge>(sub_node.getInEdge(0));
    auto& input_other = cast<nncir::DataEdge>(sub_node.getInEdge(1));
    int64_t alpha = sub_node.getAlpha();
    if (nncir::isDefaultValue<int64_t>(alpha)) {
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

    if (!iv_dtype.isNone()) {
        options = options.dtype(iv_dtype.toScalarType());
    }
    if (!iv_layout.isNone()) {
        options = options.layout(iv_layout.toLayout());
    }
    if (!iv_pin_memory.isNone()) {
        options = options.pinned_memory(iv_pin_memory.toBool());
    }
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
    int edge_id = 0;

    auto& input_self = cast<nncir::DataEdge>(to_node.getInEdge(0));
    int input_self_blob_id = input_self.getBlobId();

    auto data_type = stream_executor.findBlob(input_self_blob_id).first;
    torch::jit::IValue iv_self = stream_executor.findBlob(input_self_blob_id).second;
    assert(iv_self.isTensor());
    at::Tensor self_tensor = iv_self.toTensor();
    edge_id++;

    auto ori_dtype = to_node.getDType();
    if (nncir::isDefaultValue<int64_t>(ori_dtype)) {
        auto& ori_dtype_edge = cast<nncir::DataEdge>(to_node.getInEdge(edge_id++));
        int ori_dtype_blob_id = ori_dtype_edge.getBlobId();
        auto ori_dtype_iv = stream_executor.findBlob(ori_dtype_blob_id).second;
        assert(ori_dtype_iv.isInt());
        ori_dtype = ori_dtype_iv.toInt();
    }
    auto dtype = convertDTypeToATScalarType(static_cast<nncir::DataType>(ori_dtype));

    int non_blocking_val = to_node.getNonBlocking();
    if (nncir::isDefaultValue<int>(non_blocking_val)) {
        auto& non_blocking_edge = cast<nncir::DataEdge>(to_node.getInEdge(edge_id++));
        int non_blocking_blob_id = non_blocking_edge.getBlobId();
        auto non_blocking_iv = stream_executor.findBlob(non_blocking_blob_id).second;
        assert(non_blocking_iv.isBool());
        non_blocking_val = non_blocking_iv.toBool();
    }
    bool non_blocking = static_cast<bool>(non_blocking_val);

    int copy_val = to_node.getCopy();
    if (nncir::isDefaultValue<int>(copy_val)) {
        auto& copy_edge = cast<nncir::DataEdge>(to_node.getInEdge(edge_id++));
        int copy_blob_id = copy_edge.getBlobId();
        auto copy_iv = stream_executor.findBlob(copy_blob_id).second;
        assert(copy_iv.isBool());
        copy_val = copy_iv.toBool();
    }
    bool copy = static_cast<bool>(copy_val);

    auto optional_memory_format = to_node.getOptionalMemoryFormat();
    if (nncir::isDefaultValue<int>(optional_memory_format)) {
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
    if (nncir::isDefaultValue<int64_t>(dim0)) {
        auto& dim0_edge = cast<nncir::DataEdge>(transpose_node.getInEdge(edge_id++));
        int dim0_blob_id = dim0_edge.getBlobId();
        auto dim0_iv = stream_executor.findBlob(dim0_blob_id).second;
        assert(dim0_iv.isInt());
        dim0 = dim0_iv.toInt();
    }
    auto dim1 = transpose_node.getDim1();
    if (nncir::isDefaultValue<int64_t>(dim1)) {
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
    assert(unsqueeze_node.getNumInputs() == 2);
    auto& input_tensor = cast<nncir::DataEdge>(unsqueeze_node.getInEdge(0));
    int input_tensor_blob_id = input_tensor.getBlobId();
    torch::jit::IValue iv_tensor = stream_executor.findBlob(input_tensor_blob_id).second;
    assert(iv_tensor.isTensor());
    at::Tensor tensor = iv_tensor.toTensor();

    auto dim = unsqueeze_node.getDim();
    if (nncir::isDefaultValue<int64_t>(dim)) {
        auto& dim0_edge = cast<nncir::DataEdge>(unsqueeze_node.getInEdge(1));
        int dim_blob_id = dim0_edge.getBlobId();
        auto dim_iv = stream_executor.findBlob(dim_blob_id).second;
        assert(dim_iv.isInt());
        dim = dim_iv.toInt();
    }

    at::Tensor output = nnrt::atenUnsqueeze(tensor, dim);
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
    auto array_ref = parseIValueVector<int64_t>(self_list);

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
    auto& edge_dtype      = cast<nncir::DataEdge>(zeros_like_node.getInEdge(1));
    auto& edge_layout     = cast<nncir::DataEdge>(zeros_like_node.getInEdge(2));
    auto& edge_device     = cast<nncir::DataEdge>(zeros_like_node.getInEdge(3));
    auto& edge_pin_memory = cast<nncir::DataEdge>(zeros_like_node.getInEdge(4));
    auto dtype_id      = edge_dtype.getBlobId();
    auto layout_id     = edge_layout.getBlobId();
    auto device_id     = edge_device.getBlobId();
    auto pin_memory_id = edge_pin_memory.getBlobId();
    auto iv_dtype      = stream_executor.findBlob(dtype_id).second;
    auto iv_layout     = stream_executor.findBlob(layout_id).second;
    auto iv_device     = stream_executor.findBlob(device_id).second;
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
    auto memory_format_id  = input_memory_format.getBlobId();
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
}  // namespace nnrt
