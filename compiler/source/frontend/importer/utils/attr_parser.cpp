
#include "frontend/importer/utils/attr_parser.h"

using nn_compiler::ir::DTensor;
using nn_compiler::ir::STensor;

namespace nn_compiler
{
namespace frontend
{
int64_t AttrParser::getIntAttr(const torch::jit::Node* node, c10::Symbol symbol)
{
    assert(node->kindOf(symbol) == torch::jit::AttributeKind::i);
    return node->i(symbol);
}

std::vector<int64_t> AttrParser::getIntArrayAttr(const torch::jit::Node* node, c10::Symbol symbol)
{
    assert(node->kindOf(symbol) == torch::jit::AttributeKind::i);
    return node->is(symbol);
}

double AttrParser::getFP64Attr(const torch::jit::Node* node, c10::Symbol symbol)
{
    assert(node->kindOf(symbol) == torch::jit::AttributeKind::f);
    return node->f(symbol);
}

std::string AttrParser::getStrAttr(const torch::jit::Node* node, c10::Symbol symbol)
{
    assert(node->kindOf(symbol) == torch::jit::AttributeKind::s);
    return node->s(symbol);
}

at::Tensor AttrParser::getTensorAttr(const torch::jit::Node* node, c10::Symbol symbol)
{
    assert(node->kindOf(symbol) == torch::jit::AttributeKind::t);
    return node->t(symbol);
}

std::pair<std::vector<at::Tensor>, std::vector<at::Tensor> > AttrParser::getGeneralWeightAndBias(
    const torch::jit::Node* node)
{
    // get weights
    auto weight_node = node->inputs()[1]->node();
    std::vector<at::Tensor> weight_vec;
    if (weight_node->kind() == c10::prim::Constant && weight_node->hasAttribute(c10::attr::value)) {
        auto weight_tensor = weight_node->t(c10::attr::value);
        weight_vec.push_back(weight_tensor);
    }
    // get bias
    auto bias_node = node->inputs()[2]->node();
    std::vector<at::Tensor> bias_vec;
    if (bias_node->kind() == c10::prim::Constant && bias_node->hasAttribute(c10::attr::value)) {
        auto bias_tensor = bias_node->t(c10::attr::value);
        bias_vec.push_back(bias_tensor);
    }

    return std::make_pair(weight_vec, bias_vec);
}

std::pair<std::vector<at::Tensor>, std::vector<at::Tensor> > AttrParser::getLstmWeightAndBias(
    const torch::jit::Node* node)
{
    // the learnable parameter of aten::lstm contains 8 or 12 tensors,
    // they are all prim::Constant and these tensors are input to prim::ListConstrcut
    // which returns a tensor[], instead of to unpack these tensor from tensor[].
    // We firstly get the previous prim::ListConstruct Node and then get all the inputs
    // of prim::ListConstruct
    std::vector<at::Tensor> weight_vec;
    std::vector<at::Tensor> bias_vec;
    // get the prim::ListConstruct
    auto list_construct = node->inputs()[3]->node();
    if (list_construct->kind() != c10::prim::ListConstruct) {
        list_construct = node->inputs()[2]->node();
    }
    // get each input of prim::ListConstruct
    for (auto item : list_construct->inputs()) {
        auto constant_node = item->node();
        assert(constant_node->kind() == c10::prim::Constant);
        assert(constant_node->hasAttribute(c10::attr::value));
        // get the at::Tensor from prim::Constant
        auto torch_tensor = constant_node->t(c10::attr::value);
        if (torch_tensor.dim() == 1) {
            bias_vec.push_back(torch_tensor);
        } else {
            weight_vec.push_back(torch_tensor);
        }
    }

    return std::make_pair(weight_vec, bias_vec);
}

void ptTensor2DTensor(at::Tensor torch_tensor, std::shared_ptr<DTensor> d_tensor)
{
    c10::ScalarType dtype = torch_tensor.scalar_type();
    // dtype of the weights of LSTM are float32 or float16/half
    assert(dtype == c10::ScalarType::Float || dtype == c10::ScalarType::Half);

    // get the data pointer of tensor
    int num_elements = torch_tensor.numel();
    auto tensor_data = torch_tensor.data_ptr();

    // set the data pointer to DTensor
    if (dtype == c10::ScalarType::Float) {
        d_tensor->setData(tensor_data, num_elements * sizeof(float));
        d_tensor->setDataType(nn_compiler::ir::DataType::FLOAT32);
        // for float32, bit_width = 32
        d_tensor->setBitWidth(32);
    } else if (dtype == c10::ScalarType::Half) {
        d_tensor->setData(tensor_data, num_elements * sizeof(float) / 2);  // fp16
        d_tensor->setDataType(nn_compiler::ir::DataType::FLOAT16);
        // for float32, bit_width = 16
        d_tensor->setBitWidth(16);
    }

    // get stride of tensor
    auto strides = torch_tensor.strides().vec();

    // set the shape for DTensor
    if (torch_tensor.sizes().size() == 4) {
        d_tensor->setStride(strides);
        d_tensor->setTensorShape(
            STensor(torch_tensor.size(0), torch_tensor.size(1), torch_tensor.size(2), torch_tensor.size(3)));
    } else if (torch_tensor.sizes().size() == 3) {
        d_tensor->setStride({0, strides[0], strides[1], strides[2]});
        d_tensor->setTensorShape(STensor(0, torch_tensor.size(0), torch_tensor.size(1), torch_tensor.size(2)));
    } else if (torch_tensor.sizes().size() == 2) {
        d_tensor->setStride({0, 0, strides[0], strides[1]});
        d_tensor->setTensorShape(STensor(0, 0, torch_tensor.size(0), torch_tensor.size(1)));
    } else if (torch_tensor.sizes().size() == 1) {
        d_tensor->setStride({0, 0, 0, strides[0]});
        d_tensor->setTensorShape(STensor(0, 0, 0, torch_tensor.size(0)));
    }
}

void showString(std::shared_ptr<DTensor> dt)
{
    std::vector<uint8_t> vv = *(dt->getData<uint8_t>());
    DLOG(INFO) << "str: " << vv.data();
}
void showInt64(std::shared_ptr<DTensor> dt)
{
    std::vector<int64_t> vv = *(dt->getData<int64_t>());
    long int res = *vv.data();
    DLOG(INFO) << "int64: " << res;
}
void showFloat64(std::shared_ptr<DTensor> dt)
{
    std::vector<double> vv = *(dt->getData<double>());
    double res = *vv.data();
    DLOG(INFO) << "float64: " << res;
}

std::shared_ptr<DTensor> getDTensorData(const torch::jit::Node* node_constant)
{
    auto data = std::make_shared<DTensor>();
    data->setDataType(nn_compiler::ir::DataType::UNDEFINED);

    std::string ntype = node_constant->output()->type()->str();
    if (ntype == "str") {
        auto attr_symbol = node_constant->attributeNames().at(0);
        auto value = node_constant->s(attr_symbol);
        auto len = value.length();
        data->setData(value.c_str(), len + 1);
        data->setTensorShape(STensor(0, 0, 0, len));
        data->setDataType(nn_compiler::ir::DataType::UINT8);
        DLOG(INFO) << "set str: " << value;

    } else if (ntype == "Device") {
        auto attr_symbol = node_constant->attributeNames().at(0);
        auto value = node_constant->s(attr_symbol);
        auto len = value.length();
        data->setData(value.c_str(), len + 1);
        data->setTensorShape(STensor(0, 0, 0, len));
        data->setDataType(nn_compiler::ir::DataType::UINT8);
        DLOG(INFO) << "set Device: " << value;

    } else if (ntype == "int") {
        auto attr_symbol = node_constant->attributeNames().at(0);
        auto value = node_constant->i(attr_symbol);
        data->setData(&value, 8);
        data->setTensorShape(STensor(0, 0, 0, 1));
        data->setDataType(nn_compiler::ir::DataType::INT64);
        DLOG(INFO) << "set int64: " << value;

    } else if (ntype == "bool") {
        auto attr_symbol = node_constant->attributeNames().at(0);
        auto value = node_constant->i(attr_symbol);
        data->setData(&value, 8);
        data->setTensorShape(STensor(0, 0, 0, 1));
        data->setDataType(nn_compiler::ir::DataType::INT64);
        DLOG(INFO) << "set bool as int64: " << value;

    } else if (ntype == "Tensor") {
        auto attr_symbol = node_constant->attributeNames().at(0);
        auto torch_tensor = node_constant->t(attr_symbol);
        c10::ScalarType dtype = torch_tensor.scalar_type();
        ptTensor2DTensor(torch_tensor, data);
        DLOG(INFO) << "set Tensor with size: " << torch_tensor.sizes() << " dtype: " << dtype;

    } else if (ntype == "float") {
        auto attr_symbol = node_constant->attributeNames().at(0);
        auto value = node_constant->f(attr_symbol);
        data->setData(&value, 1 * sizeof(value));
        data->setTensorShape(STensor(0, 0, 0, 1));
        data->setDataType(nn_compiler::ir::DataType::FLOAT64);
        DLOG(INFO) << "set float64: " << value;

    } else if (ntype.find("None") != std::string::npos) {
        data->setDataType(nn_compiler::ir::DataType::UINT8);
        DLOG(INFO) << "set None: " << ntype;

    } else if (ntype.find("(") != std::string::npos && ntype.find(")") != std::string::npos) {  // tuple type
        // presupposition: same datatype in a single tuple

        auto parseNtype = [](const std::string& ntype) -> std::pair<int, nn_compiler::ir::DataType> {
            auto element_type = nn_compiler::ir::DataType::UNDEFINED;
            int element_num = 0;
            if (ntype.find("int") != std::string::npos) {
                for (int i = 0; (i = ntype.find("int", i)) != std::string::npos; element_num++, i += 3)
                    ;
                element_type = nn_compiler::ir::DataType::INT64;
            } else if (ntype.find("float") != std::string::npos) {
                for (int i = 0; (i = ntype.find("float", i)) != std::string::npos; element_num++, i += 5)
                    ;
                element_type = nn_compiler::ir::DataType::FLOAT64;
            } else if (ntype.find("bool") != std::string::npos) {
                for (int i = 0; (i = ntype.find("bool", i)) != std::string::npos; element_num++, i += 4)
                    ;
                element_type = nn_compiler::ir::DataType::INT64;
            } else {
                DLOG(FATAL) << "unspported datatype for tuple.";
            }
            return std::make_pair(element_num, element_type);
        };

        auto attr_symbol = c10::attr::value;
        auto value_vec = node_constant->ival(attr_symbol).toTuple()->elements();
        auto value = value_vec.data();
        auto parsed_ntype = parseNtype(ntype);
        data->setData(value, parsed_ntype.first * 8);
        data->setTensorShape(STensor(0, 0, 0, parsed_ntype.first));
        data->setDataType(parsed_ntype.second);
        DLOG(INFO) << "set tuple:" << value_vec;
    } else if (ntype.find("[") != std::string::npos && ntype.find("]") != std::string::npos) {  // list type
        auto attr_symbol = c10::attr::value;
        auto value_list = node_constant->ival(attr_symbol).toList();
        auto value_vec = value_list.vec();
        if (ntype.find("int") != std::string::npos) {
            std::vector<int64_t> value;
            for (auto item : value_vec) {
                value.push_back(item.toInt());
            }
            data->setData(value.data(), value.size() * 8);
            data->setTensorShape(STensor(0, 0, 0, value.size()));
            data->setDataType(nn_compiler::ir::DataType::INT64);
        } else if (ntype.find("float") != std::string::npos) {
            std::vector<double> value;
            for (auto item : value_vec) {
                value.push_back(item.toDouble());
            }
            data->setData(value.data(), value.size() * 8);
            data->setTensorShape(STensor(0, 0, 0, value.size()));
            data->setDataType(nn_compiler::ir::DataType::FLOAT64);
        } else {
            DLOG(FATAL) << "unspported datatype for list.";
        }
        DLOG(INFO) << "set list:" << value_vec;
    } else {
        DLOG(FATAL) << "not supported type:" << ntype;
        return nullptr;
    }
    return data;
}

}  // namespace frontend
}  // namespace nn_compiler
