
#include "importer/utils/attr_parser.h"

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
    Log::IR::I() << "str: " << vv.data();
}
void showInt64(std::shared_ptr<DTensor> dt)
{
    std::vector<int64_t> vv = *(dt->getData<int64_t>());
    long int res = *vv.data();
    Log::IR::I() << "int64: " << res;
}
void showFloat64(std::shared_ptr<DTensor> dt)
{
    std::vector<double> vv = *(dt->getData<double>());
    double res = *vv.data();
    Log::IR::I() << "float64: " << res;
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
        Log::IR::I() << "set str: " << value;

    } else if (ntype == "Device") {
        auto attr_symbol = node_constant->attributeNames().at(0);
        auto value = node_constant->s(attr_symbol);
        auto len = value.length();
        data->setData(value.c_str(), len + 1);
        data->setTensorShape(STensor(0, 0, 0, len));
        data->setDataType(nn_compiler::ir::DataType::UINT8);
        Log::IR::I() << "set Device: " << value;

    } else if (ntype == "int") {
        auto attr_symbol = node_constant->attributeNames().at(0);
        auto value = node_constant->i(attr_symbol);
        data->setData(&value, 8);
        data->setTensorShape(STensor(0, 0, 0, 1));
        data->setDataType(nn_compiler::ir::DataType::INT64);
        Log::IR::I() << "set int64: " << value;

    } else if (ntype == "bool") {
        auto attr_symbol = node_constant->attributeNames().at(0);
        auto value = node_constant->i(attr_symbol);
        data->setData(&value, 8);
        data->setTensorShape(STensor(0, 0, 0, 1));
        data->setDataType(nn_compiler::ir::DataType::INT64);
        Log::IR::I() << "set bool as int64: " << value;

    } else if (ntype == "Tensor") {
        auto attr_symbol = node_constant->attributeNames().at(0);
        auto torch_tensor = node_constant->t(attr_symbol);
        c10::ScalarType dtype = torch_tensor.scalar_type();
        ptTensor2DTensor(torch_tensor, data);
        Log::IR::I() << "set Tensor with size: " << torch_tensor.sizes() << " dtype: " << dtype;

    } else if (ntype == "float") {
        auto attr_symbol = node_constant->attributeNames().at(0);
        auto value = node_constant->f(attr_symbol);
        data->setData(&value, 1 * sizeof(value));
        data->setTensorShape(STensor(0, 0, 0, 1));
        data->setDataType(nn_compiler::ir::DataType::FLOAT64);
        Log::IR::I() << "set float64: " << value;

    } else if (ntype.find("None") != std::string::npos) {
        data->setDataType(nn_compiler::ir::DataType::UINT8);
        Log::IR::I() << "set None: " << ntype;

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
                Log::IR::E() << "unspported datatype for tuple.";
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
        Log::IR::I() << "set tuple:" << value_vec;
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
            Log::IR::E() << "unspported datatype for list.";
        }
        Log::IR::I() << "set list:" << value_vec;
    } else {
        Log::IR::E() << "not supported type:" << ntype;
        return nullptr;
    }
    return data;
}

}  // namespace frontend
}  // namespace nn_compiler
