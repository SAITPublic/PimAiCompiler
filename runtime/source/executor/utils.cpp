#include "executor/utils.h"
#include <torch/script.h>
#include <ostream>
#include <string>
#include <vector>
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/nn_ir.hpp"
#include "nnrt_types.h"

namespace nnrt
{
namespace nncir = nn_compiler::nn_ir;

torch::jit::IValue tensorToIValue(const torch::Tensor& tensor) { return torch::jit::IValue(tensor); }

torch::jit::IValue tensorListToIValue(const torch::TensorList& tensor_list) { return torch::jit::IValue(tensor_list); }

torch::jit::IValue strToIValue(std::string str) { return torch::jit::IValue(str); }

at::ScalarType convertDTypeToATScalarType(nnrt::DataType dtype)
{
    // according to: pytorch/c10/core/ScalarType.h
    at::ScalarType scalar_type = at::ScalarType::Byte;
    switch (dtype) {
        case nnrt::DataType::UINT8:
            scalar_type = at::ScalarType::Byte;
            break;
        case nnrt::DataType::INT8:
            scalar_type = at::ScalarType::Char;
            break;
        case nnrt::DataType::INT16:
            scalar_type = at::ScalarType::Short;
            break;
        case nnrt::DataType::INT32:
            scalar_type = at::ScalarType::Int;
            break;
        case nnrt::DataType::INT64:
            scalar_type = at::ScalarType::Long;
            break;
        case nnrt::DataType::FLOAT16:
            scalar_type = at::ScalarType::Half;
            break;
        case nnrt::DataType::FLOAT32:
            scalar_type = at::ScalarType::Float;
            break;
        case nnrt::DataType::FLOAT64:
            scalar_type = at::ScalarType::Double;
            break;
        case nnrt::DataType::BOOL:
            scalar_type = at::ScalarType::Bool;
            break;
        default:
            DLOG(FATAL) << "Complex type has not been supported.";
    }
    return scalar_type;
}

nnrt::DataType convertATScalarTypeToDType(at::ScalarType dtype)
{
    // according to: pytorch/c10/core/ScalarType.h
    nnrt::DataType data_type = nnrt::DataType::NONE;
    switch (dtype) {
        case at::ScalarType::Byte:
            data_type = nnrt::DataType::UINT8;
            break;
        case at::ScalarType::Char:
            data_type = nnrt::DataType::INT8;
            break;
        case at::ScalarType::Short:
            data_type = nnrt::DataType::INT16;
            break;
        case at::ScalarType::Int:
            data_type = nnrt::DataType::INT32;
            break;
        case at::ScalarType::Long:
            data_type = nnrt::DataType::INT64;
            break;
        case at::ScalarType::Half:
            data_type = nnrt::DataType::FLOAT16;
            break;
        case at::ScalarType::Float:
            data_type = nnrt::DataType::FLOAT32;
            break;
        case at::ScalarType::Double:
            data_type = nnrt::DataType::FLOAT64;
            break;
        case at::ScalarType::Bool:
            data_type = nnrt::DataType::BOOL;
            break;
        default:
            DLOG(FATAL) << "Complex type has not been supported.";
    }
    return data_type;
}

at::Device convertIntToATDevice(const int& value) {
    // according to: pytorch/c10/core/DeviceType.h
    at::Device device(at::DeviceType::CPU);
    switch (value) {
        case 0:
            device = at::Device(at::DeviceType::CPU);
            break;
        case 1:
            device = at::Device(at::DeviceType::CUDA);
            break;
        case 2:
            device = at::Device(at::DeviceType::MKLDNN);
            break;
        case 3:
            device = at::Device(at::DeviceType::OPENGL);
            break;
        case 4:
            device = at::Device(at::DeviceType::OPENCL);
            break;
        case 5:
            device = at::Device(at::DeviceType::IDEEP);
            break;
        case 6:
            device = at::Device(at::DeviceType::HIP);
            break;
        case 7:
            device = at::Device(at::DeviceType::FPGA);
            break;
        case 8:
            device = at::Device(at::DeviceType::MSNPU);
            break;
        case 9:
            device = at::Device(at::DeviceType::XLA);
            break;
        case 10:
            device = at::Device(at::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
            break;
        default:
            DLOG(FATAL) << "Unsupported device type.";
    }
    return device;
}

at::MemoryFormat getMemoryFormat(int optional_memory_format)
{
    // according to: pytorch/c10/core/MemoryFormat.h
    at::MemoryFormat memory_format = at::MemoryFormat::Contiguous;
    switch (optional_memory_format) {
        case 0:
            memory_format = at::MemoryFormat::Contiguous;
            break;
        case 1:
            memory_format = at::MemoryFormat::Preserve;
            break;
        case 2:
            memory_format = at::MemoryFormat::ChannelsLast;
            break;
        case 3:
            memory_format = at::MemoryFormat::ChannelsLast3d;
            break;
        default:
            DLOG(FATAL) << "Wrong reference to aten MemoryFormat.";
    }
    return memory_format;
}

torch::jit::IValue intToIValue(const int64_t& value) { return torch::jit::IValue(value); }

torch::jit::IValue doubleToIValue(const double& value) { return torch::jit::IValue(value); }

torch::jit::IValue boolToIValue(const bool& value) { return torch::jit::IValue(value); }

torch::jit::IValue listToIValue(const c10::List<at::IValue>& value) { return torch::jit::IValue(value); }

bool isScalarType(DataType dtype)
{
    return dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
           dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::INT64 ||
           dtype == DataType::FLOAT16 || dtype == DataType::FLOAT32 || dtype == DataType::FLOAT64 ||
           dtype == DataType::BOOL;
}

std::vector<int64_t> getInBlobIds(const nncir::Node& node)
{
    std::vector<int64_t> ret;
    for (int i = 0; i < node.getInEdgeIds().size(); i++) {
        auto& data_edge = cast<nncir::DataEdge>(node.getInEdge(i));
        ret.push_back(data_edge.getBlobId());
    }
    return ret;
}

/**
 * @brief Get the Out Blob Ids, the result <ids> may contain duplicates elements, for example {1, 1, 2, 3, 3}
 * 
 * @param node 
 * @return std::vector<int64_t> 
 */
std::vector<int64_t> getOutBlobIds(const nncir::Node& node)
{
    std::vector<int64_t> ret;
    for (int i = 0; i < node.getOutEdgeIds().size(); i++) {
        auto& data_edge = cast<nncir::DataEdge>(node.getOutEdge(i));
        ret.push_back(data_edge.getBlobId());
    }
    return ret;
}

std::vector<int64_t> getUniqueOutBlobIds(const nncir::Node& node)
{
    std::vector<int64_t> ret;
    for (int i = 0; i < node.getOutEdgeIds().size(); i++) {
        auto& data_edge = cast<nncir::DataEdge>(node.getOutEdge(i));
        int blob_id = data_edge.getBlobId();
        if(std::find(ret.begin(), ret.end(), blob_id) == ret.end()){
            // not exist, insert
            ret.push_back(data_edge.getBlobId());
        }
    }
    return ret;
}


static std::unordered_map<DataType, std::string> dtype_to_str_map = {
    {DataType::UNDEFINED, "UNDEFINED"}, {DataType::INT8, "INT8"},
    {DataType::UINT8, "UINT8"},         {DataType::INT16, "INT16"},
    {DataType::UINT16, "UINT16"},       {DataType::INT32, "INT32"},
    {DataType::INT64, "INT64"},         {DataType::FLOAT16, "FLOAT16"},
    {DataType::FLOAT32, "FLOAT32"},     {DataType::FLOAT64, "FLOAT64"},
    {DataType::BOOL, "BOOL"},           {DataType::STRING, "STRING"},
    {DataType::DEVICE, "DEVICE"},       {DataType::TENSOR, "TENSOR"},
    {DataType::NONE, "NONE"},           {DataType::LIST, "LIST"}};

std::string getDataTypeStr(DataType dtype) { return dtype_to_str_map[dtype]; }

template <typename T>
std::ostream& operator<<(std::ostream& o, std::vector<T>& x)
{
    std::string ch = "";
    if (typeid(T) == typeid(std::string)) ch = '\"';
    if (typeid(T) == typeid(char)) ch = '\'';
    o << "[";
    for (int i = 0; i < x.size(); ++i) {
        o << ((i == 0) ? "" : ", ") << ch << x[i] << ch;
    }
    o << "]";
    return o;
}

std::string showOpNodeInfo(const nncir::Node& node, bool show_in_blobs, bool show_out_blobs)
{
    auto node_info = node.getNodeInfo();
    std::stringstream ss;

    ss << std::endl
       << "node_id: " << node_info.id << std::endl
       << "node_name:" << node_info.name << std::endl
       << "in_blob_ids: " << getInBlobIds(node) << std::endl
       << "out_blob_ids: " << getOutBlobIds(node) << std::endl;

    // remark: node_info.in_edge_ids & node_info.in_edge_ids maybe not correct
    //    <<"in_edge_ids: "<<node_info.in_edge_ids <<std::endl
    //    <<"out_edge_ids: "<<node_info.in_edge_ids <<std::endl
    return ss.str();
}

}  // namespace nnrt
