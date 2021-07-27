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
    switch (dtype) {
        case nnrt::DataType::UINT8:
            return at::ScalarType::Byte;
        case nnrt::DataType::INT8:
            return at::ScalarType::Char;
        case nnrt::DataType::INT16:
            return at::ScalarType::Short;
        case nnrt::DataType::INT32:
            return at::ScalarType::Int;
        case nnrt::DataType::INT64:
            return at::ScalarType::Long;
        case nnrt::DataType::FLOAT16:
            return at::ScalarType::Half;
        case nnrt::DataType::FLOAT32:
            return at::ScalarType::Float;
        case nnrt::DataType::FLOAT64:
            return at::ScalarType::Double;
        case nnrt::DataType::BOOL:
            return at::ScalarType::Bool;
        default:
            DLOG(ERROR) << "Complex type has not been supported.";
    }
}

nnrt::DataType convertATScalarTypeToDType(at::ScalarType dtype)
{
    // according to: pytorch/c10/core/ScalarType.h
    switch (dtype) {
        case at::ScalarType::Byte:
            return nnrt::DataType::UINT8;
        case at::ScalarType::Char:
            return nnrt::DataType::INT8;
        case at::ScalarType::Short:
            return nnrt::DataType::INT16;
        case at::ScalarType::Int:
            return nnrt::DataType::INT32;
        case at::ScalarType::Long:
            return nnrt::DataType::INT64;
        case at::ScalarType::Half:
            return nnrt::DataType::FLOAT16;
        case at::ScalarType::Float:
            return nnrt::DataType::FLOAT32;
        case at::ScalarType::Double:
            return nnrt::DataType::FLOAT64;
        case at::ScalarType::Bool:
            return nnrt::DataType::BOOL;
        default:
            DLOG(ERROR) << "Complex type has not been supported.";
    }
}

at::Device convertIntToATDevice(const int& value) {
    // according to: pytorch/c10/core/DeviceType.h
    switch (value) {
        case 0:
            return at::Device(at::DeviceType::CPU);
        case 1:
            return at::Device(at::DeviceType::CUDA);
        case 2:
            return at::Device(at::DeviceType::MKLDNN);
        case 3:
            return at::Device(at::DeviceType::OPENGL);
        case 4:
            return at::Device(at::DeviceType::OPENCL);
        case 5:
            return at::Device(at::DeviceType::IDEEP);
        case 6:
            return at::Device(at::DeviceType::HIP);
        case 7:
            return at::Device(at::DeviceType::FPGA);
        case 8:
            return at::Device(at::DeviceType::MSNPU);
        case 9:
            return at::Device(at::DeviceType::XLA);
        case 10:
            return at::Device(at::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
        default:
            DLOG(ERROR) << "Unsupported device type.";
    }
}

at::MemoryFormat getMemoryFormat(int optional_memory_format)
{
    // according to: pytorch/c10/core/MemoryFormat.h
    switch (optional_memory_format) {
        case 0:
            return at::MemoryFormat::Contiguous;
        case 1:
            return at::MemoryFormat::Preserve;
        case 2:
            return at::MemoryFormat::ChannelsLast;
        case 3:
            return at::MemoryFormat::ChannelsLast3d;
        default:
            DLOG(ERROR) << "Wrong reference to aten MemoryFormat.";
    }
}

torch::jit::IValue intToIValue(const int64_t& value) { return torch::jit::IValue(value); }

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
