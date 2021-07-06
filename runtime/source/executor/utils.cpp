#include <torch/script.h>
#include <string>
#include <string>
#include <vector>
#include <torch/script.h>
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/nn_ir.hpp"
#include "nnrt_types.h"

#include "executor/utils.h"

namespace nnrt
{
  
namespace nncir = nn_compiler::nn_ir;

torch::jit::IValue tensorToIValue(const torch::Tensor& tensor) { return torch::jit::IValue(tensor); }

torch::jit::IValue tensorListToIValue(const torch::TensorList& tensor_list) { return torch::jit::IValue(tensor_list); }

torch::jit::IValue strToIValue(std::string str) { return torch::jit::IValue(str); }

at::ScalarType convertDTypeToATScalarType(nncir::DataType dtype)
{
    // according to: pytorch/c10/core/ScalarType.h
    switch (dtype) {
        case nncir::DataType::UINT8:
            return at::ScalarType::Byte;
        case nncir::DataType::INT8:
            return at::ScalarType::Char;
        case nncir::DataType::INT16:
            return at::ScalarType::Short;
        case nncir::DataType::INT32:
            return at::ScalarType::Int;
        case nncir::DataType::INT64:
            return at::ScalarType::Long;
        case nncir::DataType::FLOAT16:
            return at::ScalarType::Half;
        case nncir::DataType::FLOAT32:
            return at::ScalarType::Float;
        case nncir::DataType::FLOAT64:
            return at::ScalarType::Double;
        case nncir::DataType::BOOL:
            return at::ScalarType::Bool;
        default:
            DLOG(ERROR) << "Complex type has not been supported.";
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

bool isScalarType(DataType dtype) {
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

std::vector<int64_t> getOutBlobIds(const nncir::Node& node)
{
    std::vector<int64_t> ret;
    for (int i = 0; i < node.getOutEdgeIds().size(); i++) {
        auto& data_edge = cast<nncir::DataEdge>(node.getOutEdge(i));
        ret.push_back(data_edge.getBlobId());
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
}  // namespace nnrt
