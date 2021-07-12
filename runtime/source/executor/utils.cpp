#include <string>
#include <torch/script.h>

#include "executor/utils.h"

namespace nnrt
{

torch::jit::IValue tensorToIValue(const torch::Tensor& tensor) { return torch::jit::IValue(tensor); }

torch::jit::IValue tensorListToIValue(const torch::TensorList& tensor_list) { return torch::jit::IValue(tensor_list); }

torch::jit::IValue strToIValue(std::string str) { return torch::jit::IValue(str); }

at::ScalarType convertDTypeToATScalarType(nncir::DataType dtype) {
    // according to: pytorch/c10/core/ScalarType.h
    switch (dtype) {
        case nncir::DataType::UINT8 :
            return at::ScalarType::Byte;
        case nncir::DataType::INT8 :
            return at::ScalarType::Char;
        case nncir::DataType::INT16 :
            return at::ScalarType::Short;
        case nncir::DataType::INT32 :
            return at::ScalarType::Int;
        case nncir::DataType::INT64 :
            return at::ScalarType::Long;
        case nncir::DataType::FLOAT16 :
            return at::ScalarType::Half;
        case nncir::DataType::FLOAT32 :
            return at::ScalarType::Float;
        case nncir::DataType::FLOAT64 :
            return at::ScalarType::Double;
        case nncir::DataType::BOOL :
            return at::ScalarType::Bool;
        default:
            DLOG(ERROR) << "Complex type has not been supported.";
    }
}

at::MemoryFormat getMemoryFormat(int optional_memory_format) {
    // according to: pytorch/c10/core/MemoryFormat.h
    switch (optional_memory_format) {
        case 0 :
            return at::MemoryFormat::Contiguous;
        case 1 :
            return at::MemoryFormat::Preserve;
        case 2 :
            return at::MemoryFormat::ChannelsLast;
        case 3 :
            return at::MemoryFormat::ChannelsLast3d;
        default :
            DLOG(ERROR) << "Wrong reference to aten MemoryFormat.";
    }
}

}  // namespace nnrt
