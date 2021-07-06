#pragma once

#include <torch/script.h>
#include "ir/include/nn_ir.hpp"

#include "ir/include/nn_ir.hpp"
#include "nnrt_types.h"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
torch::jit::IValue boolToIValue(const bool& value);
template <typename T>
torch::jit::IValue scalarToIValue(const T& scalar)
{
    return torch::jit::IValue(scalar);
}

torch::jit::IValue tensorToIValue(const torch::Tensor& tensor);

torch::jit::IValue tensorListToIValue(const torch::TensorList& tensor_list);

torch::jit::IValue strToIValue(std::string str);

at::ScalarType convertDTypeToATScalarType(nncir::DataType dtype);

at::MemoryFormat getMemoryFormat(int optional_memory_format);

torch::jit::IValue intToIValue(const int64_t& value);

torch::jit::IValue listToIValue(const c10::List<at::IValue>& value);

template <typename T>
at::ArrayRef<T> parseIValueArrayRef(const at::ArrayRef<at::IValue>& ivalue_array) {
    assert(ivalue_array.size() > 0);
    std::vector<T> vec;

    if (ivalue_array[0].isInt()) {
        for (auto item : ivalue_array) {
            vec.push_back(item.toInt());
        }
    } else if (ivalue_array[0].isDouble()) {
        for (auto item : ivalue_array) {
            vec.push_back(item.toDouble());
        }
    } else {
        DLOG(ERROR) << "Unsupported data type occurs in parseIValueInArrayRef().";
    }

    at::ArrayRef<T> array_ref(vec);
    return array_ref;
}

template <typename ...T>
torch::jit::IValue tupleToIValue(std::tuple<T...> tuple) { return torch::jit::IValue(tuple); }

bool isScalarType(DataType dtype);

std::vector<int64_t> getOutBlobIds(const nn_compiler::nn_ir::Node& node);

std::vector<int64_t> getInBlobIds(const nn_compiler::nn_ir::Node& node);

std::string getDataTypeStr(DataType dtype);
}  // namespace nnrt
