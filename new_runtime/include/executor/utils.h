#pragma once

#include <torch/script.h>

#include "common/log.hpp"
#include "new_ir/include/layers/nn_layer.h"
#include "new_runtime/include/common/log.hpp"

namespace nn_compiler
{
namespace runtime
{

torch::jit::IValue boolToIValue(const bool& value);
template <typename T>
torch::jit::IValue scalarToIValue(const T& scalar)
{
    return torch::jit::IValue(scalar);
}

torch::jit::IValue deviceToIValue(const c10::Device& device);

torch::jit::IValue tensorToIValue(const torch::Tensor& tensor);

torch::jit::IValue tensorListToIValue(const torch::TensorList& tensor_list);

torch::jit::IValue strToIValue(std::string str);

DataType convertATScalarTypeToDType(at::ScalarType dtype);

at::Device convertIntToATDevice(const int& value);

at::MemoryFormat getMemoryFormat(int optional_memory_format);

torch::jit::IValue intToIValue(const int64_t& value);

torch::jit::IValue doubleToIValue(const double& value);

torch::jit::IValue listToIValue(const c10::List<at::IValue>& value);

template <typename T>
torch::jit::IValue vectorToIValue(const std::vector<T>& value) {
    return torch::jit::IValue(value);
}

template <typename T>
at::ArrayRef<T> parseIValueArrayRef(const at::ArrayRef<at::IValue>& ivalue_array)
{
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
        Log::RT::E() << "Unsupported data type occurs in parseIValueInArrayRef().";
    }

    at::ArrayRef<T> array_ref(vec);
    return array_ref;
}

template <typename T>
std::vector<T> parseIValueVector(const at::ArrayRef<at::IValue>& ivalue_array)
{
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
        Log::RT::E() << "Unsupported data type occurs in parseIValueVector().";
    }
    return vec;
}

template <typename T>
void parseIValueList(at::IValue& list_iv, std::vector<T>& value_vec, std::vector<int64_t>& dim, int dim_idx)
{
    if (list_iv.isList()) {
        c10::ArrayRef<at::IValue> ivs = list_iv.toListRef();
        if (!ivs[0].isList()) {
            if (ivs[0].isInt()) {
                for (auto iv : ivs) {
                    value_vec.push_back(iv.toInt());
                    dim[dim_idx]++;
                }
            } else if (ivs[0].isDouble()) {
                for (auto iv : ivs) {
                    value_vec.push_back(iv.toDouble());
                    dim[dim_idx]++;
                }
            } else {
                Log::RT::E() << "Unsupported data type occurs in parseIValueList().";
            }
        } else {
            dim.push_back(1);
            dim_idx++;
            for (auto nested_list : ivs) {
                parseIValueList(nested_list, value_vec, dim, dim_idx);
            }
        }
    } else if (list_iv.isInt()) {
        value_vec.push_back(list_iv.toInt());
    } else if (list_iv.isDouble()) {
        value_vec.push_back(list_iv.toDouble());
    } else {
        Log::RT::E() << "Unsupported data type occurs in parseIValueList().";
    }
}

template <typename... T>
torch::jit::IValue tupleToIValue(std::tuple<T...> tuple)
{
    return torch::jit::IValue(tuple);
}

bool isScalarType(DataType dtype);

std::string getDataTypeStr(DataType dtype);

DataType inferDataType(torch::jit::IValue ival);

at::ListTypePtr inferTypeFromDataType(DataType type);

std::vector<int64_t> getUniqueOutStensorIds(std::shared_ptr<nn_compiler::ir::NNLayer>& layer);
}  // namespace runtime
}  // namespace nn_compiler
