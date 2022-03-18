#pragma once

#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace ir
{
bool isSingleValueType(DataType data_type)
{
    return (data_type == DataType::INT8 || data_type == DataType::UINT8 || data_type == DataType::INT16 ||
            data_type == DataType::UINT16 || data_type == DataType::INT32 || data_type == DataType::INT64 ||
            data_type == DataType::FLOAT16 || data_type == DataType::FLOAT32 || data_type == DataType::FLOAT64 ||
            data_type == DataType::BOOL);
}

int32_t inferBitwidth(DataType type)
{
    int32_t bitwidth = 8;
    switch (type) {
        case DataType::INT8:
        case DataType::UINT8:
        case DataType ::LIST:
            bitwidth = 8;
            break;
        case DataType::INT16:
        case DataType::UINT16:
            bitwidth = 16;
            break;
        case DataType::INT64:
            bitwidth = 64;
            break;
        case DataType::INT32:
        case DataType::FLOAT32:
            bitwidth = 32;
            break;
        case DataType::FLOAT16:
            bitwidth = 16;
            break;
        default:
            DLOG(FATAL) << "Does not support data type";
            break;
    }
    return bitwidth;
}

std::string ConvertDataType(const DataType previous_type)
{
    switch (previous_type) {
        case INT8:
            return "int8";
        case UINT8:
            return "uint8";
        case INT16:
            return "int16";
        case UINT16:
            return "uint16";
        case INT32:
            return "int32";
        case INT64:
            return "int64";
        case FLOAT16:
            return "float16";
        case FLOAT32:
            return "float32";
        case FLOAT64:
            return "float64";
        case BOOL:
            return "bool";
        case STRING:
            return "string";
        case DEVICE:
            return "device";
        case TENSOR:
            return "Tensor";
        case NONE:
            return "None";
        case LIST:
            return "List";
        default:
            return "undefined";
    }
}

DataType inferDataType(int32_t bitwidth, std::string data_type)
{
    if (bitwidth == 8 && data_type == "SIGNED") {
        return DataType::INT8;
    }
    if (bitwidth == 16 && data_type == "SIGNED") {
        return DataType::INT16;
    }
    if (bitwidth == 8 && data_type == "UNSIGNED") {
        return DataType::UINT8;
    }
    if (bitwidth == 9 && data_type == "SIGNED") {
        return DataType::UINT8;
    }
    if (bitwidth == 21 && data_type == "SIGNED") {
        return DataType::INT32;
    }
    if (bitwidth == 32 && data_type == "SIGNED") {
        return DataType::INT32;
    }
    if (bitwidth == 48 && data_type == "SIGNED") {
        return DataType::INT64;
    }
    return DataType ::UNDEFINED;
}

}  // namespace ir
}  // namespace nn_compiler
