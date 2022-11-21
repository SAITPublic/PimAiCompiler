/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#include <torch/script.h>
#include <stdexcept>

#include "executor/utils/utils.h"

namespace nn_compiler
{
namespace runtime
{
namespace utils
{
torch::jit::IValue deviceToIValue(const c10::Device& device) { return torch::jit::IValue(device); }

torch::jit::IValue tensorToIValue(const torch::Tensor& tensor) { return torch::jit::IValue(tensor); }

torch::jit::IValue tensorListToIValue(const torch::TensorList& tensor_list) { return torch::jit::IValue(tensor_list); }

torch::jit::IValue strToIValue(std::string str) { return torch::jit::IValue(str); }

DataType convertATScalarTypeToDType(at::ScalarType dtype)
{
    // according to: pytorch/c10/core/ScalarType.h
    DataType data_type = DataType::NONE;
    switch (dtype) {
        case at::ScalarType::Byte:
            data_type = DataType::UINT8;
            break;
        case at::ScalarType::Char:
            data_type = DataType::INT8;
            break;
        case at::ScalarType::Short:
            data_type = DataType::INT16;
            break;
        case at::ScalarType::Int:
            data_type = DataType::INT32;
            break;
        case at::ScalarType::Long:
            data_type = DataType::INT64;
            break;
        case at::ScalarType::Half:
            data_type = DataType::FLOAT16;
            break;
        case at::ScalarType::Float:
            data_type = DataType::FLOAT32;
            break;
        case at::ScalarType::Double:
            data_type = DataType::FLOAT64;
            break;
        case at::ScalarType::Bool:
            data_type = DataType::BOOL;
            break;
        default:
            DLOG(FATAL) << "Complex type has not been supported.";
    }
    return data_type;
}

at::Device convertIntToATDevice(const int& value)
{
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
            device = at::Device(at::DeviceType::ORT);
            break;
        case 9:
            device = at::Device(at::DeviceType::XLA);
            break;
        case 10:
            device = at::Device(at::DeviceType::Vulkan);
            break;
        case 11:
            device = at::Device(at::DeviceType::Metal);
            break;
        case 12:
            device = at::Device(at::DeviceType::XPU);
            break;
        case 13:
            device = at::Device(at::DeviceType::MLC);
            break;
        case 14:
            device = at::Device(at::DeviceType::Meta);
            break;
        case 15:
            device = at::Device(at::DeviceType::HPU);
            break;
        case 16:
            device = at::Device(at::DeviceType::VE);
            break;
        case 17:
            device = at::Device(at::DeviceType::Lazy);
            break;
        case 18:
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

DataType inferDataType(torch::jit::IValue ival)
{
    DataType type = DataType::UNDEFINED;
    if (ival.isList()) {
        type = DataType::LIST;
    } else if (ival.isBool()) {
        type = DataType::BOOL;
    } else if (ival.isInt()) {
        type = DataType::INT64;
    } else if (ival.isString()) {
        type = DataType::STRING;
    } else if (ival.isNone()) {
        type = DataType::NONE;
    } else if (ival.isDouble()) {
        type = DataType::FLOAT64;
    } else if (ival.isDevice()) {
        type = DataType::DEVICE;
    } else if (ival.isTensor()) {
        type = DataType::TENSOR;
    } else if (ival.isTuple()) {
        type = DataType::TUPLE;
    } else {
        DLOG(INFO) << ival.type()->repr_str() << " is not supported yet.";
    }
    return type;
}

at::ListTypePtr inferTypeFromDataType(DataType type)
{
    at::ListTypePtr list_type = at::ListType::ofTensors();
    switch (type) {
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::INT64:
            list_type = at::ListType::ofInts();
            break;
        case DataType::FLOAT16:
        case DataType::FLOAT32:
        case DataType::FLOAT64:
            list_type = at::ListType::ofFloats();
            break;
        case DataType::BOOL:
            list_type = at::ListType::ofBools();
            break;
        case DataType::STRING:
            list_type = at::ListType::ofStrings();
            break;
        case DataType::TENSOR:
            list_type = at::ListType::ofTensors();
            break;
        default:
            DLOG(INFO) << "DataType do not support! ";
            break;
    }
    return list_type;
}

std::vector<int64_t> getUniqueOutStensorIds(std::shared_ptr<nn_compiler::ir::NNLayer>& layer)
{
    std::vector<int64_t> ret;
    auto out_stensor_id = layer->getOutSTensorID();
    for (int i = 0; i < out_stensor_id.size(); i++) {
        if (std::find(ret.begin(), ret.end(), out_stensor_id[i]) == ret.end()) {
            // not exist, insert
            ret.push_back(out_stensor_id[i]);
        }
    }
    return ret;
}

std::vector<int64_t> getDataShapeFromVector(const std::vector<int64_t>& value)
{
    std::vector<int64_t> value_;
    for (auto item : value) {
        if (item != 0) {
            value_.push_back(item);
        }
    }
    return value_;
}

torch::jit::IValue pop(std::vector<torch::jit::IValue>& stack)
{
    auto r = std::move(stack.back());
    stack.pop_back();
    return r;
}

void drop(std::vector<torch::jit::IValue>& stack, size_t n) { stack.erase(stack.end() - n, stack.end()); }

void gemmBroadcast(std::vector<int64_t>& shape1, std::vector<int64_t>& shape2)
{
    int dim1 = shape1.size();
    int dim2 = shape2.size();
    std::vector<int64_t> t1;
    std::vector<int64_t> t2;
    if (dim1 == dim2) {
        for (int i = 0; i < 4; i++) {
            if (i < dim1) {
                t1.emplace_back(shape1[i]);
                t2.emplace_back(shape2[i]);
            } else {
                t1.insert(t1.begin(), 1);
                t2.insert(t2.begin(), 1);
            }
        }
        shape1 = t1;
        shape2 = t2;
    } else if (dim1 < dim2) {
        int num = dim2 - dim1;
        if (num == 1) {
            if (shape1[0] == shape2[0]) {
                shape1.insert(shape1.begin() + 1, 1);
            } else {
                shape1.insert(shape1.begin(), 1);
            }
            gemmBroadcast(shape1, shape2);
        } else if (num == 2) {
            if (shape1[0] == shape2[0]) {
                shape1.insert(shape1.begin() + 1, 1);
                shape1.insert(shape1.begin() + 1, 1);
            } else if (shape1[0] == shape2[1]) {
                shape1.insert(shape1.begin() + 1, 1);
                shape1.insert(shape1.begin(), 1);
            } else if (shape1[1] == shape2[2]) {
                shape1.insert(shape1.begin(), 1);
                shape1.insert(shape1.begin(), 1);
            }
        } else {
            DLOG(FATAL) << "The dims of input and weight must less than 4";
        }

    } else if (dim1 > dim2) {
        int num = dim1 - dim2;
        if (num == 1) {
            if (shape1[dim1 - 1] == shape2[dim2 - 2]) {
                shape2.insert(shape2.begin(), 1);
            } else {
                shape2.emplace_back(1);
            }
            gemmBroadcast(shape1, shape2);
        } else if (num == 2) {
            if (shape1[3] == shape2[1]) {
                shape2.emplace_back(1);
            } else {
                shape2.insert(shape2.begin(), 1);
            }

            if (shape1[0] == shape2[0]) {
                shape2.insert(shape2.begin() + 1, 1);
            } else {
                shape2.insert(shape2.begin(), 1);
            }
        } else if (num == 3) {
            auto tmp = shape2[0];
            if (tmp == shape1[3]) {
                shape2 = shape1;
                shape2[2] = tmp;
                shape2[3] = 1;
            } else {
                shape2 = shape1;
                shape2[2] = 1;
                shape2[3] = tmp;
            }
        } else {
            DLOG(FATAL) << "The dims of input and weight must less than 4";
        }
    }
}

GEMM_TYPE getGemmType(std::vector<int64_t> self_shape, std::vector<int64_t> other_shape)
{
    GEMM_TYPE type_ = GEMM_TYPE::UNKNOW;
    if (self_shape[0] == 1 && self_shape[1] == 1 && other_shape[0] == 1 && other_shape[1] == 1) {
        int self_is_vector = 0;
        int other_is_vector = 0;
        if (self_shape[2] != 1) self_is_vector++;
        if (self_shape[3] != 1) self_is_vector++;
        if (other_shape[2] != 1) other_is_vector++;
        if (other_shape[3] != 1) other_is_vector++;
        if (self_is_vector == 1 && other_is_vector == 1) {
            type_ = GEMM_TYPE::VV;
        } else if (self_is_vector == 1 && other_is_vector == 2) {
            type_ = GEMM_TYPE::VM;
        } else if (self_is_vector == 2 && other_is_vector == 1) {
            type_ = GEMM_TYPE::MV;
        } else {
            type_ = GEMM_TYPE::MM;
        }
    } else {
        type_ = GEMM_TYPE::MM;
    }
    return type_;
}

}  // namespace utils
}  // namespace runtime
}  // namespace nn_compiler
