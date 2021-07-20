#include "executor/prim_utils.h"
#include <torch/script.h>
#include <stdexcept>
#include <vector>
#include "glog/logging.h"
#include "nnrt_types.h"

namespace nnrt {
torch::jit::IValue pop(std::vector<torch::jit::IValue>& stack) {
    auto r = std::move(stack.back());
    stack.pop_back();
    return r;
}

void drop(std::vector<torch::jit::IValue>& stack, size_t n) { stack.erase(stack.end() - n, stack.end()); }

/**
 * @brief Create a PyTorch Tensor
 *
 * @param data_ptr
 * @param shape
 * @param dtype
 * @return torch::Tensor
 */
torch::Tensor createPtTensor(void* data_ptr, const std::vector<int64_t>& shape, DataType dtype) {
    c10::ScalarType scalar_type;

    if (dtype == DataType::FLOAT32) {
        scalar_type = c10::ScalarType::Float;
    } else if (dtype == DataType::FLOAT16) {
        scalar_type = c10::ScalarType::Half;
    } else if (dtype == DataType::INT32) {
        scalar_type = c10::ScalarType::Int;
    } else if(dtype == DataType::INT64) {
        scalar_type = c10::ScalarType::Long;
    }
    else {
        DLOG(ERROR) << "Unsupport dtype when create Tensor";
    }
    auto sizes = c10::IntArrayRef(shape);
    return torch::from_blob(data_ptr, sizes, c10::TensorOptions().dtype(scalar_type));
}

DataType inferDataType(torch::jit::IValue ival) {
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

at::ListTypePtr inferTypeFromDataType(DataType type) {
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
    default:
        DLOG(INFO) << "DataType do not support! ";
        break;
    }
    return list_type;
}

}  // namespace nnrt
