#include "executor/prim_utils.h"
#include <torch/script.h>
#include <stdexcept>
#include <vector>
#include "glog/logging.h"
#include "nnrt_types.h"
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;
namespace nnrt
{
torch::jit::IValue pop(std::vector<torch::jit::IValue>& stack)
{
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
torch::Tensor createPtTensor(void* data_ptr, const std::vector<int64_t>& shape, DataType dtype,
                             const std::vector<int64_t>& stride)
{
    c10::ScalarType scalar_type;

    if (dtype == DataType::FLOAT32) {
        scalar_type = c10::ScalarType::Float;
    } else if (dtype == DataType::FLOAT16) {
        scalar_type = c10::ScalarType::Half;
    } else if (dtype == DataType::INT32) {
        scalar_type = c10::ScalarType::Int;
    } else if (dtype == DataType::INT64) {
        scalar_type = c10::ScalarType::Long;
    } else if (dtype == DataType::BOOL) {
        scalar_type = c10::ScalarType::Bool;
    } else {
        DLOG(ERROR) << "Unsupport dtype when create Tensor";
    }
    auto sizes = c10::IntArrayRef(shape);
    if (stride.size() == 0) {
        return torch::from_blob(data_ptr, sizes, c10::TensorOptions().dtype(scalar_type));
    } else {
        return torch::from_blob(data_ptr, sizes, stride, c10::TensorOptions().dtype(scalar_type));
    }
}

std::vector<int64_t> getDataShapeFromShape4D(nn_compiler::nn_ir::Shape4D shape)
{
    std::vector<int64_t> temp_shape;
    if (shape.n != 0) {
        temp_shape.push_back(shape.n);
    }
    if (shape.c != 0) {
        temp_shape.push_back(shape.c);
    }
    if (shape.h != 0) {
        temp_shape.push_back(shape.h);
    }
    if (shape.w != 0) {
        temp_shape.push_back(shape.w);
    }
    return temp_shape;
}

torch::Tensor loadTensor(const std::string& bin_file, const std::vector<int64_t>& shape, DataType dtype)
{
    if(!fs::is_regular_file(fs::path(bin_file))) {
        std::runtime_error("Input file not exist!");
    }
    int num_elements = 1;
    for (auto item : shape) {
        num_elements *= item;
    }
    int total_size = 0;
    if (dtype == DataType::INT64) {
        total_size = num_elements * sizeof(int64_t);
    } else if (dtype == DataType::FLOAT32) {
        total_size = num_elements * sizeof(float);
    } else if (dtype == DataType::FLOAT16) {
        total_size = num_elements * sizeof(float) / 2;
    } else if (dtype == DataType::BOOL) {
        total_size = num_elements * sizeof(bool);
    } else {
        DLOG(ERROR) << "unsupported data type!";
    }
    char* buffer = new char[total_size];
    // read binary file
    std::ifstream ifs(bin_file, std::ios::binary | std::ios::in);
    ifs.read(buffer, total_size);
    ifs.close();

    auto tensor = createPtTensor((void*)buffer, shape, dtype).clone();
    delete[] buffer;
    return tensor;
}

std::pair<torch::jit::IValue, std::pair<int, DataType>> getVariableInfo(uint8_t* ptr,
                                                                        const std::string tensor_data_type,
                                                                        int total_size)
{
    DataType scalar_type = DataType::UNDEFINED;
    torch::jit::IValue iv;
    int bytenum = 0;
    if (tensor_data_type == "int64") {
        scalar_type = DataType::INT64;
        bytenum = sizeof(int64_t);
        iv = scalarToIValue<int64_t>(*(int64_t*)(ptr + total_size * bytenum));
    } else if (tensor_data_type == "int32") {
        scalar_type = DataType::INT32;
        bytenum = sizeof(int32_t);
        iv = scalarToIValue<int32_t>(*(int32_t*)(ptr + total_size * bytenum));
    } else if (tensor_data_type == "int16") {
        scalar_type = DataType::INT16;
        bytenum = sizeof(int16_t);
        iv = scalarToIValue<int16_t>(*(int16_t*)(ptr + total_size * bytenum));
    } else if (tensor_data_type == "uint16") {
        scalar_type = DataType::UINT16;
        bytenum = sizeof(uint16_t);
        iv = scalarToIValue<uint16_t>(*(uint16_t*)(ptr + total_size * bytenum));
    } else if (tensor_data_type == "int8") {
        scalar_type = DataType::INT8;
        bytenum = sizeof(int8_t);
        iv = scalarToIValue<int8_t>(*(int8_t*)(ptr + total_size * bytenum));
    } else if (tensor_data_type == "uint8") {
        scalar_type = DataType::UINT8;
        bytenum = sizeof(uint8_t);
        iv = scalarToIValue<uint8_t>(*(uint8_t*)(ptr + total_size * bytenum));
    } else if (tensor_data_type == "float32") {
        scalar_type = DataType::FLOAT32;
        bytenum = sizeof(float);
        iv = scalarToIValue<float>(*(float*)(ptr + total_size * bytenum));
    } else if (tensor_data_type == "float64") {
        scalar_type = DataType::FLOAT64;
        bytenum = sizeof(float) * 2;
        iv = scalarToIValue<double>(*(double*)(ptr + total_size * bytenum));
    } else if (tensor_data_type == "bool") {
        scalar_type = DataType::BOOL;
        bytenum = sizeof(int64_t);
        iv = scalarToIValue<int64_t>(*(int64_t*)(ptr + total_size * bytenum));
    } else {
        DLOG(ERROR) << "Element type do not support! ";
    }
    return {iv, {bytenum, scalar_type}};
}

}  // namespace nnrt
