/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include <torch/script.h>
#include <experimental/filesystem>
#include <stdexcept>

#include "utils/utils.h"

namespace fs = std::experimental::filesystem;

namespace nn_compiler
{
namespace runtime
{
namespace utils
{
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
        DLOG(FATAL) << "Unsupport dtype when create Tensor";
    }
    auto sizes = c10::IntArrayRef(shape);
    if (stride.size() == 0) {
        return torch::from_blob(data_ptr, sizes, c10::TensorOptions().dtype(scalar_type));
    } else {
        return torch::from_blob(data_ptr, sizes, stride, c10::TensorOptions().dtype(scalar_type));
    }
}

std::vector<int64_t> getDataShapeFromSTensor(nn_compiler::ir::STensor& stensor)
{
    std::vector<int64_t> shape;
    for (auto item : stensor.getDims()) {
        if (item != 0) {
            shape.push_back(item);
        }
    }
    return shape;
}

torch::Tensor loadTensor(const std::string& bin_file, const std::vector<int64_t>& shape, DataType dtype)
{
    if (!fs::is_regular_file(fs::path(bin_file))) {
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
        DLOG(FATAL) << "unsupported data type!";
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

std::pair<int, DataType> parseNtype(std::string& ntype)
{
    // presupposition: same datatype in a single tuple

    auto ntypeParser = [](const std::string& ntype) -> std::pair<int, DataType> {
        auto element_type = DataType::UNDEFINED;
        int element_num = 0;
        if (ntype.find("int") != std::string::npos) {
            for (int i = 0; (i = ntype.find("int", i)) != std::string::npos; element_num++, i += 3)
                ;
            element_type = DataType::INT64;
        } else if (ntype.find("float") != std::string::npos) {
            for (int i = 0; (i = ntype.find("float", i)) != std::string::npos; element_num++, i += 5)
                ;
            element_type = DataType::FLOAT64;
        } else if (ntype.find("bool") != std::string::npos) {
            for (int i = 0; (i = ntype.find("bool", i)) != std::string::npos; element_num++, i += 4)
                ;
            element_type = DataType::INT64;
        } else {
            DLOG(FATAL) << "unspported datatype for tuple.";
        }
        return std::make_pair(element_num, element_type);
    };
    return ntypeParser(ntype);
}

torch::jit::IValue convertVaraibleData2IValve(uint8_t* ptr, DataType d_type)
{
    torch::jit::IValue iv;
    switch (d_type) {
        case DataType::INT64:
            iv = scalarToIValue<int64_t>(*(int64_t*)(ptr));
            break;
        case DataType::INT32:
            iv = scalarToIValue<int32_t>(*(int32_t*)(ptr));
            break;
        case DataType::INT16:
            iv = scalarToIValue<int16_t>(*(int16_t*)(ptr));
            break;
        case DataType::INT8:
            iv = scalarToIValue<int8_t>(*(int8_t*)(ptr));
            break;
        case DataType::UINT8:
            iv = scalarToIValue<uint8_t>(*(uint8_t*)(ptr));
            break;
        case DataType::FLOAT32:
            iv = scalarToIValue<float>(*(float*)(ptr));
            break;
        case DataType::FLOAT64:
            iv = scalarToIValue<double>(*(double*)(ptr));
            break;
        case DataType::BOOL:
            iv = scalarToIValue<int64_t>(*(int64_t*)(ptr));
            break;
        default:
            DLOG(FATAL) << "Element type do not support! ";
            break;
    }
    return iv;
}

}  // namespace utils
}  // namespace runtime
}  // namespace nn_compiler
