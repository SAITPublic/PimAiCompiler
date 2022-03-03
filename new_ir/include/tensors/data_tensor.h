/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include <stdint.h>
#include <string.h>
#include <vector>

#include "new_ir/include/tensors/shape_tensor.h"
#include "new_ir/include/types.h"

namespace nn_compiler
{
namespace ir
{
class DTensor
{
   public:
    DTensor() : DTensor(0, 0, DataType::UNDEFINED, 8) {}

    explicit DTensor(const void* data_ptr, uint32_t data_bytes, DataType data_type, int32_t bit_width)
    {
        static uint32_t id = 0;
        setID(id);
        id++;
        setData(data_ptr, data_bytes);
        setDataType(data_type);
        setBitWidth(bit_width);
    }

    explicit DTensor(const DTensor& rhs)
    {
        this->data_arr_ = rhs.data_arr_;
        this->data_type_ = rhs.data_type_;
        this->bit_width_ = rhs.bit_width_;
        this->shape_tensor_ = rhs.shape_tensor_;
        this->stride_ = rhs.stride_;

        this->id_ = rhs.id_;
    }

    std::shared_ptr<DTensor> clone() { return std::shared_ptr<DTensor>(new DTensor(*this)); }

    template <typename DType>
    const std::shared_ptr<std::vector<DType>> getData()
    {
        auto res = std::make_shared<std::vector<DType>>(data_arr_.size() / sizeof(DType));
        memcpy(res->data(), data_arr_.data(), data_arr_.size());
        return res;
    }

    uint32_t getID() { return id_; }

    void setData(const void* data_ptr, uint32_t data_bytes)
    {
        data_arr_.resize(data_bytes);
        memcpy(data_arr_.data(), data_ptr, data_bytes);
    }

    DataType getDataType() { return data_type_; }

    void setDataType(DataType data_type) { data_type_ = data_type; }

    void setStride(std::vector<int64_t> stride) { stride_ = stride; }

    std::vector<int64_t> getStride() { return stride_; }

    int32_t getBitWidth() const { return bit_width_; }

    void setBitWidth(int32_t bit_width) { bit_width_ = bit_width; }

    void setTensorShape(const STensor shape_tensor) { shape_tensor_ = shape_tensor; }

    STensor getTensorShape() const { return shape_tensor_; }

    ~DTensor() {}

   private:
    uint32_t id_;
    void setID(uint32_t id) { id_ = id; }

    DataType data_type_ = DataType::UNDEFINED;

    std::vector<uint8_t> data_arr_;

    int32_t bit_width_ = 0;

    std::vector<int64_t> stride_ = {0, 0, 0, 0};

    STensor shape_tensor_;
};

}  // namespace ir
}  // namespace nn_compiler
