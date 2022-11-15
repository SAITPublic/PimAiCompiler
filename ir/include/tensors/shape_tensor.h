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

#pragma once

#include <atomic>

#include "ir/include/types.h"

namespace nn_compiler
{
namespace ir
{
class STensor
{
   public:
    explicit STensor(std::vector<int32_t> dims)
    {
        dim_size_ = dims.size();
        for (int i = 0; i < dim_size_; i++) {
            dims_.emplace_back(dims[i]);
        }
        setID(getNextId());
    }

    STensor() { setID(getNextId()); }

    uint32_t getID() { return id_; }

    void setDims(std::vector<int32_t> dims)
    {
        dims_.clear();
        dim_size_ = dims.size();
        for (int i = 0; i < dim_size_; i++) {
            dims_.emplace_back(dims[i]);
        }
    }
    std::vector<int32_t> getDims() { return dims_; }

    int32_t getDimSize() { return dim_size_; }

    void setFeaturemapType(DataType data_type) { featuremap_type_ = data_type; }

    DataType getFeaturemapType() const { return featuremap_type_; }

    void setReprType(std::string type_str) { repr_type_ = type_str; }

    std::string getReprType() { return repr_type_; }

   private:
    void setID(uint32_t id) { id_ = id; }

    static uint32_t getNextId()
    {
        static std::atomic<uint32_t> id{0};
        return id++;
    }

    uint32_t id_ = 0;
    std::string repr_type_ = "";
    std::vector<int32_t> dims_;
    int32_t dim_size_ = 0;
    DataType featuremap_type_ = DataType::UNDEFINED;
};

}  // namespace ir
}  // namespace nn_compiler
