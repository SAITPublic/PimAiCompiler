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

#include <atomic>

#include "new_ir/include/types.h"

namespace nn_compiler {
namespace ir {

class TSSTensor {
 public:
    explicit TSSTensor(std::vector<int32_t> dims) {
        dim_size_ = dims.size();
        for (int i = 0; i < dim_size_; i++) {
            dims_.emplace_back(dims[i]);
        }
        setID(getNextId());
    }

    TSSTensor() {
        setID(getNextId());
    }

    uint32_t getID() {
        return id_;
    }

    void setParentLayer(uint32_t layer_id) {
        parent_layer_id_ = layer_id;
    }
    uint32_t getParentLayer() {
        return parent_layer_id_;
    }

    void setDim(std::vector<int32_t> dims) {
        dims_.clear();
        dim_size_ = dims.size();
        for (int i = 0; i < dim_size_; i++) {
            dims_.emplace_back(dims[i]);
        }
    }
    std::vector<int32_t> getDim() {
        return dims_;
    }

    void setDimN(int32_t num_dim, int32_t dim_val) {
        if (num_dim > dim_size_) {
          Log::IR::I() << "Max dimension is " << dim_size_ << ", but " << num_dim
                       << " is given.";
          assert(false);
        }
        dims_[num_dim - 1] = dim_val;
    }
    int32_t getDimN(int32_t num_dim) {
        if (num_dim > dim_size_) {
          Log::IR::I() << "Max dimension is " << dim_size_ << ", but " << num_dim
                       << " is given.";
          assert(false);
        }
        return dims_[num_dim - 1];
    }

    int32_t getDimSize() {
        return dim_size_;
    }

 private:
    static uint32_t getNextId() {
        static std::atomic<uint32_t> id{0};
        return id++;
    }

    uint32_t id_;
    void setID(uint32_t id) {
        id_ = id;
    }
    std::vector<int32_t> dims_;
    int32_t dim_size_ = 0;
    uint32_t parent_layer_id_ = 0;
};

} // namespace ir
} // namespace nn_compiler