/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file    blob.cpp
 * @brief   This is Blob class
 * @details This source defines Blob class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */

#include <climits>

#include "common/include/arithmetics.hpp"
#include "common/include/wrapper_types.hpp"
#include "ir/include/blob.hpp"

namespace nn_compiler {
namespace nn_ir {

Blob::Blob(const Blob& other)
    : id_(other.graph_.getNextBlobId()), name_(other.name_), graph_(other.graph_), blob_type_(other.blob_type_),
      data_type_(other.data_type_), quant_type_(other.quant_type_), shape_type_(other.shape_type_), dim_(other.dim_),
      size_alignment_(other.size_alignment_), pos_alignment_(other.pos_alignment_), bit_width_(other.bit_width_),
      liveness_(other.liveness_), zero_point_(other.zero_point_), compress_(other.compress_),
      flc_fragments_(other.flc_fragments_), frac_len_ptr_(other.frac_len_ptr_) {}

Blob::~Blob() = default;

void Blob::setMemoryInfo(int32_t id, MEMORY_OFFSET_T addr, const nn_ir::DataLayout& layout, MEMORY_SIZE_T msb_stride) {
    MEMORY_SIZE_T size = layout.calcSizeInBytes(getUnitSize());

    for (auto& mem_info : memory_infos_[id]) {
        mem_info.addr       = addr;
        mem_info.size       = size;
        mem_info.msb_stride = msb_stride;
        mem_info.layout     = layout;
    }
}

} // namespace nn_ir
} // namespace nn_compiler
