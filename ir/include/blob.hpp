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
 * @file.    blob.hpp
 * @brief.   This is Blob class
 * @details. This header defines Blob class.
 * @version. 0.1.
 */

#pragma once

#include "common/arithmetics.hpp"
#include "common/common.hpp"
#include "common/cow_ptr.hpp"
#include "common/types.hpp"
#include "common/wrapper_types.hpp"
#include "ir/common/log.hpp"
#include "ir/nn_ir.hpp"

#include "ir/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class Blob {
 public:
    virtual ~Blob() = 0;

    Blob& operator=(const Blob&) = delete;
    Blob& operator=(Blob&&) = delete;

    std::unique_ptr<Blob> clone() const& { return std::unique_ptr<Blob>(this->cloneImpl()); }
    std::unique_ptr<Blob> clone() && { return std::unique_ptr<Blob>(std::move(*this).cloneImpl()); }

    nn_type::BlobUnitSize getUnitSize() const {
        auto bw = getBitWidthByType(data_type_);
        return nn_type::BlobUnitSize(bw);
    }
    uint32_t getSize() const { return dim_.getNumberOfElements(); }
    uint32_t getAlignedSize() const { return getAlignedShape().getNumberOfElements(); }
    uint64_t getSizeInBytes() const { return ::getSizeInBytes(getSize(), getUnitSize()); }

    uint64_t getAlignedSizeInBytes() const { return ::getSizeInBytes(getAlignedSize(), getUnitSize()); }
    uint32_t getUnitSizeInBytes() const { return ::getSizeInBytes(getUnitSize()); }

    BLOB_ID_T                         getId() const { return id_; }
    std::string                       getName() const { return name_; }
    BlobType                          getBlobType() const { return blob_type_; }
    DataType                          getDataType() const { return data_type_; }
    QuantType                         getQuantType() const { return quant_type_; }
    ShapeType                         getShapeType() const { return shape_type_; }
    Shape4D                           getShape() const { return dim_; }
    Shape4D                           getSizeAlignment() const { return size_alignment_; }
    Shape4D                           getPositionAlignment() const { return pos_alignment_; }
    Shape4D                           getAlignedShape() const { return dim_.getAlignedUpBy(size_alignment_); }
    BIT_WIDTH_T                       getBitWidth() const { return bit_width_; }
    std::pair<NODE_ID_T, NODE_ID_T>   getLiveness() const { return liveness_; }
    int32_t                           getZeroPoint() const { return zero_point_; }
    const std::vector<FRAC_LENGTH_T>& getFracLen() const { return *frac_len_ptr_; }
    bool                              getCompress() const { return compress_; }
    uint32_t                          getFLCFragments() const { return flc_fragments_; }
    const NNIR&                       getGraph() const { return graph_; }

    nn_ir::MemoryInfo getFirstMemoryAllocation(int32_t id) const {
        Log::IR::E_IF(id == INVALID_ID) << __FUNCTION__ << " is called with invalid id";
        auto mem_alloc = getMemoryAllocation(id);
        Log::IR::E_IF(mem_alloc.empty()) << *this << " has no MemoryInfo for id " << id;
        return mem_alloc.front();
    }

    FRAC_LENGTH_T getFracLen(uint32_t ch) const {
        if (frac_len_ptr_->size() == 1) {
            // Fraction length can be specified by a single scalar, in this case it
            // applies to all channels
            return (*frac_len_ptr_)[0];
        } else if (ch < frac_len_ptr_->size()) {
            return (*frac_len_ptr_)[ch];
        } else {
            // Default to 0 if missing
            return 0;
        }
    }

    const std::map<int32_t, std::vector<MemoryInfo>>& getAllMemoryAllocations() const { return memory_infos_; }

    // In following functions "int32_t id" can be either EDGE_ID_T or NODE_ID_T,
    // depending on the usage of the blob (featuremap or kernel etc)
    const std::vector<MemoryInfo>& getMemoryAllocation(int32_t id) const {
        auto it = memory_infos_.find(id);
        return it == memory_infos_.end() ? empty_memory_info_ : it->second;
    }

    void setMemoryAllocation(int32_t id, const std::vector<MemoryInfo>& memory_info) {
        memory_infos_[id] = memory_info;
    }

    void setAddrForAllAllocations(MEMORY_OFFSET_T addr, MEMORY_SIZE_T msb_stride = 0) {
        for (auto& mem_info_vector : memory_infos_) {
            for (nn_ir::MemoryInfo& mem_info : mem_info_vector.second) {
                mem_info.addr       = addr;
                mem_info.msb_stride = msb_stride;
            }
        }
    }

    void setMemoryAddr(int32_t id, MEMORY_OFFSET_T addr, MEMORY_SIZE_T msb_stride) {
        auto& memory_infos = memory_infos_[id];
        for (auto& mem_info : memory_infos) {
            mem_info.addr       = addr;
            mem_info.msb_stride = msb_stride;
        }
    }

    void setMemoryInfo(int32_t id, MEMORY_OFFSET_T addr, const nn_ir::DataLayout& layout, MEMORY_SIZE_T msb_stride = 0);

    void setMemoryMsbStride(int32_t id, MEMORY_SIZE_T msb_stride) {
        auto& memory_infos = memory_infos_[id];
        for (auto& mem_info : memory_infos) {
            mem_info.msb_stride = msb_stride;
        }
    }

    void setDim(Shape4D dim) { dim_ = dim; }
    void setAlignment(const Shape4D& alignment_unit) {
        // By default both alignments are the same
        size_alignment_ = alignment_unit;
        pos_alignment_  = alignment_unit;
    }
    void setAlignment(const Shape4D& size_align, const Shape4D& position_align) {
        size_alignment_ = size_align;
        pos_alignment_  = position_align;
    }
    void setCompress(bool compress) { compress_ = compress; }
    void setFLCFragments(uint32_t n) { flc_fragments_ = n; }

    void setName(std::string name) { name_ = name; }
    void setLiveness(std::pair<NODE_ID_T, NODE_ID_T> liveness) { liveness_ = liveness; }
    void setLivenessEnd(NODE_ID_T end) {
        if (liveness_.first == INVALID_ID) {
            liveness_.first = end;
        }
        liveness_.second = end;
    }
    void setLivenessStart(NODE_ID_T start) {
        if (liveness_.second == INVALID_ID || liveness_.second < start) {
            liveness_.second = start;
        }
        liveness_.first = start;
    }
    void expandLiveness(std::pair<NODE_ID_T, NODE_ID_T> liveness) {
        liveness_.first  = std::min(liveness_.first, liveness.first);
        liveness_.second = std::max(liveness_.second, liveness.second);
    }

 protected:
    explicit Blob(const BlobInfo& blob_info)
        : id_(blob_info.id), name_(blob_info.name), graph_(blob_info.graph), blob_type_(blob_info.blob_type),
          data_type_(blob_info.data_type), quant_type_(blob_info.quant_type), shape_type_(blob_info.shape_type),
          dim_(blob_info.dim), size_alignment_(blob_info.alignment_unit), pos_alignment_(blob_info.alignment_unit),
          bit_width_(blob_info.bit_width), liveness_(blob_info.liveness), zero_point_(blob_info.zero_point),
          compress_(blob_info.compress), flc_fragments_(1),
          frac_len_ptr_(estd::make_cow<std::vector<FRAC_LENGTH_T>>(std::move(blob_info.frac_len))) {}
    // TODO(s-steve-jang): ADDITIONAL ATTRIBUTES SHOULD BE INCLUDED
    //  quant_level_(blob_info.quant_level),

    Blob(const Blob&);
    Blob(Blob&&) = default;

    std::map<int32_t, std::vector<MemoryInfo>> memory_infos_;
    std::vector<MemoryInfo>                    empty_memory_info_;

 private:
    virtual Blob* cloneImpl() const& = 0;
    virtual Blob* cloneImpl() &&     = 0;

    BLOB_ID_T                       id_;
    std::string                     name_;
    const nn_ir::NNIR&              graph_;
    BlobType                        blob_type_;
    DataType                        data_type_;
    QuantType                       quant_type_;
    ShapeType                       shape_type_;
    Shape4D                         dim_;
    Shape4D                         size_alignment_; // Allocation alignment for blob shape
    Shape4D                         pos_alignment_;  // Coordinate alignment for split / concat
    BIT_WIDTH_T                     bit_width_;
    std::pair<NODE_ID_T, NODE_ID_T> liveness_ = {INVALID_ID, INVALID_ID};
    int32_t                         zero_point_;
    bool                            compress_;
    uint32_t                        flc_fragments_;
    //  QuantLevelType                  quant_level_; // TODO(s-steve-jang): new attribute
    estd::cow_ptr<std::vector<FRAC_LENGTH_T>> frac_len_ptr_;
}; // class Blob

inline std::ostream& operator<<(std::ostream& s, const Blob& blob) {
    s << "Blob #" << blob.getId() << " \"" << blob.getName() << "\" " << blob.getBlobType() << ' ' << blob.getDataType()
      << ' ' << blob.getShape();
    return s;
}

} // namespace nn_ir
} // namespace nn_compiler
