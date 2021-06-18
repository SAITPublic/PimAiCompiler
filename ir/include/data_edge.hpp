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
 * @file.    data_edge.hpp
 * @brief.   This is DataEdge class
 * @details. This header defines DataEdge class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/blob.hpp"
#include "ir/edge.hpp"
#include "ir/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class DataEdge : public Edge {
 public:
    explicit DataEdge(const EdgeInfo& edge_info, BLOB_ID_T blob_id)
        : Edge(edge_info, EdgeType::DATA), blob_id_(blob_id) {}

    std::unique_ptr<DataEdge> clone() const& { return std::unique_ptr<DataEdge>(this->cloneImpl()); }
    std::unique_ptr<DataEdge> clone() && { return std::unique_ptr<DataEdge>(std::move(*this).cloneImpl()); }

    BLOB_ID_T getBlobId() const { return blob_id_; }
    Blob*     getBlob() const;
    void      setBlobId(BLOB_ID_T id) { blob_id_ = id; }

    nn_ir::MemoryInfo getFirstMemoryAllocation() const { return getBlob()->getFirstMemoryAllocation(getId()); }
    std::vector<nn_ir::MemoryInfo> getMemoryAllocation() const { return getBlob()->getMemoryAllocation(getId()); }

    template <typename T>
    static bool classof(const Edge* edge) {
        static_assert(std::is_same<T, DataEdge>::value, "incorrect type");
        return edge->getEdgeType() == EdgeType::DATA;
    }

 private:
    DataEdge(const DataEdge&) = default;
    DataEdge(DataEdge&&)      = default;

    DataEdge* cloneImpl() const& override { return new DataEdge(*this); }
    DataEdge* cloneImpl() && override { return new DataEdge(std::move(*this)); }

    BLOB_ID_T blob_id_;
}; // class DataEdge

} // namespace nn_ir
} // namespace nn_compiler
