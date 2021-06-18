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
 * @file.    featuremap_blob.hpp
 * @brief.   This is FeaturemapBlob class
 * @details. This header defines FeaturemapBlob class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/log.hpp"
#include "common/include/types.hpp"

#include "ir/include/blob.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class FeaturemapBlob : public Blob {
 public:
    explicit FeaturemapBlob(const BlobInfo& blob_info) : Blob(blob_info) {}

    std::unique_ptr<FeaturemapBlob> clone() const& { return std::unique_ptr<FeaturemapBlob>(this->cloneImpl()); }
    std::unique_ptr<FeaturemapBlob> clone() && { return std::unique_ptr<FeaturemapBlob>(std::move(*this).cloneImpl()); }

    template <typename T>
    static bool classof(const Blob* blob) {
        static_assert(std::is_same<T, FeaturemapBlob>::value, "incorrect type");
        return blob->getBlobType() == BlobType::FEATUREMAP;
    }

 private:
    FeaturemapBlob(const FeaturemapBlob&) = default;
    FeaturemapBlob(FeaturemapBlob&&)      = default;

    FeaturemapBlob* cloneImpl() const& override { return new FeaturemapBlob(*this); }
    FeaturemapBlob* cloneImpl() && override { return new FeaturemapBlob(std::move(*this)); }
}; // class FeaturemapBlob

} // namespace nn_ir
} // namespace nn_compiler
