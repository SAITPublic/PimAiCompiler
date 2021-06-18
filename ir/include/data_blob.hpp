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
 * @file.    data_blob.hpp
 * @brief.   This is DataBlob class
 * @details. This header defines DataBlob class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"
#include "ir/common/log.hpp"

#include "ir/blob.hpp"
#include "ir/ir_types.hpp"
#include <variant>

#include "common/cow_ptr.hpp"

namespace nn_compiler {
namespace nn_ir {

class DataBlob : public Blob {
 public:
    // just short alias for vectors of specific data
    template <typename T>
    using DVec    = std::vector<T>;
    using MetaVec = DVec<std::uint8_t>;

    // this filed contains all kinds of supported types by data blob
    using VAR_TYPE = std::variant<DVec<std::int8_t>,
                                  DVec<std::uint8_t>,
                                  DVec<std::int16_t>,
                                  DVec<std::uint16_t>,
                                  DVec<std::int32_t>,
                                  DVec<std::uint32_t>,
                                  DVec<std::int64_t>,
                                  DVec<std::uint64_t>,
                                  DVec<float16>,
                                  DVec<float>,
                                  DVec<double>>;

    template <typename DType>
    explicit DataBlob(const BlobInfo& blob_info, std::vector<DType> data_arr)
        : Blob(blob_info), buf_ptr_(estd::make_cow<VAR_TYPE>(std::move(data_arr))) {}

    template <typename DType>
    explicit DataBlob(const BlobInfo& blob_info, std::vector<DType> data_arr, MetaVec meta_arr)
        : Blob(blob_info), buf_ptr_(estd::make_cow<VAR_TYPE>(std::move(data_arr))),
          meta_buf_ptr_(estd::make_cow<MetaVec>(std::move(meta_arr))) {}

    std::unique_ptr<DataBlob> clone() const& { return std::unique_ptr<DataBlob>(this->cloneImpl()); }
    std::unique_ptr<DataBlob> clone() && { return std::unique_ptr<DataBlob>(std::move(*this).cloneImpl()); }

    template <typename DType>
    const std::vector<DType>& getBuf() const {
        Log::IR::E_IF(!std::holds_alternative<DVec<DType>>(*buf_ptr_)) << "data blob doesn't contain this type!";
        return std::get<DVec<DType>>(*buf_ptr_);
    }

    const VAR_TYPE& getBuf() const { return *buf_ptr_; }

    template <typename DType>
    std::vector<DType>& getBufModifiable() {
        Log::IR::E_IF(!std::holds_alternative<DVec<DType>>(*buf_ptr_)) << "data blob doesn't contain this type!";
        return std::get<DVec<DType>>(*buf_ptr_);
    }

    template <typename DType>
    void setBuf(const std::vector<DType>& buf) {
        Log::IR::E_IF(!std::holds_alternative<DVec<DType>>(*buf_ptr_)) << "invalid data blob type assignment!";
        buf_ptr_ = estd::make_cow<VAR_TYPE>(buf);
    }

    const MetaVec& getMetaBuf() const { return *meta_buf_ptr_; }
    MetaVec&       getMetaBufModifiable() { return *meta_buf_ptr_; }

    void setMetaBuf(const MetaVec& meta_buf) { meta_buf_ptr_ = estd::make_cow<MetaVec>(meta_buf); }

    bool hasMetadata() const { return meta_buf_ptr_->size() != 0; }

    MEMORY_SIZE_T getBufSizeInByte() const {
        switch (getDataType()) {
            case nn_ir::DataType::UINT16:
            case nn_ir::DataType::INT16:
            case nn_ir::DataType::UINT8:
            case nn_ir::DataType::INT8:
                // 16bit integer pixels are separated into two 8-bit pixels
                return getBufSize();
            case nn_ir::DataType::FLOAT32:
            case nn_ir::DataType::FLOAT16:
                return ::getSizeInBytes(getBufSize(), getUnitSize());
            default:
                Log::IR::E() << "data blob doesn't contain this type by getBufSize" << getDataType();
        }
        Log::IR::E() << "data blob doesn't contain this type by getBufSize" << getDataType();
    }

    MEMORY_SIZE_T getBufSize() const {
        switch (getDataType()) {
            case nn_ir::DataType::UINT16:
                return getBuf<uint16_t>().size();
            case nn_ir::DataType::INT16:
                return getBuf<int16_t>().size();
            case nn_ir::DataType::UINT8:
                return getBuf<uint8_t>().size();
            case nn_ir::DataType::INT8:
                return getBuf<int8_t>().size();
            case nn_ir::DataType::FLOAT32:
                return getBuf<float>().size();
            case nn_ir::DataType::FLOAT16:
                return getBuf<float16>().size();
            default:
                Log::IR::E() << "data blob doesn't contain this type by getBufSize" << getDataType();
        }
        Log::IR::E() << "data blob doesn't contain this type by getBufSize" << getDataType();
    }

    template <typename T>
    static bool classof(const Blob* blob) {
        static_assert(std::is_same<T, DataBlob>::value, "incorrect type");
        return (blob->getBlobType() == BlobType::WEIGHT || blob->getBlobType() == BlobType::BIAS ||
                blob->getBlobType() == BlobType::LUT);
    }

    void toByteVector(std::vector<uint8_t>& result) const {
        auto vbuf_visitor = [&result](auto&& dVec) -> void { result.insert(result.end(), dVec.begin(), dVec.end()); };
        std::visit(vbuf_visitor, *buf_ptr_);
    }

    bool isSigned() const {
        switch (getDataType()) {
            case nn_ir::DataType::INT8:
            case nn_ir::DataType::INT16:
                return true;
            case nn_ir::DataType::UINT8:
            case nn_ir::DataType::UINT16:
                return false;
            default:
                Log::IR::E() << "Invalid Data Blob Type " << getDataType();
        }
    }

 private:
    DataBlob(const DataBlob&) = default;
    DataBlob(DataBlob&&)      = default;

    DataBlob* cloneImpl() const& override { return new DataBlob(*this); }
    DataBlob* cloneImpl() && override { return new DataBlob(std::move(*this)); }

    estd::cow_ptr<VAR_TYPE> buf_ptr_;
    estd::cow_ptr<MetaVec>  meta_buf_ptr_;
}; // class DataBlob

/**
 * @brief. A helper function template to run user-provided functor on data_blob's internal buffer with
 *         pregenerated data. This helper allows to avoid "switch-case on data type" code on user side,
 *         making it look like this:
 *
 *         template <typename T>
 *         struct isSupportedType {
 *             static constexpr bool supported_value       = any_of<T, uint8_t, int8_t, uint16_t, int16_t>::value;
 *             static constexpr bool not_implemented_value = any_of<T, half_float::half, float>::value;
 *         };
 *
 *         nn_ir::processDataBlob<addZeroChannelsHelper, isSupportedType>(cast<nn_ir::DataBlob>(kernel_blob));
 *
 *         In the above code, user wants to run `addZeroChannelsHelper` on `kernel_blob` internal buffer if it holds
 *         `uint8_t`, `int8_t`, `uint16_t` or `int16_t` data, bypass (do nothing) if it holds `fp16` data and fail
 *         with error message if it holds some other type of data (e.g. `float`).
 *
 * @requirements. To use the helper below, user should define:
 *
 *         1) A `FunctorT` object template with overloaded `operator()`. The first parameter of `operator()`
 *            must be of type `nn_ir::DataBlob&`.
 *         2) A `type_traits_struct` structure template that defines `supported_value` and
 *            `not_implemented_value` value traits to specify whether we can apply `FunctorT` on given DataBlob
 *            (`supported_value` is true), should not apply it and bypass (`not_implemented_value` is true) or fail
 *            with error (neither is true).
 */
template <template <typename DType> class FunctorT, template <typename T> class type_traits_struct, typename... Args>
void processDataBlob(nn_ir::DataBlob& data_blob, Args&&... args) {
    std::visit(
        [&](auto const& l) {
            using T = typename std::decay_t<decltype(l)>::value_type;
            if constexpr (type_traits_struct<T>::supported_value) {
                FunctorT<T>()(data_blob, std::forward<Args>(args)...);
            } else if constexpr (type_traits_struct<T>::not_implemented_value) {
                Log::IR::I() << "[processDataBlob] Called on unsupported data type" << data_blob.getDataType()
                             << ", do nothing\n";
            } else {
                Log::IR::E() << "[processDataBlob] Unsupported data type " << data_blob.getDataType();
            }
        },
        data_blob.getBuf());
}

} // namespace nn_ir
} // namespace nn_compiler
