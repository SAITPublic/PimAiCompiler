/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
// Tensor = aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first,
//                                    Scalar padding_value, int total_length)
class AtenPadPackedSequenceLayer : public NNLayer
{
   public:
    /**
     * @brief AtenPadPackedSequenceLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenPadPackedSequenceLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenPadPackedSequenceLayer(const AtenPadPackedSequenceLayer& aten_pad_packed_sequence_layer)
        : NNLayer(aten_pad_packed_sequence_layer)
    {
        this->batch_first_ = aten_pad_packed_sequence_layer.batch_first_;
        this->padding_value_ = aten_pad_packed_sequence_layer.padding_value_;
        this->total_length_ = aten_pad_packed_sequence_layer.total_length_;
    }

    virtual ~AtenPadPackedSequenceLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenPadPackedSequenceLayer>(new AtenPadPackedSequenceLayer(*this));
    }

    void setBatchFirst(int batch_first) { batch_first_ = batch_first; }

    int getBatchFirst() const { return batch_first_; }

    void setPaddingValue(float padding_value) { padding_value_ = padding_value; }

    float getPaddingValue() const { return padding_value_; }

    void setTotalLength(int64_t total_length) { total_length_ = total_length; }

    int64_t getTotalLength() const { return total_length_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenPadPackedSequenceAttr     ";
        DLOG(INFO) << "    batch_first is                 " << batch_first_;
        DLOG(INFO) << "    padding_value is               " << padding_value_;
        DLOG(INFO) << "    total_length is                " << total_length_;
    }

   private:
    int batch_first_ = INT32_MAX;
    float padding_value_ = FLT_MAX;
    int64_t total_length_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
