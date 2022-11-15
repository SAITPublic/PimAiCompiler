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

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
// TORCH_API Tensor slice:
// (const Tensor & self, int64_t dim=0, c10::optional<int64_t> start=0,
//  c10::optional<int64_t> end=9223372036854775807, int64_t step=1);

class AtenSliceLayer : public NNLayer
{
   public:
    /**
     * @brief AtenSliceLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenSliceLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenSliceLayer(const AtenSliceLayer& aten_slice_layer) : NNLayer(aten_slice_layer)
    {
        this->setAttr(aten_slice_layer.dim_, aten_slice_layer.start_, aten_slice_layer.end_, aten_slice_layer.step_);
    }

    virtual ~AtenSliceLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenSliceLayer>(new AtenSliceLayer(*this)); }

    void printAttr() { DLOG(INFO) << "      AtenSliceAttr      "; }

    void setAttr(int64_t dim, int64_t start, int64_t end, int64_t step)
    {
        this->dim_ = dim;
        this->start_ = start;
        this->end_ = end;
        this->step_ = step;
    }

    std::vector<int64_t> getAttr() { return std::vector<int64_t>{this->dim_, this->start_, this->end_, this->step_}; }

    void setDim(int64_t dim) { this->dim_ = dim; }

    int64_t getDim() { return this->dim_; }

    void setStart(int64_t start) { this->start_ = start; }

    int64_t getStart() { return this->start_; }

    void setEnd(int64_t end) { this->end_ = end; }

    int64_t getEnd() { return this->end_; }

    void setStep(int64_t step) { this->step_ = step; }

    int64_t getStep() { return this->step_; }

   private:
    // Attributes
    int64_t dim_ = INT64_MIN;
    int64_t start_ = INT64_MIN;
    int64_t end_ = INT64_MIN;
    int64_t step_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
