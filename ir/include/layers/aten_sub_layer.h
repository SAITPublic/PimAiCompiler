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
// Element-wise sub
// API:
// TORCH_API Tensor sub(const Tensor & self, const Tensor & other, Scalar alpha=1);
// TORCH_API Tensor sub(const Tensor & self, Scalar other, Scalar alpha=1);

class AtenSubLayer : public NNLayer
{
   public:
    AtenSubLayer() {}

    AtenSubLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenSubLayer(const AtenSubLayer &aten_sub_layer) : NNLayer(aten_sub_layer)
    {
        this->alpha_ = aten_sub_layer.alpha_;
    }

    virtual ~AtenSubLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenSubLayer>(new AtenSubLayer(*this)); }

    void printAttr() { DLOG(INFO) << "AtenSubAttr"; }

    void setAlpha(int64_t alpha) { alpha_ = alpha; }

    int64_t getAlpha() { return alpha_; }

   private:
    int64_t alpha_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
