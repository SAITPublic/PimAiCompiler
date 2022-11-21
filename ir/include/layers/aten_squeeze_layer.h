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
class AtenSqueezeLayer : public NNLayer
{
   public:
    AtenSqueezeLayer() {}

    AtenSqueezeLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenSqueezeLayer(const AtenSqueezeLayer& aten_squeeze_layer) : NNLayer(aten_squeeze_layer)
    {
        this->dim_ = aten_squeeze_layer.dim_;
    }

    virtual ~AtenSqueezeLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenSqueezeLayer>(new AtenSqueezeLayer(*this)); }

    void setDim(int64_t dim) { dim_ = dim; }

    int64_t getDim() const { return dim_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenSqueezeAttr    ";
        DLOG(INFO) << "    dim is              " << dim_;
    }

   private:
    int64_t dim_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
