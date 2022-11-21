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
class AtenLeakyReluLayer : public NNLayer
{
   public:
    AtenLeakyReluLayer() {}

    AtenLeakyReluLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenLeakyReluLayer(const AtenLeakyReluLayer &aten_leaky_relu_layer) : NNLayer(aten_leaky_relu_layer)
    {
        this->scalar_ = aten_leaky_relu_layer.scalar_;
    }

    virtual ~AtenLeakyReluLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenLeakyReluLayer>(new AtenLeakyReluLayer(*this));
    }

    void setScalar(double scalar) { scalar_ = scalar; }

    double getScalar() const { return scalar_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenLeakyReluAttr      ";
        DLOG(INFO) << "    sclar is               " << scalar_;
    }

   private:
    double scalar_ = DBL_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
