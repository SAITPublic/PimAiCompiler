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
class AtenAddLayer : public NNLayer
{
   public:
    /**
     * @brief AtenAddLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenAddLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenAddLayer(const AtenAddLayer& aten_add_layer) : NNLayer(aten_add_layer)
    {
        this->alpha_ = aten_add_layer.alpha_;
    }

    virtual ~AtenAddLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenAddLayer>(new AtenAddLayer(*this)); }

    void printAttr() { DLOG(INFO) << "    AtenAddAttr      "; }

    void setAlpha(int64_t alpha) { alpha_ = alpha; }

    int64_t getAlpha() { return alpha_; }

   private:
    int64_t alpha_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
