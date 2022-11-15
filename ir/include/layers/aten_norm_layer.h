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
class AtenNormLayer : public NNLayer
{
   public:
    AtenNormLayer() {}

    AtenNormLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenNormLayer(const AtenNormLayer& aten_norm_layer) : NNLayer(aten_norm_layer)
    {
        this->p_ = aten_norm_layer.p_;
    }

    virtual ~AtenNormLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenNormLayer>(new AtenNormLayer(*this)); }

    void setP(int64_t p) { p_ = p; }

    int64_t getP() { return p_; }

    void printAttr() { DLOG(INFO) << "    AtenNormAttr      "; }

   private:
    int64_t p_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
