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
class AtenMaxLayer : public NNLayer
{
   public:
    AtenMaxLayer() {}

    AtenMaxLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenMaxLayer(const AtenMaxLayer &aten_max_layer) : NNLayer(aten_max_layer)
    {
        this->dim_ = aten_max_layer.dim_;
        this->keep_dim_ = aten_max_layer.keep_dim_;
    }

    virtual ~AtenMaxLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenMaxLayer>(new AtenMaxLayer(*this)); }

    void setDim(int64_t dim) { dim_ = dim; }

    int64_t getDim() { return dim_; }

    void setKeepDim(int keep_dim) { keep_dim_ = keep_dim; }

    int getKeepDim() { return keep_dim_; }

    void printAttr() { DLOG(INFO) << "   AtemMaxAttr   "; }

   private:
    int64_t dim_ = INT64_MIN;
    int keep_dim_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
