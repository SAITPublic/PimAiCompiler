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
class AtenMinLayer : public NNLayer
{
   public:
    AtenMinLayer() {}

    AtenMinLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenMinLayer(const AtenMinLayer &aten_min_layer) : NNLayer(aten_min_layer)
    {
        this->dim_or_y_ = aten_min_layer.dim_or_y_;
        this->_keep_dim = aten_min_layer._keep_dim;
    }

    virtual ~AtenMinLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenMinLayer>(new AtenMinLayer(*this)); }

    void setDimOrY(int dim_or_y) { dim_or_y_ = dim_or_y; }

    void setKeepDim(int keep_dim) { _keep_dim = keep_dim; }

    int getDimOrY() { return dim_or_y_; }

    int getKeepDim() { return _keep_dim; }

    void printAttr()
    {
        DLOG(INFO) << " AtemMinAttr ";
        DLOG(INFO) << " dim_or_y is " << dim_or_y_;
        DLOG(INFO) << " keepdim is  " << _keep_dim;
    }

   private:
    int dim_or_y_ = INT32_MAX;
    int _keep_dim = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
