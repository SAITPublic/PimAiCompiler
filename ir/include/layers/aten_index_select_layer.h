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
class AtenIndexSelectLayer : public NNLayer
{
   public:
    AtenIndexSelectLayer() {}

    AtenIndexSelectLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenIndexSelectLayer(const AtenIndexSelectLayer& aten_index_select_layer)
        : NNLayer(aten_index_select_layer)
    {
        this->dim_ = aten_index_select_layer.dim_;
    }

    virtual ~AtenIndexSelectLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenIndexSelectLayer>(new AtenIndexSelectLayer(*this));
    }

    void setDim(int dim) { dim_ = dim; }

    int getDim() const { return dim_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenIndexSelectAttr     ";
        DLOG(INFO) << "    dim is                  " << dim_;
    }

   private:
    int dim_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
