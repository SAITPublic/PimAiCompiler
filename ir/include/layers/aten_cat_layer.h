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
class AtenCatLayer : public NNLayer
{
   public:
    AtenCatLayer() {}

    AtenCatLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenCatLayer(const AtenCatLayer& aten_cat_layer) : NNLayer(aten_cat_layer)
    {
        this->dim_ = aten_cat_layer.dim_;
        this->mem_layer_id_ = aten_cat_layer.mem_layer_id_;
    }

    virtual ~AtenCatLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenCatLayer>(new AtenCatLayer(*this)); }

    void setDim(int dim) { dim_ = dim; }
    int getDim() { return dim_; }

    void setMemLayerId(int64_t mem_layer_id) { mem_layer_id_ = mem_layer_id; }
    int64_t getMemLayerId() { return mem_layer_id_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenCatAttr       ";
        DLOG(INFO) << "    dim is            " << dim_;
    }

   private:
    int dim_ = INT32_MAX;
    int64_t mem_layer_id_ = -1;
};

}  // namespace ir
}  // namespace nn_compiler
