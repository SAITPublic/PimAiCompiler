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
class AtenGetItemLayer : public NNLayer
{
   public:
    /**
     * @brief AtenGetItemLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenGetItemLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenGetItemLayer(const AtenGetItemLayer& aten_get_item_layer) : NNLayer(aten_get_item_layer)
    {
        this->idx_ = aten_get_item_layer.idx_;
    }

    virtual ~AtenGetItemLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenGetItemLayer>(new AtenGetItemLayer(*this)); }

    void setIdx(int idx) { idx_ = idx; }

    int getIdx() { return idx_; }

    void printAttr() { DLOG(INFO) << "    AtenGetItemAttr      "; }

   private:
    int idx_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
