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
// Tensor[] = aten::_set_item(const Tensor& self, int indices, Tensor v)
class AtenSetItemLayer : public NNLayer
{
   public:
    /**
     * @brief AtenSetItemLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenSetItemLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenSetItemLayer(const AtenSetItemLayer& aten_set_item_layer) : NNLayer(aten_set_item_layer)
    {
        this->indices_ = aten_set_item_layer.indices_;
    }

    virtual ~AtenSetItemLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenSetItemLayer>(new AtenSetItemLayer(*this)); }

    void setIndices(int indices) { indices_ = indices; }

    int getIndices() const { return indices_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenSetItemAttr     ";
        DLOG(INFO) << "    indices is          " << indices_;
    }

   private:
    int indices_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
