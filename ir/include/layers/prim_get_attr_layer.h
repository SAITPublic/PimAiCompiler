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
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace ir
{
class PrimGetAttrLayer : public NNLayer
{
   public:
    /**
     * @brief PrimGetAttrLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */

    PrimGetAttrLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit PrimGetAttrLayer(const PrimGetAttrLayer& get_attr_layer) : NNLayer(get_attr_layer)
    {
        this->values_ = get_attr_layer.values_;
        this->ntype_ = get_attr_layer.ntype_;
    }

    virtual ~PrimGetAttrLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<PrimGetAttrLayer>(new PrimGetAttrLayer(*this)); }

    void printAttr() { DLOG(INFO) << "      PrimGetAttrAttr     "; }

   private:
    std::vector<std::shared_ptr<DTensor>> values_;
    std::string ntype_;
};

}  // namespace ir
}  // namespace nn_compiler
