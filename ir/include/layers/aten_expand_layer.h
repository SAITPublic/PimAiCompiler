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
class AtenExpandLayer : public NNLayer
{
   public:
    AtenExpandLayer() {}

    AtenExpandLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenExpandLayer(const AtenExpandLayer &aten_expand_layer) : NNLayer(aten_expand_layer)
    {
        this->implicit_ = aten_expand_layer.implicit_;
    }

    virtual ~AtenExpandLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenExpandLayer>(new AtenExpandLayer(*this)); }

    void setImplicit(int implicit) { implicit_ = implicit; }

    int getImplicit() { return implicit_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenExpandAttr      ";
        DLOG(INFO) << "    implicit is         " << implicit_;
    }

   private:
    int implicit_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
