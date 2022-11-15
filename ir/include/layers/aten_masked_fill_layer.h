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
class AtenMaskedFillLayer : public NNLayer
{
   public:
    AtenMaskedFillLayer() {}

    AtenMaskedFillLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenMaskedFillLayer(const AtenMaskedFillLayer &aten_masked_fill_layer) : NNLayer(aten_masked_fill_layer)
    {
        this->is_inplace_ = aten_masked_fill_layer.is_inplace_;
    }

    virtual ~AtenMaskedFillLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenMaskedFillLayer>(new AtenMaskedFillLayer(*this));
    }

    void setIsInplace(bool is_inplace) { is_inplace_ = is_inplace; }

    bool getIsInplace() const { return is_inplace_; }

    void printAttr() { DLOG(INFO) << "    AtenMaskedFillAttr      "; }

   private:
    bool is_inplace_ = false;
};

}  // namespace ir
}  // namespace nn_compiler
