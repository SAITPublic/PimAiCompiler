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
class AtenIndexPutLayer : public NNLayer
{
   public:
    AtenIndexPutLayer() {}

    AtenIndexPutLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenIndexPutLayer(const AtenIndexPutLayer &aten_index_put_layer) : NNLayer(aten_index_put_layer)
    {
        this->accumulate_ = aten_index_put_layer.accumulate_;
    }

    virtual ~AtenIndexPutLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenIndexPutLayer>(new AtenIndexPutLayer(*this));
    }

    void setAccumulate(int accumulate) { accumulate_ = accumulate; }

    int getAccumulate() const { return accumulate_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenIndexPutAttr             ";
        DLOG(INFO) << "    accumulate is                " << accumulate_;
    }

   private:
    int accumulate_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
