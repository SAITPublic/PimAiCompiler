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
class AtenTriuLayer : public NNLayer
{
   public:
    AtenTriuLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenTriuLayer(const AtenTriuLayer& aten_triu_layer) : NNLayer(aten_triu_layer)
    {
        this->Diagonal_ = aten_triu_layer.Diagonal_;
    }

    virtual ~AtenTriuLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenTriuLayer>(new AtenTriuLayer(*this)); }

    void setDiagonal(int64_t Diagonal) { Diagonal_ = Diagonal; }

    int64_t getDiagonal() const { return Diagonal_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenTriuAttr     ";
        DLOG(INFO) << "    Diagonal is           " << Diagonal_;
    }

   private:
    int64_t Diagonal_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
