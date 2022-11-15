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
class PrimTupleIndexLayer : public NNLayer
{
   public:
    /**
     * @brief PrimTupleIndexLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    PrimTupleIndexLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit PrimTupleIndexLayer(const PrimTupleIndexLayer& prim_tuple_index_layer) : NNLayer(prim_tuple_index_layer)
    {
        this->index_ = prim_tuple_index_layer.index_;
    }

    virtual ~PrimTupleIndexLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<PrimTupleIndexLayer>(new PrimTupleIndexLayer(*this));
    }

    int64_t getIndex() const { return index_; }

    void setIndex(int64_t index) { index_ = index; }

    void printAttr()
    {
        DLOG(INFO) << "   PrimTupleIndexAttr    ";
        DLOG(INFO) << "   index is    " << index_;
    }

   private:
    int64_t index_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
