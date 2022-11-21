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
class PrimToListLayer : public NNLayer
{
   public:
    /**
     * @brief PrimToListLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    PrimToListLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit PrimToListLayer(const PrimToListLayer& prim_to_list_layer) : NNLayer(prim_to_list_layer)
    {
        element_type_ = prim_to_list_layer.element_type_;
        dim_ = prim_to_list_layer.dim_;
    }

    virtual ~PrimToListLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<PrimToListLayer>(new PrimToListLayer(*this)); }

    void setElementType(int64_t element_type) { element_type_ = element_type; }

    int64_t getElementType() { return element_type_; }

    void setDim(int64_t dim) { dim_ = dim; }

    int64_t getDim() { return dim_; }

    void printAttr()
    {
        DLOG(INFO) << "PrimToListAttr    ";
        DLOG(INFO) << "element_type is   " << element_type_;
        DLOG(INFO) << "dim is            " << dim_;
    }

   private:
    int64_t element_type_ = INT64_MIN;
    int64_t dim_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
