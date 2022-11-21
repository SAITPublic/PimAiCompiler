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
// Tensor select(const Tensor& self, int64_t dim, int64_t index)
class AtenSelectLayer : public NNLayer
{
   public:
    /**
     * @brief AtenSelectLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenSelectLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenSelectLayer(const AtenSelectLayer& aten_select_layer) : NNLayer(aten_select_layer)
    {
        this->dim_ = aten_select_layer.dim_;
        this->index_ = aten_select_layer.index_;
    }

    virtual ~AtenSelectLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenSelectLayer>(new AtenSelectLayer(*this)); }

    void setDim(int64_t dim) { dim_ = dim; }

    int64_t getDim() const { return dim_; }

    void setIndex(int64_t index) { index_ = index; }

    int64_t getIndex() const { return index_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenSelectAttr     ";
        DLOG(INFO) << "    dim is             " << dim_;
        DLOG(INFO) << "    index is           " << index_;
    }

   private:
    int64_t dim_ = INT64_MIN;
    int64_t index_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
