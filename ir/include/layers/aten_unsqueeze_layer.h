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
class AtenUnsqueezeLayer : public NNLayer
{
   public:
    /**
     * @brief AtenUnsqueezeLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenUnsqueezeLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenUnsqueezeLayer(const AtenUnsqueezeLayer& aten_unsqueeze_layer) : NNLayer(aten_unsqueeze_layer)
    {
        this->dim_ = aten_unsqueeze_layer.dim_;
        this->is_inplace_ = aten_unsqueeze_layer.is_inplace_;
    }

    virtual ~AtenUnsqueezeLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenUnsqueezeLayer>(new AtenUnsqueezeLayer(*this));
    }

    void setDim(int64_t dim) { dim_ = dim; }

    void setIsInplace(bool is_inplace) { is_inplace_ = is_inplace; }

    int64_t getDim() const { return dim_; }

    bool getIsInplace() const { return is_inplace_; }

    void printAttr()
    {
        DLOG(INFO) << "     AtenUnsqueezeAttr      ";
        DLOG(INFO) << "     dim is                 " << dim_;
        DLOG(INFO) << "     is_inplace is          " << is_inplace_;
    }

   private:
    int64_t dim_ = INT64_MIN;
    bool is_inplace_ = false;
};

}  // namespace ir
}  // namespace nn_compiler
