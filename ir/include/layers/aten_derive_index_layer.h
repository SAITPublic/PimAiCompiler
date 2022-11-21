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
class AtenDeriveIndexLayer : public NNLayer
{
   public:
    /**
     * @brief AtenDeriveIndexLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenDeriveIndexLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenDeriveIndexLayer(const AtenDeriveIndexLayer& aten_derive_index_layer)
        : NNLayer(aten_derive_index_layer)
    {
        this->start_ = aten_derive_index_layer.start_;
        this->step_ = aten_derive_index_layer.step_;
    }

    virtual ~AtenDeriveIndexLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenDeriveIndexLayer>(new AtenDeriveIndexLayer(*this));
    }

    void setStep(int64_t step) { step_ = step; }

    int64_t getStep() const { return step_; }

    void setStart(int64_t start) { start_ = start; }

    int64_t getStart() const { return start_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenDeriveIndexAttr      ";
        DLOG(INFO) << "    start is                 " << start_;
        DLOG(INFO) << "    step is                  " << step_;
    }

   private:
    int64_t start_ = INT64_MIN;
    int64_t step_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
