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
class PrimConstantLayer : public NNLayer
{
   public:
    /**
     * @brief PrimConstantLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */

    PrimConstantLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit PrimConstantLayer(const PrimConstantLayer& constant_layer) : NNLayer(constant_layer)
    {
        value_ = constant_layer.value_;
        ntype_ = constant_layer.ntype_;
        to_remove_ = constant_layer.to_remove_;
    }

    virtual ~PrimConstantLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<PrimConstantLayer>(new PrimConstantLayer(*this));
    }

    void printAttr() { DLOG(INFO) << "      PrimConstantAttr     "; }

    void setAttr(std::shared_ptr<DTensor> data) { value_ = *data; }

    std::shared_ptr<DTensor> getAttr() { return value_.clone(); }

    void setNType(std::string& ntype) { ntype_ = ntype; }

    std::string getNType() const { return ntype_; }

    void setToRemove(bool to_remove) { to_remove_ = to_remove; }

    bool getToRemove() { return to_remove_; }

   private:
    // type : str/device/int/bool/float/Tensor/None/
    // use DTensor store all the type.
    DTensor value_;
    std::string ntype_;
    bool to_remove_ = false;
};

}  // namespace ir
}  // namespace nn_compiler
