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
// prim::CallMethod(targetNetworkName, inputs)
class PrimCallMethodLayer : public NNLayer
{
   public:
    PrimCallMethodLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit PrimCallMethodLayer(const PrimCallMethodLayer& callmethod_layer) : NNLayer(callmethod_layer)
    {
        this->target_network_name_ = callmethod_layer.target_network_name_;
    }

    virtual ~PrimCallMethodLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<PrimCallMethodLayer>(new PrimCallMethodLayer(*this));
    }

    void printAttr() { DLOG(INFO) << "    target network name is     " << target_network_name_; }

    void setAttr(std::string target_network_name) { target_network_name_ = target_network_name; }

    std::string getAttr() const { return target_network_name_; }

   private:
    std::string target_network_name_;
};

}  // namespace ir
}  // namespace nn_compiler
