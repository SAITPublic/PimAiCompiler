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
// def zeros(g, sizes, dtype, layout, device, pin_memory=False)

class PrimIfLayer : public NNLayer
{
   public:
    /**
     * @brief PrimIfLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    PrimIfLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit PrimIfLayer(const PrimIfLayer& prim_if_layer) : NNLayer(prim_if_layer)
    {
        this->then_net_ = prim_if_layer.then_net_;
        this->else_net_ = prim_if_layer.else_net_;
        this->else_net_start_layer_ = prim_if_layer.else_net_start_layer_;
    }

    virtual ~PrimIfLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<PrimIfLayer>(new PrimIfLayer(*this)); }

    void setThenNet(const std::string netName) { then_net_ = netName; }

    void setElseNet(const std::string netName) { else_net_ = netName; }

    const std::string getThenNet() const { return then_net_; }

    const std::string getElseNet() const { return else_net_; }

    void setElseNetStartLayer(int64_t else_net_start_layer) { else_net_start_layer_ = else_net_start_layer; }

    int64_t getElseNetStartLayer() { return else_net_start_layer_; }

    void printAttr()
    {
        DLOG(INFO) << "    PrimIfAttr        ";
        DLOG(INFO) << "    Then net is                   " << then_net_;
        DLOG(INFO) << "    Else net is                   " << else_net_;
        DLOG(INFO) << "    Id of Else net start layer is " << else_net_start_layer_;
    }

   private:
    std::string then_net_;
    std::string else_net_;

    int64_t else_net_start_layer_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
