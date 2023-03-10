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
#include "ir/include/types.h"

namespace nn_compiler
{
namespace ir
{
// def zeros(g, sizes, dtype, layout, device, pin_memory=False)

class PrimEndIfLayer : public NNLayer
{
   public:
    /**
     * @brief PrimIfLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    PrimEndIfLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit PrimEndIfLayer(const PrimEndIfLayer& prim_end_if_layer) : NNLayer(prim_end_if_layer)
    {
        goto_layer_ = prim_end_if_layer.goto_layer_;
        is_else_net_ = prim_end_if_layer.is_else_net_;
        if_layer_id_ = prim_end_if_layer.if_layer_id_;
    }

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<PrimEndIfLayer>(new PrimEndIfLayer(*this)); }

    void setGotoLayer(int64_t goto_layer) { goto_layer_ = goto_layer; }
    void setIsElseNet(bool is_else_net) { is_else_net_ = is_else_net; }
    void setIfLayerId(int64_t if_layer_id) { if_layer_id_ = if_layer_id; }

    int64_t getGotoLayer() const { return goto_layer_; }
    bool getIsElseNet() const { return is_else_net_; }
    int64_t getIfLayerId() const { return if_layer_id_; }

    void printAttr()
    {
        DLOG(INFO) << "    PrimEndIf Attr  ";
        DLOG(INFO) << "    goto_layer      " << goto_layer_;
        DLOG(INFO) << "    is_else_net     " << is_else_net_;
        DLOG(INFO) << "    if_layer_id     " << if_layer_id_;
    }

   private:
    int64_t goto_layer_ = INT64_MIN;
    bool is_else_net_ = false;
    int64_t if_layer_id_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
