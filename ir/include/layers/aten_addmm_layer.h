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
class AtenAddmmLayer : public NNLayer
{
   public:
    /**
     * @brief AtenAddmmLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenAddmmLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenAddmmLayer(const AtenAddmmLayer& aten_addmm_layer) : NNLayer(aten_addmm_layer)
    {
        act_type_ = aten_addmm_layer.act_type_;
    }

    virtual ~AtenAddmmLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenAddmmLayer>(new AtenAddmmLayer(*this)); }

    void printAttr() { DLOG(INFO) << "    AtenAddmmAttr      "; }

    void set_act_type(const std::string& type) { act_type_ = type; }

    std::string get_act_type() { return act_type_; }

   private:
    std::string act_type_ = "none";
};

}  // namespace ir
}  // namespace nn_compiler
