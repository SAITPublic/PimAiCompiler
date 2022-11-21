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
class AtenCloneLayer : public NNLayer
{
   public:
    AtenCloneLayer() {}

    AtenCloneLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenCloneLayer(const AtenCloneLayer& aten_clone_layer) : NNLayer(aten_clone_layer)
    {
        this->memory_format_ = aten_clone_layer.memory_format_;
    }

    virtual ~AtenCloneLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenCloneLayer>(new AtenCloneLayer(*this)); }

    void setMemoryFormat(int memory_format) { memory_format_ = memory_format; }

    int getMemoryFormat() { return memory_format_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenCloneAttr          ";
        DLOG(INFO) << "    memory_format is       " << memory_format_;
    }

   private:
    int memory_format_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
