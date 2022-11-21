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
class AtenOneHotLayer : public NNLayer
{
   public:
    AtenOneHotLayer() {}

    AtenOneHotLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenOneHotLayer(const AtenOneHotLayer& aten_one_hot_layer) : NNLayer(aten_one_hot_layer)
    {
        this->num_classes_ = aten_one_hot_layer.num_classes_;
    }

    virtual ~AtenOneHotLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenOneHotLayer>(new AtenOneHotLayer(*this)); }

    void setNumClasses(int64_t num_classes) { num_classes_ = num_classes; }

    int64_t getNumClasses() const { return num_classes_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenOneHotAttr   ";
        DLOG(INFO) << "    num_classes_ is       " << num_classes_;
    }

   private:
    int64_t num_classes_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
