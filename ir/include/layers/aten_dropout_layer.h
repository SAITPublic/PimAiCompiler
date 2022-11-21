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
class AtenDropoutLayer : public NNLayer
{
   public:
    AtenDropoutLayer() {}

    AtenDropoutLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenDropoutLayer(const AtenDropoutLayer& aten_drop_layer) : NNLayer(aten_drop_layer)
    {
        this->proportion_ = aten_drop_layer.proportion_;
        this->train_ = aten_drop_layer.train_;
    }

    virtual ~AtenDropoutLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenDropoutLayer>(new AtenDropoutLayer(*this)); }

    void setProportion(double proportion) { proportion_ = proportion; }
    void setTrain(int train) { train_ = train; }
    double getProportion() { return proportion_; }
    int getTrain() { return train_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenDropoutAttr            ";
        DLOG(INFO) << "    proportion is              " << proportion_;
        DLOG(INFO) << "    train value is             " << train_;
    }

   private:
    double proportion_ = DBL_MAX;
    int train_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
