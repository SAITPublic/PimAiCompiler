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
class PrimLoopLayer : public NNLayer
{
   public:
    /**
     * @brief PrimLoopLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    PrimLoopLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit PrimLoopLayer(const PrimLoopLayer& prim_loop_layer) : NNLayer(prim_loop_layer)
    {
        this->trip_count_ = prim_loop_layer.trip_count_;
        this->cond_ = prim_loop_layer.cond_;
        this->body_net_ = prim_loop_layer.body_net_;
        this->goto_layer_ = prim_loop_layer.goto_layer_;
    }

    virtual ~PrimLoopLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<PrimLoopLayer>(new PrimLoopLayer(*this)); }

    void setTripCount(const int64_t trip_count) { trip_count_ = trip_count; }

    void setCond(int64_t cond) { cond_ = cond; }

    void setBodyNet(const std::string body_net) { body_net_ = body_net; }

    const int64_t getTripCount() { return trip_count_; }

    int64_t getCond() { return cond_; }

    const std::string getBodyNet() { return body_net_; }

    void setGotoLayer(int64_t goto_layer) { goto_layer_ = goto_layer; }

    int64_t getGotoLayer() { return goto_layer_; }

    void printAttr()
    {
        DLOG(INFO) << "     PrimLoopAttr    ";
        DLOG(INFO) << "     body_net        " << body_net_;
        DLOG(INFO) << "     trip_count      " << trip_count_;
        DLOG(INFO) << "     cond            " << cond_;
        DLOG(INFO) << "     goto_layer      " << goto_layer_;
    }

   private:
    int64_t trip_count_ = INT64_MIN;
    int64_t cond_ = INT64_MIN;
    std::string body_net_;

    int64_t goto_layer_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
