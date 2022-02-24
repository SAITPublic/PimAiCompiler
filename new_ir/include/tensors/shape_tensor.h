/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include <atomic>

#include "new_ir/include/types.h"

namespace nn_compiler
{
namespace ir
{

class STensor
{
   public:
    STensor(uint32_t batch, uint32_t channel, uint32_t height, uint32_t width)
        : n_(batch), c_(channel), h_(height), w_(width)
    {
        setID(getNextId());
    }

    STensor() : STensor{0, 0, 0, 0} {}

    uint32_t getID() { return id_; }

    void setBatch(uint32_t batch) { n_ = batch; }
    uint32_t getBatch() const { return n_; }

    void setChannel(uint32_t channel) { c_ = channel; }
    uint32_t getChannel() const { return c_; }

    void setHeight(uint32_t height) { h_ = height; }
    uint32_t getHeight() const { return h_; }

    void setWidth(uint32_t width) { w_ = width; }
    uint32_t getWidth() const { return w_; }

   private:
    static uint32_t getNextId()
    {
        static std::atomic<uint32_t> id{0};
        return id++;
    }

    uint32_t id_;
    void setID(uint32_t id) { id_ = id; }

    uint32_t n_ = 0;
    uint32_t c_ = 0;
    uint32_t h_ = 0;
    uint32_t w_ = 0;
};

}  // namespace ir
}  // namespace nn_compiler
