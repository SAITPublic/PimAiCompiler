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
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace ir
{
/*
  len(args) == 5:
        aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
 */
class AtenArange1Layer : public NNLayer
{
   public:
    AtenArange1Layer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenArange1Layer(const AtenArange1Layer& aten_arange_layer) : NNLayer(aten_arange_layer)
    {
        this->end_ = aten_arange_layer.end_;
        this->dtype_ = aten_arange_layer.dtype_;
        this->layout_ = aten_arange_layer.layout_;
        this->device_ = aten_arange_layer.device_;
        this->pin_memory_ = aten_arange_layer.pin_memory_;
    }

    virtual ~AtenArange1Layer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenArange1Layer>(new AtenArange1Layer(*this)); }

    void setEnd(int64_t end) { end_ = end; }

    void setDtype(int64_t dtype) { dtype_ = dtype; }

    void setLayout(int64_t layout) { layout_ = layout; }

    void setDevice(std::string device) { device_ = device; }

    void setPinMemory(int pin_memory) { pin_memory_ = pin_memory; }

    int64_t getEnd() const { return end_; }

    int64_t getDtype() const { return dtype_; }

    int64_t getLayout() const { return layout_; }

    std::string getDevice() const { return device_; }

    int getPinMemory() const { return pin_memory_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenArangeAttr          ";
        DLOG(INFO) << "    end is                 " << end_;
        DLOG(INFO) << "    dtype is                " << dtype_;
        DLOG(INFO) << "    layout is               " << layout_;
        DLOG(INFO) << "    pin_memory is           " << pin_memory_;
    }

   private:
    int64_t end_ = INT64_MIN;
    int64_t dtype_ = INT64_MIN;
    int64_t layout_ = INT64_MIN;
    std::string device_ = "";
    int pin_memory_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
