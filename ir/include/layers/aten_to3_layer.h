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
// Tensor to(const Tensor & self, const Tensor & other,
//     bool non_blocking=false, bool copy=false, c10::optional<MemoryFormat> memory_format=c10::nullopt);

class AtenTo3Layer : public NNLayer
{
   public:
    AtenTo3Layer() {}

    AtenTo3Layer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenTo3Layer(const AtenTo3Layer& aten_to_layer) : NNLayer(aten_to_layer)
    {
        this->non_blocking_ = aten_to_layer.non_blocking_;
        this->copy_ = aten_to_layer.copy_;
        this->optional_memory_format_ = aten_to_layer.optional_memory_format_;
    }

    virtual ~AtenTo3Layer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenTo3Layer>(new AtenTo3Layer(*this)); }

    void setNonBlocking(int nonblocking) { non_blocking_ = nonblocking; }

    void setCopy(int copy) { copy_ = copy; }

    void setOptionalMemoryFormat(int optional_memory_format) { optional_memory_format_ = optional_memory_format; }

    int getNonBlocking() const { return non_blocking_; }

    int getCopy() const { return copy_; }

    int getOptionalMemoryFormat() { return optional_memory_format_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenTo3Attr                    ";
        DLOG(INFO) << "    non_blocking                   " << non_blocking_;
        DLOG(INFO) << "    copy is                        " << copy_;
        DLOG(INFO) << "    optional_memory_format is      " << optional_memory_format_;
    }

   private:
    int non_blocking_ = INT32_MAX;
    int copy_ = INT32_MAX;
    /* according to pytorch/c10/core/MemoryFormat.h,
       enum MemoryFormat: { Contiguous, Preserve, ChannelsLast, ChannelsLast3d }
       -1 stands for optional_memory_format_ = NONE.
     */
    int optional_memory_format_ = -1;
};

}  // namespace ir
}  // namespace nn_compiler
