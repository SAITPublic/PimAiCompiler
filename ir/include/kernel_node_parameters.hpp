/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "ir/include/ir_includes.hpp"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {

class KernelNodeParameters {
 public:
    KernelNodeParameters() = default;

    KernelNodeParameters(
        Pad4 padding_size, Shape2D kernel_size, Shape2D origin_kernel_size, Shape2D stride_size, Shape2D dilation_size)
        : padding_size_(padding_size), kernel_size_(kernel_size), origin_kernel_size_(origin_kernel_size),
          stride_size_(stride_size), dilation_size_(dilation_size) {}

    KernelNodeParameters(Pad4 padding_size, Shape2D kernel_size, Shape2D stride_size, Shape2D dilation_size)
        : padding_size_(padding_size), kernel_size_(kernel_size), origin_kernel_size_(kernel_size),
          stride_size_(stride_size), dilation_size_(dilation_size) {}

    Pad4    getPaddingSize() const { return padding_size_; }
    Shape2D getKernelSize() const { return kernel_size_; }
    Shape2D getOriginKernelSize() const { return origin_kernel_size_; }
    Shape2D getStrideSize() const { return stride_size_; }
    Shape2D getDilationSize() const { return dilation_size_; }

    void setPaddingSize(Pad4 padding_size) { padding_size_ = padding_size; }

 private:
    Pad4    padding_size_       = {0, 0, 0, 0};
    Shape2D kernel_size_        = {{.h = 1, .w = 1}};
    Shape2D origin_kernel_size_ = {{.h = 1, .w = 1}};
    Shape2D stride_size_        = {{.h = 1, .w = 1}};
    Shape2D dilation_size_      = {{.h = 1, .w = 1}};
};

} // namespace nn_ir
} // namespace nn_compiler
