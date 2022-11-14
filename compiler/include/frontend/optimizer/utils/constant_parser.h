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

#include "half.hpp"

#include "ir/include/nn_model.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
namespace optimizer_utils
{
class ConstantParser
{
   public:
    ConstantParser() = default;

    template <typename T>
    std::vector<std::vector<T>> parse(const std::shared_ptr<nn_compiler::ir::DTensor>& dtensor)
    {
        auto stride = dtensor->getStride();
        auto x_stride = stride[2], y_stride = stride[3];
        auto origin_data = dtensor->getData<T>();
        auto shape = dtensor->getTensorShape().getDims();
        assert(shape.size() == 4);  // n, c, h, w

        std::vector<std::vector<T>> matrix;

        int cnt = 0;
        if (x_stride == 0 || y_stride == 0) {
            for (auto i = 0; i < shape[2]; i++) {
                std::vector<T> vec;
                for (auto j = 0; j < shape[3]; j++) {
                    vec.push_back((*origin_data)[cnt++]);
                }
                matrix.push_back(vec);
            }
        } else {
            for (auto i = 0; i < shape[2]; i++) {
                std::vector<T> vec;
                for (auto j = 0; j < shape[3]; j++) {
                    auto idx = i * x_stride + j * y_stride;
                    vec.push_back((*origin_data)[idx]);
                }
                matrix.push_back(vec);
            }
        }

        return matrix;
    }
};

}  // namespace optimizer_utils
}  // namespace frontend
}  // namespace nn_compiler
