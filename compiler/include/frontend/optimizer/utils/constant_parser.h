#pragma once

#include "half.hpp"

#include "ir/include/nn_network.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
class ConstantParser
{
   public:
    ConstantParser();

    template <typename T>
    std::vector<std::vector<T>> parse(const std::shared_ptr<nn_compiler::ir::DTensor>& dtensor)
    {
        auto stride = dtensor->getStride();
        auto x_stride = stride[2], y_stride = stride[3];
        auto origin_data = dtensor->getData<T>();
        std::vector<int> shape;
        auto b = dtensor->getTensorShape().getBatch();
        auto c = dtensor->getTensorShape().getChannel();
        auto h = dtensor->getTensorShape().getHeight();
        auto w = dtensor->getTensorShape().getWidth();
        if (b != 0) shape.push_back(b);
        if (c != 0) shape.push_back(c);
        if (h != 0) shape.push_back(h);
        if (w != 0) shape.push_back(w);

        std::vector<std::vector<T>> matrix;

        int cnt = 0;
        if (x_stride == 0 || y_stride == 0) {
            for (auto i = 0; i < shape[0]; i++) {
                std::vector<T> vec;
                for (auto j = 0; j < shape[1]; j++) {
                    vec.push_back((*origin_data)[cnt++]);
                }
                matrix.push_back(vec);
            }
        } else {
            for (auto i = 0; i < shape[0]; i++) {
                std::vector<T> vec;
                for (auto j = 0; j < shape[1]; j++) {
                    auto idx = i * x_stride + j * y_stride;
                    vec.push_back((*origin_data)[idx]);
                }
                matrix.push_back(vec);
            }
        }

        return matrix;
    }
};

}  // namespace frontend
}  // namespace nn_compiler
