/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/include/nn_nodes/aten_embedding_node.hpp"
#include "ir/include/common/log.hpp"

namespace nn_compiler {
namespace nn_ir {

void AtenEmbeddingNode::setWeights(std::vector<float16> weights_data, nn_ir::Shape2D weights_shape) {
    weights_.clear();
    auto count = 0;
    for (auto h = 0; h < weights_shape.h; h++) {
        std::vector<float16> data;
        for (auto w = 0; w < weights_shape.w; w++) {
            data.push_back(weights_data[count++]);
        }
        auto data_tensor_cpu =torch::from_blob(data.data(), {1, 1, weights_shape.w},
                                               at::TensorOptions().dtype(torch::kFloat16));
        auto data_tensor = std::move(data_tensor_cpu.cuda());
        weights_.push_back(data_tensor);
    }
}

} // namespace nn_ir
} // namespace nn_compiler
