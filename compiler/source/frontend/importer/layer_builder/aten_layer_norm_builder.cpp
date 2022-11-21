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

#include "frontend/importer/layer_builder/layer_builder.h"
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace frontend
{
std::shared_ptr<ir::NNLayer> AtenLayerNormBuilder::buildLayer(const torch::jit::Node* node_ref)
{
    DLOG(INFO) << "build aten::layer_norm";

    nn_compiler::ir::LayerType type = nn_compiler::ir::LayerType::ATENLAYERNORM;
    std::string name = "";

    aten_layer_norm_layer_ = std::make_shared<ir::AtenLayerNormLayer>(name, type);

    auto weight_bias = parser()->getGeneralWeightAndBias(node_ref, 2, 3);
    aten_layer_norm_layer_->setWeights(weight_bias.first);
    aten_layer_norm_layer_->setBiases(weight_bias.second);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(aten_layer_norm_layer_);

    return layer;
}

}  // namespace frontend
}  // namespace nn_compiler
