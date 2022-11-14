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

#include "frontend/importer/layer_builder/layer_builder.h"

namespace nn_compiler
{
namespace frontend
{
std::shared_ptr<ir::NNLayer> PrimCallMethodBuilder::buildLayer(const torch::jit::Node* node_ref)
{
    DLOG(INFO) << "NOT uesed PrimCallMethodBuilder::buildLayer";
    return nullptr;
}
std::shared_ptr<ir::NNLayer> PrimCallMethodBuilder::buildLayerCustom(const std::string target_network_name)
{
    DLOG(INFO) << "build prim::callmethod";
    nn_compiler::ir::LayerType type = nn_compiler::ir::LayerType::PRIMCALLMETHOD;
    std::string name = "";

    prim_callmethod_layer_ = std::make_shared<ir::PrimCallMethodLayer>(name, type);
    prim_callmethod_layer_->setAttr(target_network_name);

    const auto& layer = std::dynamic_pointer_cast<ir::NNLayer>(prim_callmethod_layer_);
    return layer;
}

}  // namespace frontend
}  // namespace nn_compiler
