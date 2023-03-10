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

#pragma once

#include "frontend/importer/layer_builder/layer_builder.h"

namespace nn_compiler
{
namespace frontend
{
class ModelBuilder
{
   public:
    ModelBuilder() {}

    void build(std::unique_ptr<ir::NNModel>& nn_model, const std::string& torch_model_path);

   private:
    std::shared_ptr<torch::jit::Module> parseTorchScript(const std::string& torch_model);

    void torchToNNGraph(std::unique_ptr<ir::NNModel>& nn_model, std::shared_ptr<torch::jit::Module>& torch_model);

    template <typename T>
    void importTorchScriptMethodBlock(std::unique_ptr<ir::NNModel>& nn_model, const std::string& name,
                                      const T& method_block, bool is_main_graph = false);

    void importModuleAttributes(std::shared_ptr<torch::jit::Module> torch_model);

    std::shared_ptr<ir::NNLayer> createLayer(std::shared_ptr<frontend::LayerBuilder> builder,
                                             const torch::jit::Node* node, std::unique_ptr<ir::NNModel>& nn_model);

    uint32_t getUniqueBlockId();

    uint32_t getUniqueTensorId(std::unique_ptr<ir::NNModel>& nn_model);

    void getFinalType(c10::TypePtr t, std::set<c10::TypePtr>& type_set);

    void isInplaceNode(std::string op_node, std::shared_ptr<ir::NNLayer>& layer);

    nn_compiler::ir::DataType convertTorchScriptType(const c10::TypePtr& type);

    std::map<uint32_t, std::pair<std::vector<std::string>, std::string>> shape_tensors_;

    std::map<std::string, std::shared_ptr<ir::NNGraph>> graphs_;

    std::map<std::string, torch::jit::IValue> module_attributes_;

    std::map<std::string, std::shared_ptr<ir::NNLayer>> attr_layer_map;

    std::shared_ptr<torch::jit::Module> torch_model_;

    frontend::LayerBuilders layer_builders_;

    std::map<const torch::jit::Value*, uint32_t> value_tensor_map_;

    uint32_t block_counter_ = 0;
};

}  // namespace frontend
}  // namespace nn_compiler
