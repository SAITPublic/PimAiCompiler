#pragma once

#include "importer/layer_builder/layer_builder.h"
#include "ir/include/common/log.hpp"
#include "ir/include/layers/nn_layer.h"
#include "ir/include/nn_model.h"
#include "ir/include/nn_network.h"

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

    void torchToNNNnetwork(std::unique_ptr<ir::NNModel>& nn_model, std::shared_ptr<torch::jit::Module>& torch_model);

    template <typename T>
    void importTorchScriptMethodBlock(std::unique_ptr<ir::NNModel>& nn_model, const std::string& name,
                                      const T& method_block, bool is_main_graph = false);

    void importModuleAttributes(std::shared_ptr<torch::jit::Module> torch_model);

    uint32_t getUniqueBlockId();

    uint32_t getUniqueTensorId(std::unique_ptr<ir::NNModel>& nn_model);

    void getFinalType(c10::TypePtr t, std::set<c10::TypePtr>& type_set);

    void isInplaceNode(std::string op_node, std::shared_ptr<ir::NNLayer>& layer);

    nn_compiler::ir::DataType convertTorchScriptType(const c10::TypePtr& type);

    std::map<uint32_t, std::pair<std::vector<std::string>, std::string>> shape_tensors_;

    std::map<std::string, std::shared_ptr<ir::NNNetwork>> networks_;

    std::map<std::string, torch::jit::IValue> module_attributes_;

    std::map<std::string, std::shared_ptr<ir::NNLayer>> attr_layer_map;

    std::shared_ptr<torch::jit::Module> torch_model_;

    frontend::LayerBuilders layer_builders_;

    std::map<const torch::jit::Value*, uint32_t> value_tensor_map_;

    uint32_t block_counter_ = 0;
};

}  // namespace frontend
}  // namespace nn_compiler
