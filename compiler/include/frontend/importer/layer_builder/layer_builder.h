#pragma once

#include <torch/script.h>
#include <vector>

#include "all_layers.h"
#include "import/utils/attr_parser.h"
#include "new_ir/include/layers/nn_layer.h"
#include "new_ir/include/nn_network.h"
#include "new_ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace frontend
{
class LayerBuilder
{
   public:
    LayerBuilder() { attr_parser_ = std::make_shared<AttrParser>(); }

    virtual std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref) = 0;

    virtual std::shared_ptr<ir::DTensor> getKernel(const torch::jit::Node* node_ref,
                                                   const torch::jit::Module* torchscript_model)
    {
        return nullptr;
    }

    virtual std::shared_ptr<ir::DTensor> getBias(const torch::jit::Node* node_ref,
                                                 const torch::jit::Module* torchscript_model)
    {
        return nullptr;
    }

    std::shared_ptr<AttrParser> parser() { return attr_parser_; }

    virtual ~LayerBuilder() = default;

   private:
    std::shared_ptr<AttrParser> attr_parser_;
};

class AtenAddBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAddLayer> aten_add_layer_;
};

class AtenAddmmBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAddmmLayer> aten_addmm_layer_;
};

class AtenAndBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAndLayer> aten_and_layer_;
};

class AtenAnyBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAnyLayer> aten_any_layer_;
};

class AtenAppendBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAppendLayer> aten_append_layer_;
};

class LayerBuilders
{
   public:
    LayerBuilders()
    {
        // Register layer builder.
        layer_builders_["aten::add"] = std::make_shared<AtenAddBuilder>();
        layer_builders_["aten::add_"] = std::make_shared<AtenAddBuilder>();
        layer_builders_["aten::addmm"] = std::make_shared<AtenAddmmBuilder>();
        layer_builders_["aten::__and__"] = std::make_shared<AtenAndBuilder>();
        layer_builders_["aten::any"] = std::make_shared<AtenAnyBuilder>();
        layer_builders_["aten::append"] = std::make_shared<AtenAppendBuilder>();
    }

    std::shared_ptr<LayerBuilder> get(std::string layer_type)
    {
        if (layer_builders_[layer_type] == nullptr) {
            LOG(INFO) << "layer type " << layer_type << " is unsupport.";
        }
        return layer_builders_[layer_type];
    }

    virtual std::pair<std::string, std::shared_ptr<ir::DTensor>> getKernel(torch::jit::Module& torchscript_model)
    {
        return std::make_pair("", nullptr);
    }

    virtual std::pair<std::string, std::shared_ptr<ir::DTensor>> getBias(torch::jit::Module& torchscript_model)
    {
        return std::make_pair("", nullptr);
    }

    ~LayerBuilders() { this->layer_builders_.clear(); }

   private:
    std::map<std::string, std::shared_ptr<LayerBuilder>> layer_builders_;
};

}  // namespace frontend
}  // namespace nn_compiler
