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

class AtenArange1Builder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenArange1Layer> aten_arange_layer_;
};

class AtenArange2Builder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenArange2Layer> aten_arange_layer_;
};

class AtenArange3Builder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenArange3Layer> aten_arange_layer_;
};

class AtenAsTensorBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenAsTensorLayer> aten_as_tensor_layer_;
};

class AtenBatchNorm2dBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenBatchNorm2dLayer> aten_batch_norm2d_layer_;

    void ptTensor2nncompilerDTensor(at::Tensor torch_tensor, nn_compiler::ir::DTensor& d_tensor);
    void getLearnableParameters(const torch::jit::Node* node_conv2d,
                                std::vector<nn_compiler::ir::DTensor>& weight_tensors,
                                std::vector<nn_compiler::ir::DTensor>& bias_tensors);
};

class AtenBitwiseNotBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenBitwiseNotLayer> aten_bitwise_not_layer_;
};

class AtenBmmBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenBmmLayer> aten_bmm_layer_;
};

class AtenBoolBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenBoolLayer> aten_bool_layer_;
};

class AtenCatBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenCatLayer> aten_cat_layer_;
};

class AtenCeilBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenCeilLayer> aten_ceil_layer_;
};

class AtenChunkBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenChunkLayer> aten_chunk_layer_;
};

class AtenClampBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenClampLayer> aten_clamp_layer_;
};

class AtenClearBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenClearLayer> aten_clear_layer_;
};

class AtenContiguousBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenContiguousLayer> aten_contiguous_layer_;
};

class AtenConv2dBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenConv2dLayer> aten_conv2d_layer_;

   private:
    void ptTensor2nncompilerDTensor(at::Tensor torch_tensor, nn_compiler::ir::DTensor& d_tensor);
    void getLearnableParameters(const torch::jit::Node* node_conv2d,
                                std::vector<nn_compiler::ir::DTensor>& weight_tensors,
                                std::vector<nn_compiler::ir::DTensor>& bias_tensors);
};

class AtenCopyBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenCopyLayer> aten_copy_layer_;
};

class AtenCpuBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenCpuLayer> aten_cpu_layer_;
};

class AtenCudaBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenCudaLayer> aten_cuda_layer_;
};

class AtenDeriveIndexBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenDeriveIndexLayer> aten_derive_index_layer_;
};

class AtenDimBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenDimLayer> aten_dim_layer_;
};

class AtenDivBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenDivLayer> aten_div_layer_;
};

class AtenDropoutBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenDropoutLayer> aten_dropout_layer_;
};

class AtenEmbeddingBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenEmbeddinNNLayer> aten_embedding_layer_;
};

class AtenEqBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenEqLayer> aten_eq_layer_;
};

class AtenEqualBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenEqualLayer> aten_equal_layer_;
};

class AtenExpandBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenExpandLayer> aten_expand_layer_;
};

class AtenFillBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenFillLayer> aten_fill_layer_;
};

class AtenFloorDivideBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenFloorDivideLayer> aten_floor_divide_layer_;
};

class AtenFormatBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenFormatLayer> aten_format_layer_;
};

class AtenGatherBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenGatherLayer> aten_gather_layer_;
};

class AtenGeBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenGeLayer> aten_ge_layer_;
};

class AtenGtBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenGtLayer> aten_gt_layer_;
};

class AtenIndexBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIndexLayer> aten_index_layer_;
};

class AtenGetItemBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenGetItemLayer> aten_get_item_layer_;
};

class AtenIndexPutBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIndexPutLayer> aten_index_put_layer_;
};

class AtenIndexSelectBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIndexSelectLayer> aten_index_select_layer_;
};

class AtenIntBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIntLayer> aten_int_layer_;
};

class AtenIsBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenIsLayer> aten_is_layer_;
};

class AtenItemBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenItemLayer> aten_item_layer_;
};

class AtenLeakyReluBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLeakyReluLayer> aten_leaky_relu_layer_;
};

class AtenLenBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLenLayer> aten_len_layer_;
};

class AtenLinearBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLinearLayer> aten_linear_layer_;
    void ptTensor2nncompilerDTensor(at::Tensor torch_tensor, nn_compiler::ir::DTensor& d_tensor);
    void getLearnableParameters(const torch::jit::Node* node_def, std::vector<nn_compiler::ir::DTensor>& weight_tensors,
                                std::vector<nn_compiler::ir::DTensor>& bias_tensors);
};

class AtenListBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenListLayer> aten_list_layer_;
};

class AtenLogBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLoNNLayer> aten_log_layer_;
};

class AtenLogSoftmaxBuilder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLogSoftmaxLayer> aten_log_softmax_layer_;
};

class AtenLSTM1Builder : public LayerBuilder
{
   public:
    std::shared_ptr<ir::NNLayer> buildLayer(const torch::jit::Node* node_ref);

   private:
    std::shared_ptr<ir::AtenLSTM1Layer> aten_lstm1_layer_;

   private:
    void ptTensor2nncompilerDTensor(at::Tensor torch_tensor, nn_compiler::ir::DTensor& d_tensor);
    void getLearnableParameters(const torch::jit::Node* node_lstm,
                                std::vector<nn_compiler::ir::DTensor>& weight_tensors,
                                std::vector<nn_compiler::ir::DTensor>& bias_tensors);
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
        layer_builders_["aten::arange1"] = std::make_shared<AtenArange1Builder>();
        layer_builders_["aten::arange2"] = std::make_shared<AtenArange2Builder>();
        layer_builders_["aten::arange3"] = std::make_shared<AtenArange3Builder>();
        layer_builders_["aten::as_tensor"] = std::make_shared<AtenAsTensorBuilder>();
        layer_builders_["aten::batch_norm"] = std::make_shared<AtenBatchNorm2dBuilder>();
        layer_builders_["aten::bitwise_not"] = std::make_shared<AtenBitwiseNotBuilder>();
        layer_builders_["aten::bmm"] = std::make_shared<AtenBmmBuilder>();
        layer_builders_["aten::Bool"] = std::make_shared<AtenBoolBuilder>();
        layer_builders_["aten::cat"] = std::make_shared<AtenCatBuilder>();
        layer_builders_["aten::ceil"] = std::make_shared<AtenCeilBuilder>();
        layer_builders_["aten::chunk"] = std::make_shared<AtenChunkBuilder>();
        layer_builders_["aten::clamp"] = std::make_shared<AtenClampBuilder>();
        layer_builders_["aten::clear"] = std::make_shared<AtenClearBuilder>();
        layer_builders_["aten::contiguous"] = std::make_shared<AtenContiguousBuilder>();
        layer_builders_["aten::conv2d"] = std::make_shared<AtenConv2dBuilder>();
        layer_builders_["aten::copy_"] = std::make_shared<AtenCopyBuilder>();
        layer_builders_["aten::cpu"] = std::make_shared<AtenCpuBuilder>();
        layer_builders_["aten::cuda"] = std::make_shared<AtenCudaBuilder>();
        layer_builders_["aten::dim"] = std::make_shared<AtenDimBuilder>();
        layer_builders_["aten::div"] = std::make_shared<AtenDivBuilder>();
        layer_builders_["aten::div_"] = std::make_shared<AtenDivBuilder>();
        layer_builders_["aten::dropout"] = std::make_shared<AtenDropoutBuilder>();
        layer_builders_["aten::dropout_"] = std::make_shared<AtenDropoutBuilder>();
        layer_builders_["aten::embedding"] = std::make_shared<AtenEmbeddingBuilder>();
        layer_builders_["aten::eq"] = std::make_shared<AtenEqBuilder>();
        layer_builders_["aten::equal"] = std::make_shared<AtenEqualBuilder>();
        layer_builders_["aten::expand"] = std::make_shared<AtenExpandBuilder>();
        layer_builders_["aten::fill_"] = std::make_shared<AtenFillBuilder>();
        layer_builders_["aten::floor_divide"] = std::make_shared<AtenFloorDivideBuilder>();
        layer_builders_["aten::format"] = std::make_shared<AtenFormatBuilder>();
        layer_builders_["aten::gather"] = std::make_shared<AtenGatherBuilder>();
        layer_builders_["aten::ge"] = std::make_shared<AtenGeBuilder>();
        layer_builders_["aten::__getitem__"] = std::make_shared<AtenGetItemBuilder>();
        layer_builders_["aten::gt"] = std::make_shared<AtenGtBuilder>();
        layer_builders_["aten::index"] = std::make_shared<AtenIndexBuilder>();
        layer_builders_["aten::index_put_"] = std::make_shared<AtenIndexPutBuilder>();
        layer_builders_["aten::index_select"] = std::make_shared<AtenIndexSelectBuilder>();
        layer_builders_["aten::Int"] = std::make_shared<AtenIntBuilder>();
        layer_builders_["aten::item"] = std::make_shared<AtenItemBuilder>();
        layer_builders_["aten::leaky_relu"] = std::make_shared<AtenLeakyReluBuilder>();
        layer_builders_["aten::len"] = std::make_shared<AtenLenBuilder>();
        layer_builders_["aten::linear"] = std::make_shared<AtenLinearBuilder>();
        layer_builders_["aten::list"] = std::make_shared<AtenListBuilder>();
        layer_builders_["aten::log"] = std::make_shared<AtenLogBuilder>();
        layer_builders_["aten::log_softmax"] = std::make_shared<AtenLogSoftmaxBuilder>();
        layer_builders_["aten::lstm1"] = std::make_shared<AtenLSTM1Builder>();
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
