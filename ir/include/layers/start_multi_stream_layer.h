#pragma once

#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace ir
{
class StartMultiStreamLayer : public NNLayer
{
   public:
    /**
     * @brief StartMultiStreamLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */

    StartMultiStreamLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit StartMultiStreamLayer(const StartMultiStreamLayer& start_multi_stream_layer) :
        NNLayer(start_multi_stream_layer)
    {
        this->layers_ = start_multi_stream_layer.layers_;
        this->layers_num_ = start_multi_stream_layer.layers_num_;
    }

    virtual ~StartMultiStreamLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<StartMultiStreamLayer>(new StartMultiStreamLayer(*this));
    }

    void setLayers(std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers) { layers_ = layers; }

    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> getLayers() { return layers_; }

    void setLayerNum(int layer_num) { layers_num_ = layer_num; }

    int getLayerNum() { return layers_num_; }

    void printAttr()
    {
        DLOG(INFO) << "      StartMultiStream Attr     ";
        DLOG(INFO) << "      layers_num is             " << layers_num_;
    }

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers_;
    int layers_num_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
