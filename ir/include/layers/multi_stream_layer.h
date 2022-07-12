#pragma once

#include "c10/hip/HIPFunctions.h"
#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace ir
{
class MultiStreamLayer : public NNLayer
{
   public:
    /**
     * @brief MultiStreamLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */

    MultiStreamLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit MultiStreamLayer(const MultiStreamLayer& multi_stream_layer) : NNLayer(multi_stream_layer)
    {
        this->layers_ = multi_stream_layer.layers_;
        this->layers_num_ = multi_stream_layer.layers_num_;
    }

    virtual ~MultiStreamLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<MultiStreamLayer>(new MultiStreamLayer(*this));
    }

    void setLayers(std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers) { layers_ = layers; }

    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> getLayers() { return layers_; }

    void setLayerNum(int layer_num) { layers_num_ = layer_num; }

    int getLayerNum() { return layers_num_; }

    void setStreams()
    {
        if (layers_num_ == INT32_MAX) return ;
        for (int i = 0; i < layers_num_; i++) {
            hipStream_t stream;
            hipStreamCreate(&stream);
            streams_.push_back(stream);
        }
    }

    std::vector<hipStream_t> getStreams() { return streams_; }

    void printAttr()
    {
        DLOG(INFO) << "      MultiStream Attr     ";
        DLOG(INFO) << "      layers_num is             " << layers_num_;
    }

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers_;
    int layers_num_ = INT32_MAX;
    std::vector<hipStream_t> streams_;
};

}  // namespace ir
}  // namespace nn_compiler
