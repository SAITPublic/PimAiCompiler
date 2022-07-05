#pragma once

#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace ir
{
class EndMultiStreamLayer : public NNLayer
{
   public:
    /**
     * @brief EndMultiStreamLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */

    EndMultiStreamLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit EndMultiStreamLayer(const EndMultiStreamLayer& end_multi_stream_layer) :
        NNLayer(end_multi_stream_layer)
    {
        this->layers_num_ = end_multi_stream_layer.layers_num_;
    }

    virtual ~EndMultiStreamLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<EndMultiStreamLayer>(new EndMultiStreamLayer(*this));
    }

    void setLayerNum(int layer_num) { layers_num_ = layer_num; }

    int getLayerNum() { return layers_num_; }

    void printAttr()
    {
        DLOG(INFO) << "      EndMultiStream Attr     ";
        DLOG(INFO) << "      layers_num is            " << layers_num_;
    }

   private:
    int layers_num_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
