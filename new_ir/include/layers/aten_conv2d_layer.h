#pragma once
#include <torch/script.h>
#include "new_ir/include/tensors/data_tensor.h"
#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

// Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias,
//               IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
//               int64_t groups);
class AtenConv2dLayer : public NNLayer {
 public:
    /**
     * @brief AtenConv2dLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenConv2dLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenConv2dLayer(const AtenConv2dLayer& aten_conv2d_layer) :
        NNLayer(aten_conv2d_layer) {
        this->_weights   = aten_conv2d_layer._weights;
        this->_bias      = aten_conv2d_layer._bias;
        this->_stride    = aten_conv2d_layer._stride;
        this->_padding   = aten_conv2d_layer._padding;
        this->_dialation = aten_conv2d_layer._dialation;
        this->_groups    = aten_conv2d_layer._groups;
    }

    virtual ~AtenConv2dLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenConv2dLayer>(new AtenConv2dLayer(*this));
    }

    std::vector<at::Tensor> getWeights() { return this->_weights; }

    void setWeights(const std::vector<at::Tensor> &weights) {
        this->_weights = weights;
    }

    std::vector<at::Tensor> getBiases() { return this->_bias; }

    void setBiases(const std::vector<at::Tensor> &bias) {
        this->_bias = bias;
    }

    void setStride(const std::vector<int64_t> &stride) { _stride = stride; }

    const std::vector<int64_t> getStride() const { return _stride; }

    void setPadding(const std::vector<int64_t> &padding) { _padding = padding; }

    const std::vector<int64_t> getPadding() const { return _padding; }

    void setDialation(const std::vector<int64_t> &dialation) { _dialation = dialation; }

    const std::vector<int64_t> getDialation() const { return _dialation; }

    void setGroups(int64_t groups) { _groups = groups; }

    int64_t getGroups() const { return _groups; }

    void printAttr() {
        Log::IR::I() << "    AtenConv2dAttr          ";
        Log::IR::I() << "    stride[0] is            "<< _stride[0];
        Log::IR::I() << "    stride[1] is            "<< _stride[1];
        Log::IR::I() << "    padding[0] is           "<< _padding[0];
        Log::IR::I() << "    padding[1] is           "<< _padding[1];
        Log::IR::I() << "    dialation[0] is         "<< _dialation[0];
        Log::IR::I() << "    dialation[1] is         "<< _dialation[1];
        Log::IR::I() << "    groups is               "<< _groups;
    }

 private:
    std::vector<at::Tensor> _weights;
    std::vector<at::Tensor> _bias;
    std::vector<int64_t> _stride = {1, 1};
    std::vector<int64_t> _padding = {0, 0};
    std::vector<int64_t> _dialation = {1, 1};

    int64_t _groups = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
