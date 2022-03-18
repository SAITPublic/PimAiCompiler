#pragma once
#include <torch/script.h>
#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace ir
{
// Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias,
//               IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
//               int64_t groups);
class AtenConv2dLayer : public NNLayer
{
   public:
    /**
     * @brief AtenConv2dLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenConv2dLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenConv2dLayer(const AtenConv2dLayer &aten_conv2d_layer) : NNLayer(aten_conv2d_layer)
    {
        this->weights_ = aten_conv2d_layer.weights_;
        this->bias_ = aten_conv2d_layer.bias_;
        this->weight_ids_ = aten_conv2d_layer.weight_ids_;
        this->bias_ids_ = aten_conv2d_layer.bias_ids_;
        this->stride_ = aten_conv2d_layer.stride_;
        this->padding_ = aten_conv2d_layer.padding_;
        this->dialation_ = aten_conv2d_layer.dialation_;
        this->groups_ = aten_conv2d_layer.groups_;
    }

    virtual ~AtenConv2dLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenConv2dLayer>(new AtenConv2dLayer(*this)); }

    std::vector<at::Tensor> getWeights() { return this->weights_; }

    void setWeights(const std::vector<at::Tensor> &weights) { weights_ = weights; }

    std::vector<at::Tensor> getBiases() { return this->bias_; }

    void setBiases(const std::vector<at::Tensor> &bias) { bias_ = bias; }

    std::vector<int64_t> getWeightIds() { return weight_ids_; }

    void setWeightIds(const std::vector<int64_t> &weight_ids) { weight_ids_ = weight_ids; }

    std::vector<int64_t> getBiasIds() { return bias_ids_; }

    void setBiasIds(const std::vector<int64_t> &bias_ids) { bias_ids_ = bias_ids; }

    void setStride(const std::vector<int64_t> &stride) { stride_ = stride; }

    const std::vector<int64_t> getStride() const { return stride_; }

    void setPadding(const std::vector<int64_t> &padding) { padding_ = padding; }

    const std::vector<int64_t> getPadding() const { return padding_; }

    void setDialation(const std::vector<int64_t> &dialation) { dialation_ = dialation; }

    const std::vector<int64_t> getDialation() const { return dialation_; }

    void setGroups(int64_t groups) { groups_ = groups; }

    int64_t getGroups() const { return groups_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenConv2dAttr          ";
        DLOG(INFO) << "    stride[0] is            " << stride_[0];
        DLOG(INFO) << "    stride[1] is            " << stride_[1];
        DLOG(INFO) << "    padding[0] is           " << padding_[0];
        DLOG(INFO) << "    padding[1] is           " << padding_[1];
        DLOG(INFO) << "    dialation[0] is         " << dialation_[0];
        DLOG(INFO) << "    dialation[1] is         " << dialation_[1];
        DLOG(INFO) << "    groups is               " << groups_;
    }

   private:
    std::vector<at::Tensor> weights_;
    std::vector<at::Tensor> bias_;
    std::vector<int64_t> weight_ids_;
    std::vector<int64_t> bias_ids_;
    std::vector<int64_t> stride_ = {1, 1};
    std::vector<int64_t> padding_ = {0, 0};
    std::vector<int64_t> dialation_ = {1, 1};

    int64_t groups_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
