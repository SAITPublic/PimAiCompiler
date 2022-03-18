#pragma once
#include <torch/script.h>
#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace ir
{
class AtenBatchNorm2dLayer : public NNLayer
{
   public:
    AtenBatchNorm2dLayer() {}

    AtenBatchNorm2dLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenBatchNorm2dLayer(const AtenBatchNorm2dLayer& batch_norm2d_layer) : NNLayer(batch_norm2d_layer)
    {
        this->weights_ = batch_norm2d_layer.weights_;
        this->bias_ = batch_norm2d_layer.bias_;
        this->weight_ids_ = batch_norm2d_layer.weight_ids_;
        this->bias_ids_ = batch_norm2d_layer.bias_ids_;
        this->training_ = batch_norm2d_layer.training_;
        this->momentum_ = batch_norm2d_layer.momentum_;
        this->eps_ = batch_norm2d_layer.eps_;
        this->cudnn_enabled_ = batch_norm2d_layer.cudnn_enabled_;
    }

    virtual ~AtenBatchNorm2dLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenBatchNorm2dLayer>(new AtenBatchNorm2dLayer(*this));
    }

    std::vector<at::Tensor> getWeights() { return this->weights_; }

    void setWeights(const std::vector<at::Tensor>& weights) { weights_ = weights; }

    std::vector<at::Tensor> getBiases() { return this->bias_; }

    void setBiases(const std::vector<at::Tensor>& bias) { bias_ = bias; }

    std::vector<int64_t> getWeightIds() { return weight_ids_; }

    void setWeightIds(const std::vector<int64_t>& weight_ids) { weight_ids_ = weight_ids; }

    std::vector<int64_t> getBiasIds() { return bias_ids_; }

    void setBiasIds(const std::vector<int64_t>& bias_ids) { bias_ids_ = bias_ids; }

    void setTraining(int training) { training_ = training; }

    int getTraining() { return training_; }

    void setMomentum(double momentum) { momentum_ = momentum; }

    double getMomentum() { return momentum_; }

    void setEps(double eps) { eps_ = eps; }

    double getEps() { return eps_; }

    void setCudnnEnabled(int cudnn_enabled) { cudnn_enabled_ = cudnn_enabled; }

    int getCudnnEnabled() { return cudnn_enabled_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenAsTensorAttr       ";
        DLOG(INFO) << "    training is          " << training_;
        DLOG(INFO) << "    momentum is          " << momentum_;
        DLOG(INFO) << "    eps is               " << eps_;
        DLOG(INFO) << "    cudnn_enabled is     " << cudnn_enabled_;
    }

   private:
    std::vector<at::Tensor> weights_;
    std::vector<at::Tensor> bias_;
    std::vector<int64_t> weight_ids_;
    std::vector<int64_t> bias_ids_;
    int training_ = INT32_MAX;
    double momentum_ = DBL_MAX;
    double eps_ = DBL_MAX;
    int cudnn_enabled_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
