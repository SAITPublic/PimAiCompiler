#pragma once

#include "new_ir/include/layers/nn_layer.h"
#include "new_ir/include/tensors/data_tensor.h"

namespace nn_compiler {
namespace ir {


class AtenBatchNorm2dLayer : public NNLayer {
 public:
    AtenBatchNorm2dLayer() {}

    AtenBatchNorm2dLayer(std::string name, LayerType type)
                     : NNLayer(name, type) {
    }

    explicit AtenBatchNorm2dLayer(const AtenBatchNorm2dLayer& batch_norm2d_layer) :
        NNLayer(batch_norm2d_layer) {
        this->weights_ = batch_norm2d_layer.weights_;
        this->bias_ = batch_norm2d_layer.bias_;
        this->training_  = batch_norm2d_layer.training_;
        this->momentum_ = batch_norm2d_layer.momentum_;
        this->eps_ = batch_norm2d_layer.eps_;
        this->cudnn_enabled_ = batch_norm2d_layer.cudnn_enabled_;
    }

    virtual ~AtenBatchNorm2dLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenBatchNorm2dLayer>(new AtenBatchNorm2dLayer(*this));
    }

    std::vector<DTensor> getWeights() { return this->weights_; }

    void setWeights(const std::vector<DTensor> &weights) {
        this->weights_ = weights;
    }

    std::vector<DTensor> getBiases() { return this->bias_; }

    void setBiases(const std::vector<DTensor> &bias) {
        this->bias_ = bias;
    }

    void setTraining(int training) { training_ = training; }

    int getTraining() { return training_; }

    void setMomentum(double momentum) { momentum_ = momentum; }

    double getMomentum() { return momentum_; }

    void setEps(double eps) { eps_ = eps; }

    double getEps() { return eps_; }

    void setCudnnEnabled(int cudnn_enabled) { cudnn_enabled_ = cudnn_enabled; }

    int getCudnnEnabled() { return cudnn_enabled_; }

    void printAttr() {
        Log::IR::I() << "    AtenAsTensorAttr       ";
        Log::IR::I() << "    training is          " << training_;
        Log::IR::I() << "    momentum is          " << momentum_;
        Log::IR::I() << "    eps is               " << eps_;
        Log::IR::I() << "    cudnn_enabled is     " << cudnn_enabled_;
    }

 private:
    std::vector<DTensor> weights_;
    std::vector<DTensor> bias_;
    int training_  = INT32_MAX;
    double momentum_ = DBL_MAX;
    double eps_ = DBL_MAX;
    int cudnn_enabled_ = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
