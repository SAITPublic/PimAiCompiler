#pragma once
#include <torch/script.h>
#include "ir/include/tensors/data_tensor.h"
#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenLinearLayer : public NNLayer {
 public:
    /**
     * @brief AtenLinearLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenLinearLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenLinearLayer(const AtenLinearLayer& aten_linear_layer) :
        NNLayer(aten_linear_layer) {
        this->weights_ = aten_linear_layer.weights_;
        this->biases_  = aten_linear_layer.biases_;
    }

    virtual ~AtenLinearLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenLinearLayer>(new AtenLinearLayer(*this));
    }

    std::vector<at::Tensor> getWeights() { return this->weights_; }

    std::vector<at::Tensor> getBiases() { return this->biases_; }

    void setWeights(const std::vector<at::Tensor> &weights) { weights_ = weights; }

    void setBiases(const std::vector<at::Tensor> &biases) { biases_ = biases; }

    std::vector<int64_t> getWeightIds() { return weight_ids_; }

    void setWeightIds(const std::vector<int64_t>& weight_ids) { weight_ids_ = weight_ids; }

    std::vector<int64_t> getBiasIds() { return bias_ids_; }

    void setBiasIds(const std::vector<int64_t>& bias_ids) { bias_ids_ = bias_ids; }

    void printAttr() {
        Log::IR::I() << "    AtenLinearAttr      ";
    }

 private:
    std::vector<at::Tensor> weights_;
    std::vector<at::Tensor> biases_;
    std::vector<int64_t> weight_ids_;
    std::vector<int64_t> bias_ids_;
};

}  // namespace ir
}  // namespace nn_compiler
