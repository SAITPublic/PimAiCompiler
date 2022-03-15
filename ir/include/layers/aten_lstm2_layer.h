#pragma once
#include <torch/script.h>
#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"

/**
 * API_2:
 * TORCH_API std::tuple<Tensor,Tensor,Tensor> lstm(
 *  const Tensor & data,
 *  const Tensor & batch_sizes,
 *  TensorList hx,
 *  TensorList params,
 *  bool has_biases,
 *  int64_t num_layers,
 *  double dropout,
 *  bool train,
 *  bool bidirectional);
 *
 */

namespace nn_compiler {
namespace ir {
class AtenLSTM2Layer : public NNLayer {
 public:
    /**
     * @brief Construct a new Aten LSTM2 Layer object
     *
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenLSTM2Layer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenLSTM2Layer(const AtenLSTM2Layer &aten_lstm2_layer) : NNLayer(aten_lstm2_layer) {
        this->setAttr(aten_lstm2_layer.has_biases_, aten_lstm2_layer.num_layers_,
                      aten_lstm2_layer.dropout_, aten_lstm2_layer.train_,
                      aten_lstm2_layer.bidirectional_);
    }

    virtual ~AtenLSTM2Layer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenLSTM2Layer>(new AtenLSTM2Layer(*this));
    }

    void printAttr() {
        Log::IR::I() << "     AtenLSTM2Attr   ";
        Log::IR::I() << "     has_biases is    " << this->has_biases_;
        Log::IR::I() << "     num_layers is    " << this->num_layers_;
        Log::IR::I() << "     dropout is       " << this->dropout_;
        Log::IR::I() << "     bidirectional is " << this->bidirectional_;
        Log::IR::I() << "     train is         " << this->train_;
    }

    void setAttr(int has_biases, int64_t num_layers, double dropout, int train,
                 int bidirectional) {
        this->has_biases_    = has_biases;
        this->num_layers_    = num_layers;
        this->dropout_       = dropout;
        this->train_         = train;
        this->bidirectional_ = bidirectional;
    }

    std::vector<at::Tensor> getWeights() { return this->weights_; }

    void setWeights(const std::vector<at::Tensor> &weights) {
        this->weights_ = weights;
    }

    std::vector<at::Tensor> getBiases() { return this->biases_; }

    void setBiases(const std::vector<at::Tensor> &biases) {
        this->biases_ = biases;
    }

    std::vector<int64_t> getWeightIds() { return weight_ids_; }

    void setWeightIds(const std::vector<int64_t>& weight_ids) { weight_ids_ = weight_ids; }

    std::vector<int64_t> getBiasIds() { return bias_ids_; }

    void setBiasIds(const std::vector<int64_t>& bias_ids) { bias_ids_ = bias_ids; }

    int getHasBiases() { return this->has_biases_; }

    void setHasBiases(int has_biases) {
        this->has_biases_ = has_biases;
    }

    int64_t getNumLayers() { return this->num_layers_; }

    void setNumLayers(int64_t num_layers) {
        this->num_layers_ = num_layers;
    }

    double getDropout() { return dropout_; }

    void setDropout(double dropout) {
        this->dropout_ = dropout;
    }

    int getTrain() { return this->train_; }

    void setTrain(int train) {
        this->train_ = train;
    }

    int getBidirectional() { return this->bidirectional_; }

    void setBidirectional(int bidirectional) {
        this->bidirectional_ = bidirectional;
    }

    struct AtenLSTM2LayerAttr {
        int has_biases;
        int64_t num_layers;
        double dropout;
        int train;
        int bidirectional;
    };

    AtenLSTM2LayerAttr getAttr() {
        AtenLSTM2LayerAttr attrs;
        attrs.has_biases    = this->has_biases_;
        attrs.num_layers    = this->num_layers_;
        attrs.dropout       = this->dropout_;
        attrs.train         = this->train_;
        attrs.bidirectional = this->bidirectional_;
        return attrs;
    }

   private:
    int     has_biases_    = INT32_MAX;
    int64_t num_layers_    = INT64_MIN;
    double  dropout_       = DBL_MAX;
    int     train_         = INT32_MAX;
    int     bidirectional_ = INT32_MAX;
    // weights & bias, 8 or 12 tensors
    std::vector<at::Tensor> weights_;     // only weight, dim > 1
    std::vector<at::Tensor> biases_;        // bias, dim == 1
    std::vector<int64_t> weight_ids_;
    std::vector<int64_t> bias_ids_;
};
}  // namespace ir
}  // namespace nn_compiler
