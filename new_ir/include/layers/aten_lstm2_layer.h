#pragma once

#include "new_ir/include/layers/nn_layer.h"
#include "new_ir/include/tensors/data_tensor.h"

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
    AtenLSTM2Layer(std::string name, std::string type) : NNLayer(name, type) {}

    explicit AtenLSTM2Layer(const AtenLSTM2Layer &aten_lstm2_layer) : NNLayer(aten_lstm2_layer) {
        this->setAttr(aten_lstm2_layer._has_biases, aten_lstm2_layer._num_layers,
                      aten_lstm2_layer._dropout, aten_lstm2_layer._train,
                      aten_lstm2_layer._bidirectional);
    }

    virtual ~AtenLSTM2Layer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenLSTM2Layer>(new AtenLSTM2Layer(*this));
    }

    void printAttr() {
        Log::IR::I() << "     AtenLSTM2Attr   ";
        Log::IR::I() << "     has_biases is    " << this->_has_biases;
        Log::IR::I() << "     num_layers is    " << this->_num_layers;
        Log::IR::I() << "     dropout is       " << this->_dropout;
        Log::IR::I() << "     bidirectional is " << this->_bidirectional;
        Log::IR::I() << "     train is         " << this->_train;
    }

 private:
    int     _has_biases    = INT32_MAX;
    int64_t _num_layers    = INT64_MIN;
    double  _dropout       = DBL_MAX;
    int     _train         = INT32_MAX;
    int     _bidirectional = INT32_MAX;
    // weights & bias, 8 or 12 tensors
    std::vector<DTensor> _weights;     // only weight, dim > 1
    std::vector<DTensor> _biases;        // bias, dim == 1

 public:
    void setAttr(int has_biases, int64_t num_layers, double dropout, int train,
                 int bidirectional) {
        this->_has_biases    = has_biases;
        this->_num_layers    = num_layers;
        this->_dropout       = dropout;
        this->_train         = train;
        this->_bidirectional = bidirectional;
    }

    std::vector<DTensor> getWeights() { return this->_weights; }

    void setWeights(const std::vector<DTensor> &weights) {
        this->_weights = weights;
    }

    std::vector<DTensor> getBiases() { return this->_biases; }

    void setBiases(const std::vector<DTensor> &biases) {
        this->_biases = biases;
    }

    int getHasBiases() { return this->_has_biases; }

    void setHasBiases(int has_biases) {
        this->_has_biases = has_biases;
    }

    int64_t getNumLayers() { return this->_num_layers; }

    void setNumLayers(int64_t num_layers) {
        this->_num_layers = num_layers;
    }

    double getDropout() { return _dropout; }

    void setDropout(double dropout) {
        this->_dropout = dropout;
    }

    int getTrain() { return this->_train; }

    void setTrain(int train) {
        this->_train = train;
    }

    int getBidirectional() { return this->_bidirectional; }

    void setBidirectional(int bidirectional) {
        this->_bidirectional = bidirectional;
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
        attrs.has_biases    = this->_has_biases;
        attrs.num_layers    = this->_num_layers;
        attrs.dropout       = this->_dropout;
        attrs.train         = this->_train;
        attrs.bidirectional = this->_bidirectional;
        return attrs;
    }
};
}  // namespace ir
}  // namespace nn_compiler
