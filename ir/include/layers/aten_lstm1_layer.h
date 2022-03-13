#pragma once
#include <torch/script.h>
#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"

/**
 * API_1:
 * TORCH_API std::tuple<Tensor,Tensor,Tensor> lstm(
 *  const Tensor & input,
 *  TensorList hx,
 *  TensorList params,
 *  bool has_biases,
 *  int64_t num_layers,
 *  double dropout,
 *  bool train,
 *  bool bidirectional,
 *  bool batch_first)
 *
 */

namespace nn_compiler
{
namespace ir
{
class AtenLSTM1Layer : public NNLayer
{
   public:
    /**
     * @brief Construct a new Aten LSTM1 Layer object
     *
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenLSTM1Layer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenLSTM1Layer(const AtenLSTM1Layer &aten_lstm1_layer) : NNLayer(aten_lstm1_layer)
    {
        this->setAttr(aten_lstm1_layer._has_biases, aten_lstm1_layer._num_layers, aten_lstm1_layer._dropout,
                      aten_lstm1_layer._train, aten_lstm1_layer._bidirectional, aten_lstm1_layer._batch_first);
    }

    virtual ~AtenLSTM1Layer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenLSTM1Layer>(new AtenLSTM1Layer(*this)); }

    void setMatchCustomOpt(bool match_custom_opt) { match_custom_opt_ = match_custom_opt; }
    void setCustomOptNumber(int custom_opt_number) { custom_opt_number_ = custom_opt_number; }

    bool getMatchCustomOpt() { return match_custom_opt_; }
    int getCustomOptNumber() { return custom_opt_number_; }

    void printAttr()
    {
        Log::IR::I() << "     AtenLSTM1Attr   ";
        Log::IR::I() << "     has_biases is    " << this->_has_biases;
        Log::IR::I() << "     num_layers is    " << this->_num_layers;
        Log::IR::I() << "     dropout is       " << this->_dropout;
        Log::IR::I() << "     bidirectional is " << this->_bidirectional;
        Log::IR::I() << "     train is         " << this->_train;
        Log::IR::I() << "     batch_first is   " << this->_batch_first;
    }

   private:
    int _has_biases = INT32_MAX;
    int64_t _num_layers = INT64_MIN;
    double _dropout = DBL_MAX;
    int _train = INT32_MAX;
    int _bidirectional = INT32_MAX;
    int _batch_first = INT32_MAX;
    // weights & bias, 8 or 12 tensors
    std::vector<at::Tensor> _weights;     // only weight, dim > 1
    std::vector<at::Tensor> _biases;        // bias, dim == 1

    // LSTM type, helper info for building ir, it will not be saved to ir file
    int _lstm_type = 0;
    bool match_custom_opt_;
    int custom_opt_number_;

   public:
    void setAttr(int has_biases, int64_t num_layers, double dropout, int train, int bidirectional, int batch_first)
    {
        this->_has_biases = has_biases;
        this->_num_layers = num_layers;
        this->_dropout = dropout;
        this->_train = train;
        this->_bidirectional = bidirectional;
        this->_batch_first = batch_first;
    }

    std::vector<at::Tensor> getWeights() { return this->_weights; }

    void setWeights(const std::vector<at::Tensor> &weights) {
        this->_weights = weights;
    }

    std::vector<at::Tensor> getBiases() { return this->_biases; }

    void setBiases(const std::vector<at::Tensor> &biases) {
        this->_biases = biases;
    }

    int getHasBiases() { return this->_has_biases; }

    void setHasBiases(int has_biases) { this->_has_biases = has_biases; }

    int64_t getNumLayers() { return this->_num_layers; }

    void setNumLayers(int64_t num_layers) { this->_num_layers = num_layers; }

    double getDropout() { return _dropout; }

    void setDropout(double dropout) { this->_dropout = dropout; }

    int getTrain() { return this->_train; }

    void setTrain(int train) { this->_train = train; }

    int getBidirectional() { return this->_bidirectional; }

    void setBidirectional(int bidirectional) { this->_bidirectional = bidirectional; }

    int getBatchFirst() { return this->_batch_first; }

    void setBatchFirst(int batch_first) { this->_batch_first = batch_first; }

    struct AtenLSTM1LayerAttr {
        int has_biases;
        int64_t num_layers;
        double dropout;
        int train;
        int bidirectional;
        int batch_first;
    };

    AtenLSTM1LayerAttr getAttr()
    {
        AtenLSTM1LayerAttr attrs;
        attrs.has_biases = this->_has_biases;
        attrs.num_layers = this->_num_layers;
        attrs.dropout = this->_dropout;
        attrs.train = this->_train;
        attrs.bidirectional = this->_bidirectional;
        attrs.batch_first = this->_batch_first;
        return attrs;
    }
};
}  // namespace ir
}  // namespace nn_compiler
