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
        this->weights_ = aten_lstm1_layer.weights_;
        this->biases_ = aten_lstm1_layer.biases_;
        this->weight_ids_ = aten_lstm1_layer.weight_ids_;
        this->bias_ids_ = aten_lstm1_layer.bias_ids_;
        this->setAttr(aten_lstm1_layer.has_biases_, aten_lstm1_layer.num_layers_, aten_lstm1_layer.dropout_,
                      aten_lstm1_layer.train_, aten_lstm1_layer.bidirectional_, aten_lstm1_layer.batch_first_);
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
        Log::IR::I() << "     has_biases is    " << this->has_biases_;
        Log::IR::I() << "     num_layers is    " << this->num_layers_;
        Log::IR::I() << "     dropout is       " << this->dropout_;
        Log::IR::I() << "     bidirectional is " << this->bidirectional_;
        Log::IR::I() << "     train is         " << this->train_;
        Log::IR::I() << "     batch_first is   " << this->batch_first_;
    }

    void setAttr(int has_biases, int64_t num_layers, double dropout, int train, int bidirectional, int batch_first)
    {
        this->has_biases_ = has_biases;
        this->num_layers_ = num_layers;
        this->dropout_ = dropout;
        this->train_ = train;
        this->bidirectional_ = bidirectional;
        this->batch_first_ = batch_first;
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

    void setHasBiases(int has_biases) { this->has_biases_ = has_biases; }

    int64_t getNumLayers() { return this->num_layers_; }

    void setNumLayers(int64_t num_layers) { this->num_layers_ = num_layers; }

    double getDropout() { return dropout_; }

    void setDropout(double dropout) { this->dropout_ = dropout; }

    int getTrain() { return this->train_; }

    void setTrain(int train) { this->train_ = train; }

    int getBidirectional() { return this->bidirectional_; }

    void setBidirectional(int bidirectional) { this->bidirectional_ = bidirectional; }

    int getBatchFirst() { return this->batch_first_; }

    void setBatchFirst(int batch_first) { this->batch_first_ = batch_first; }

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
        attrs.has_biases = this->has_biases_;
        attrs.num_layers = this->num_layers_;
        attrs.dropout = this->dropout_;
        attrs.train = this->train_;
        attrs.bidirectional = this->bidirectional_;
        attrs.batch_first = this->batch_first_;
        return attrs;
    }

   private:
    int has_biases_ = INT32_MAX;
    int64_t num_layers_ = INT64_MIN;
    double dropout_ = DBL_MAX;
    int train_ = INT32_MAX;
    int bidirectional_ = INT32_MAX;
    int batch_first_ = INT32_MAX;

    // weights & bias, 8 or 12 tensors
    std::vector<at::Tensor> weights_;     // only weight, dim > 1
    std::vector<at::Tensor> biases_;        // bias, dim == 1
    std::vector<int64_t> weight_ids_;
    std::vector<int64_t> bias_ids_;

    int lstm_type_ = 0;
    bool match_custom_opt_;
    int custom_opt_number_;
};
}  // namespace ir
}  // namespace nn_compiler
