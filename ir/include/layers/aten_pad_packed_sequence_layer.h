#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

//Tensor = aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first,
//                                    Scalar padding_value, int total_length)
class AtenPadPackedSequenceLayer : public NNLayer {
 public:
    /**
     * @brief AtenPadPackedSequenceLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenPadPackedSequenceLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenPadPackedSequenceLayer(
        const AtenPadPackedSequenceLayer& aten_pad_packed_sequence_layer) :
            NNLayer(aten_pad_packed_sequence_layer) {
        this->_batch_first   = aten_pad_packed_sequence_layer._batch_first;
        this->_padding_value = aten_pad_packed_sequence_layer._padding_value;
        this->_total_length  = aten_pad_packed_sequence_layer._total_length;
    }

    virtual ~AtenPadPackedSequenceLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenPadPackedSequenceLayer>
                    (new AtenPadPackedSequenceLayer(*this));
    }

    void setBatchFirst(int batch_first) { _batch_first = batch_first; }

    int getBatchFirst() const { return _batch_first; }

    void setPaddingValue(float padding_value) { _padding_value = padding_value; }

    float getPaddingValue() const { return _padding_value; }

    void setTotalLength(int64_t total_length) { _total_length = total_length; }

    int64_t getTotalLength() const { return _total_length; }

    void printAttr() {
        DLOG(INFO) << "    AtenPadPackedSequenceAttr     ";
        DLOG(INFO) << "    batch_first is                 " << _batch_first;
        DLOG(INFO) << "    padding_value is               " << _padding_value;
        DLOG(INFO) << "    total_length is                " << _total_length;
    }

 private:
    int  _batch_first     = INT32_MAX;
    float _padding_value  = FLT_MAX;
    int64_t _total_length = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
