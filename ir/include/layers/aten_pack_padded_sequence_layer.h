#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

//Tensor = aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first)
class AtenPackPaddedSequenceLayer : public NNLayer {
 public:
    /**
     * @brief AtenPackPaddedSequenceLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenPackPaddedSequenceLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenPackPaddedSequenceLayer(
        const AtenPackPaddedSequenceLayer& aten_pack_padded_sequence_layer) :
            NNLayer(aten_pack_padded_sequence_layer) {
        this->_batch_first = aten_pack_padded_sequence_layer._batch_first;
    }

    virtual ~AtenPackPaddedSequenceLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenPackPaddedSequenceLayer>
                    (new AtenPackPaddedSequenceLayer(*this));
    }

    void setBatchFirst(int batch_first) { _batch_first = batch_first; }

    int getBatchFirst() const { return _batch_first; }

    void printAttr() {
        DLOG(INFO) << "    AtenPackPaddedSequenceAttr     ";
        DLOG(INFO) << "    batch_first is                 "<< _batch_first;
    }

 private:
    int _batch_first = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
