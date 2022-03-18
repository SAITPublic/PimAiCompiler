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
        this->batch_first_ = aten_pack_padded_sequence_layer.batch_first_;
    }

    virtual ~AtenPackPaddedSequenceLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenPackPaddedSequenceLayer>
                    (new AtenPackPaddedSequenceLayer(*this));
    }

    void setBatchFirst(int batch_first) { batch_first_ = batch_first; }

    int getBatchFirst() const { return batch_first_; }

    void printAttr() {
        DLOG(INFO) << "    AtenPackPaddedSequenceAttr     ";
        DLOG(INFO) << "    batch_first is                 "<< batch_first_;
    }

 private:
    int batch_first_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
