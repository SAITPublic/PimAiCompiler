#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

// topk(self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
class AtenTopkLayer : public NNLayer {
 public:
    /**
     * @brief AtenTopkLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenTopkLayer(std::string name, LayerType type) : NNLayer(name, type) {
    }

    explicit AtenTopkLayer(const AtenTopkLayer& aten_topk_layer) :
        NNLayer(aten_topk_layer) {
        this->_k       = aten_topk_layer._k;
        this->_dim     = aten_topk_layer._dim;
        this->_largest = aten_topk_layer._largest;
        this->_sorted  = aten_topk_layer._sorted;
    }

    virtual ~AtenTopkLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenTopkLayer>(new AtenTopkLayer(*this));
    }

    void setK(int k) { _k = k; }

    int getK() { return _k; }

    void setDim(int dim) { _dim = dim; }

    int getDim() { return _dim; }

    void setLargest(int largest) { _largest = largest; }

    int getLargest() { return _largest; }

    void setSorted(int sorted) { _sorted = sorted; }

    int getSorted() { return _sorted; }

    void printAttr() {
        Log::IR::I() << "    AtenTopkAttr    ";
        Log::IR::I() << "    k is            "<< _k;
        Log::IR::I() << "    dim is          "<< _dim;
        Log::IR::I() << "    largest is      "<< _largest;
        Log::IR::I() << "    sorted is       "<< _sorted;
    }

 private:
    int _k       = INT32_MAX;
    int _dim     = INT32_MAX;
    int _largest = INT32_MAX;
    int _sorted  = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
