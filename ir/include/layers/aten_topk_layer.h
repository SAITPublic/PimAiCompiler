#pragma once

#include "ir/include/layers/nn_layer.h"

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
        this->k_       = aten_topk_layer.k_;
        this->dim_     = aten_topk_layer.dim_;
        this->largest_ = aten_topk_layer.largest_;
        this->sorted_  = aten_topk_layer.sorted_;
    }

    virtual ~AtenTopkLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenTopkLayer>(new AtenTopkLayer(*this));
    }

    void setK(int k) { k_ = k; }

    int getK() { return k_; }

    void setDim(int dim) { dim_ = dim; }

    int getDim() { return dim_; }

    void setLargest(int largest) { largest_ = largest; }

    int getLargest() { return largest_; }

    void setSorted(int sorted) { sorted_ = sorted; }

    int getSorted() { return sorted_; }

    void printAttr() {
        DLOG(INFO) << "    AtenTopkAttr    ";
        DLOG(INFO) << "    k is            "<< k_;
        DLOG(INFO) << "    dim is          "<< dim_;
        DLOG(INFO) << "    largest is      "<< largest_;
        DLOG(INFO) << "    sorted is       "<< sorted_;
    }

 private:
    int k_       = INT32_MAX;
    int dim_     = INT32_MAX;
    int largest_ = INT32_MAX;
    int sorted_  = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
