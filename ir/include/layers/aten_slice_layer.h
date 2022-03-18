#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

// TORCH_API Tensor slice:
// (const Tensor & self, int64_t dim=0, c10::optional<int64_t> start=0,
//  c10::optional<int64_t> end=9223372036854775807, int64_t step=1);

class AtenSliceLayer : public NNLayer {
 public:
    /**
     * @brief AtenSliceLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenSliceLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenSliceLayer(const AtenSliceLayer& aten_slice_layer) :
        NNLayer(aten_slice_layer) {
        this->setAttr(aten_slice_layer._dim, aten_slice_layer._start, 
                      aten_slice_layer._end, aten_slice_layer._step);
    }

    virtual ~AtenSliceLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenSliceLayer>(new AtenSliceLayer(*this));
    }

     void printAttr() {
        DLOG(INFO) << "      AtenSliceAttr      ";
    }

 private:
    // Attributes
    int64_t _dim   = INT64_MIN;
    int64_t _start = INT64_MIN;
    int64_t _end   = INT64_MIN;
    int64_t _step  = INT64_MIN;

 public:
    void setAttr(int64_t dim, int64_t start, int64_t end, int64_t step) {
        this->_dim   = dim;
        this->_start = start;
        this->_end   = end;
        this->_step  = step;
    }

    std::vector<int64_t> getAttr() {
        return std::vector<int64_t>{this->_dim, this->_start, this->_end, this->_step};
    }

    void setDim(int64_t dim) {
        this->_dim = dim;
    }

    int64_t getDim() { return this->_dim; }

    void setStart(int64_t start) {
        this->_start = start;
    }

    int64_t getStart() { return this->_start; }

    void setEnd(int64_t end) {
        this->_end = end;
    }

    int64_t getEnd() { return this->_end; }

    void setStep(int64_t step) {
        this->_step = step;
    }

    int64_t getStep() { return this->_step; }
};

}  // namespace ir
}  // namespace nn_compiler
