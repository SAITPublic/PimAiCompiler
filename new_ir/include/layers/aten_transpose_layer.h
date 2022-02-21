#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

//Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1)
class AtenTransposeLayer : public NNLayer {
 public:
    /**
     * @brief AtenTransposeLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenTransposeLayer(std::string name, std::string type)
            : NNLayer(name, type) {
    }

    explicit AtenTransposeLayer(const AtenTransposeLayer& aten_transpose_layer) :
        NNLayer(aten_transpose_layer) {
        this->_dim0 = aten_transpose_layer._dim0;
        this->_dim1 = aten_transpose_layer._dim1;
    }

    virtual ~AtenTransposeLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenTransposeLayer>(new AtenTransposeLayer(*this));
    }

    void setAttr(int64_t dim0, int64_t dim1) {
        this->_dim0 = dim0;
        this->_dim1 = dim1;
    }

    std::vector<int64_t> getAttr() {
        return std::vector<int64_t>{this->_dim0, this->_dim1};
    }

    void setDim0(int64_t dim0) { this->_dim0 = dim0; }

    int64_t getDim0() { return this->_dim0; }

    void setDim1(int64_t dim1) { this->_dim1 = dim1; }

    int64_t getDim1() { return this->_dim1; }

    void printAttr() {
        Log::IR::I() << "    AtenTransposeAttr    ";
        Log::IR::I() << "    dim0 is              "<< _dim0;
        Log::IR::I() << "    dim1 is              "<< _dim1;
    }

 private:
    int64_t _dim0 = INT64_MIN;
    int64_t _dim1 = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
