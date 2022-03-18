#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
// Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1)
class AtenTransposeLayer : public NNLayer
{
   public:
    /**
     * @brief AtenTransposeLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenTransposeLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenTransposeLayer(const AtenTransposeLayer& aten_transpose_layer) : NNLayer(aten_transpose_layer)
    {
        this->dim0_ = aten_transpose_layer.dim0_;
        this->dim1_ = aten_transpose_layer.dim1_;
    }

    virtual ~AtenTransposeLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenTransposeLayer>(new AtenTransposeLayer(*this));
    }

    void setAttr(int64_t dim0, int64_t dim1)
    {
        this->dim0_ = dim0;
        this->dim1_ = dim1;
    }

    std::vector<int64_t> getAttr() { return std::vector<int64_t>{this->dim0_, this->dim1_}; }

    void setDim0(int64_t dim0) { this->dim0_ = dim0; }

    int64_t getDim0() { return this->dim0_; }

    void setDim1(int64_t dim1) { this->dim1_ = dim1; }

    int64_t getDim1() { return this->dim1_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenTransposeAttr    ";
        DLOG(INFO) << "    dim0 is              " << dim0_;
        DLOG(INFO) << "    dim1 is              " << dim1_;
    }

   private:
    int64_t dim0_ = INT64_MIN;
    int64_t dim1_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
