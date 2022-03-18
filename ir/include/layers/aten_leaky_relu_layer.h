#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenLeakyReluLayer : public NNLayer
{
   public:
    AtenLeakyReluLayer() {}

    AtenLeakyReluLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenLeakyReluLayer(const AtenLeakyReluLayer &aten_leaky_relu_layer) : NNLayer(aten_leaky_relu_layer)
    {
        this->scalar_ = aten_leaky_relu_layer.scalar_;
    }

    virtual ~AtenLeakyReluLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenLeakyReluLayer>(new AtenLeakyReluLayer(*this));
    }

    void setScalar(double scalar) { scalar_ = scalar; }

    double getScalar() const { return scalar_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenLeakyReluAttr      ";
        DLOG(INFO) << "    sclar is               " << scalar_;
    }

   private:
    double scalar_ = DBL_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
