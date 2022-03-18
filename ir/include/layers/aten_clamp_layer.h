#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenClampLayer : public NNLayer
{
   public:
    AtenClampLayer() {}

    AtenClampLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenClampLayer(const AtenClampLayer& aten_clamp_layer) : NNLayer(aten_clamp_layer)
    {
        this->min_ = aten_clamp_layer.min_;
        this->max_ = aten_clamp_layer.max_;
    }

    virtual ~AtenClampLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenClampLayer>(new AtenClampLayer(*this)); }

    void setMin(int min) { min_ = min; }

    int getMin() { return min_; }

    void setMax(int max) { max_ = max; }

    int getMax() { return max_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenClampAttr          ";
        DLOG(INFO) << "    min is                 " << min_;
        DLOG(INFO) << "    max is                 " << max_;
    }

   private:
    int min_ = INT32_MAX;
    int max_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
