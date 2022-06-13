#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenFullLikeLayer : public NNLayer
{
   public:
    AtenFullLikeLayer() {}

    AtenFullLikeLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenFullLikeLayer(const AtenFullLikeLayer& aten_full_like_layer) : NNLayer(aten_full_like_layer)
    {
        this->full_value_ = aten_full_like_layer.full_value_;
    }

    virtual ~AtenFullLikeLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenFullLikeLayer>(new AtenFullLikeLayer(*this));
    }

    void setFullValue(int full_value) { full_value_ = full_value; }

    int getFullValue() const { return full_value_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenFullLikeLayer       ";
        DLOG(INFO) << "    full_value is           " << full_value_;
    }

   private:
    int full_value_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
