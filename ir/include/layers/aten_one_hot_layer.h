
#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenOneHotLayer : public NNLayer
{
   public:
    AtenOneHotLayer() {}

    AtenOneHotLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenOneHotLayer(const AtenOneHotLayer& aten_one_hot_layer) : NNLayer(aten_one_hot_layer)
    {
        this->num_classes_ = aten_one_hot_layer.num_classes_;
    }

    virtual ~AtenOneHotLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenOneHotLayer>(new AtenOneHotLayer(*this)); }

    void setNumClasses(int64_t num_classes) { num_classes_ = num_classes; }

    int64_t getNumClasses() const { return num_classes_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenOneHotAttr   ";
        DLOG(INFO) << "    num_classes_ is       " << num_classes_;
    }

   private:
    int64_t num_classes_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
