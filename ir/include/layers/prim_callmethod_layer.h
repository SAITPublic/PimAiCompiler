#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
// prim::CallMethod(targetNetworkName, inputs)
class PrimCallMethodLayer : public NNLayer
{
   public:
    PrimCallMethodLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit PrimCallMethodLayer(const PrimCallMethodLayer& callmethod_layer) : NNLayer(callmethod_layer)
    {
        this->target_network_name_ = callmethod_layer.target_network_name_;
    }

    virtual ~PrimCallMethodLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<PrimCallMethodLayer>(new PrimCallMethodLayer(*this));
    }

    void printAttr() { DLOG(INFO) << "    target network name is     " << target_network_name_; }

    void setAttr(std::string target_network_name) { target_network_name_ = target_network_name; }

    std::string getAttr() const { return target_network_name_; }

   private:
    std::string target_network_name_;
};

}  // namespace ir
}  // namespace nn_compiler
