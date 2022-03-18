#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

// prim::CallMethod(targetNetworkName, inputs)
class PrimCallMethodLayer : public NNLayer {
 public:
    PrimCallMethodLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit PrimCallMethodLayer(const PrimCallMethodLayer& callmethod_layer) :
        NNLayer(callmethod_layer) {
        this->_target_network_name = callmethod_layer._target_network_name;
    }

    virtual ~PrimCallMethodLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<PrimCallMethodLayer>(new PrimCallMethodLayer(*this));
    }


    void printAttr() {
        DLOG(INFO) << "    target network name is     " << _target_network_name;
    }

    void setAttr(std::string target_network_name) {
        _target_network_name = target_network_name;
    }

    std::string getAttr() const {
        return _target_network_name;
    }

 private:
    std::string _target_network_name;
};

} // namespace ir
} // namespace nn_compiler
