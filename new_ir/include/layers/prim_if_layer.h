#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

//def zeros(g, sizes, dtype, layout, device, pin_memory=False)

class PrimIfLayer : public NNLayer {
 public:
    /**
     * @brief PrimIfLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    PrimIfLayer(std::string name, std::string type)
            : NNLayer(name, type) {
    }

    explicit PrimIfLayer(const PrimIfLayer& prim_if_layer) :
        NNLayer(prim_if_layer) {
        this->_then_net = prim_if_layer._then_net;
        this->_else_net = prim_if_layer._else_net;
    }

    virtual ~PrimIfLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<PrimIfLayer>(new PrimIfLayer(*this));
    }

    void setThenNet(const std::string netName) {
        _then_net = netName;
    }

    void setElseNet(const std::string netName) {
        _else_net = netName;
    }

    const std::string getThenNet() const {
        return _then_net;
    }

    const std::string getElseNet() const {
        return _else_net;
    }

    void printAttr() {
        Log::IR::I() << "    PrimIfAttr        ";
        Log::IR::I() << "    Then net is       " << _then_net;
        Log::IR::I() << "    Else net is       " << _else_net;
    }

 private:
    std::string _then_net;
    std::string _else_net;
};

} // namespace ir
} // namespace nn_compiler
