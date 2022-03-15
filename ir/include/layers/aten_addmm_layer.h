#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenAddmmLayer : public NNLayer {
 public:
    /**
     * @brief AtenAddmmLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenAddmmLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenAddmmLayer(const AtenAddmmLayer& aten_addmm_layer) : NNLayer(aten_addmm_layer) {
        act_type_ = aten_addmm_layer.act_type_;
    }

    virtual ~AtenAddmmLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenAddmmLayer>(new AtenAddmmLayer(*this));
    }

    void printAttr() {
        Log::IR::I() << "    AtenAddmmAttr      ";
    }

    void set_act_type(const std::string& type) {
           act_type_ = type;
    }

    std::string get_act_type() {
        return act_type_;
    }

 private:
    std::string act_type_ = "none";
};

}  // namespace ir
}  // namespace nn_compiler
