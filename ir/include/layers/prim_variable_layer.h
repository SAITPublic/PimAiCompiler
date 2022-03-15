#pragma once

#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler {
namespace ir {

class PrimVariableLayer : public NNLayer {
 public:
    /**
     * @brief PrimVariableLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */

    PrimVariableLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit PrimVariableLayer(const PrimVariableLayer& variable_layer) :
        NNLayer(variable_layer) {
        values_ = variable_layer.values_;
        ntype_ = variable_layer.ntype_;
        single_dtensor_ = variable_layer.single_dtensor_;
        to_remove_ = variable_layer.to_remove_;
        is_constant_ = variable_layer.is_constant_;
    }

    virtual ~PrimVariableLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<PrimVariableLayer>(new PrimVariableLayer(*this));
    }


    void printAttr() {
        Log::IR::I() << "      PrimVariableAttr     ";
    }

    void setAttr(std::shared_ptr<DTensor> data) {
        values_.emplace_back(data);
    }

    std::vector<std::shared_ptr<DTensor>> getAttr() {
        return values_;
    }

    void clearAttr() { values_.clear(); }

    void setNType(std::string ntype) {
        ntype_ = ntype;
    }

    std::string getNType() const {
        return ntype_;
    }

    void setSingleDTensor(bool single_dtensor) { single_dtensor_ = single_dtensor; }

    bool getSingleDTensor() { return single_dtensor_; }

    void setToRemove(bool to_remove) {
        to_remove_ = to_remove;
    }

    bool getToRemove() {
        return to_remove_;
    }

    void setIsConstant(bool is_constant) {
        is_constant_ = is_constant;
    }

    bool getIsConstant() {
        return is_constant_;
    }

 private:
    std::vector<std::shared_ptr<DTensor>> values_;
    std::string ntype_;

    bool single_dtensor_ = false;
    bool to_remove_ = false;
    bool is_constant_ = false;
};

}  // namespace ir
}  // namespace nn_compiler
