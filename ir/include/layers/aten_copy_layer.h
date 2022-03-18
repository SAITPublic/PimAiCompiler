#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenCopyLayer : public NNLayer {
 public:
    /**
     * @brief AtenCopyLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenCopyLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenCopyLayer(const AtenCopyLayer& aten_copy_layer) :
        NNLayer(aten_copy_layer) {
        this->_non_blocking = aten_copy_layer._non_blocking;
    }

    virtual ~AtenCopyLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenCopyLayer>(new AtenCopyLayer(*this));
    }

    void setNonBlocking(int nonblocking) { _non_blocking = nonblocking; }

    int getNonBlocking() const { return _non_blocking; }

    void printAttr() {
        DLOG(INFO) << "    AtenCopyAttr      ";
        DLOG(INFO) << "    non_blocking is   "<< _non_blocking;
    }

 private:
    int  _non_blocking = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
