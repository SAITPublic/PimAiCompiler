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
        this->non_blocking_ = aten_copy_layer.non_blocking_;
    }

    virtual ~AtenCopyLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenCopyLayer>(new AtenCopyLayer(*this));
    }

    void setNonBlocking(int nonblocking) { non_blocking_ = nonblocking; }

    int getNonBlocking() const { return non_blocking_; }

    void printAttr() {
        DLOG(INFO) << "    AtenCopyAttr      ";
        DLOG(INFO) << "    non_blocking is   "<< non_blocking_;
    }

 private:
    int  non_blocking_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
