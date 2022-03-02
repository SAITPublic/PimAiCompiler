#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenAsTensorLayer : public NNLayer {
 public:
    AtenAsTensorLayer() {}

    AtenAsTensorLayer(std::string name, LayerType type)
                     : NNLayer(name, type) {
    }

    explicit AtenAsTensorLayer(const AtenAsTensorLayer& aten_as_tensor_layer) : 
        NNLayer(aten_as_tensor_layer) {
        this->_dtype  = aten_as_tensor_layer._dtype;
        this->_device = aten_as_tensor_layer._device;
    }

    virtual ~AtenAsTensorLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenAsTensorLayer>(new AtenAsTensorLayer(*this));
    }

    void setDtype(int64_t dtype) { _dtype = dtype; }

    int64_t getDtype() const { return _dtype; }

    void setDevice(std::string device) { _device = device; }

    std::string getDevice() const { return _device; }

    void printAttr() {
        Log::IR::I() << "    AtenAsTensorAttr       ";
        Log::IR::I() << "    dtype is               " << _dtype;
        Log::IR::I() << "    device is              " << _device;
    }

 private:
    int64_t _dtype      = INT64_MIN;
    std::string _device = "";
};

} // namespace ir
} // namespace nn_compiler
