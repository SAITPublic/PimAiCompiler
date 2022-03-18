
#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenOnesLayer : public NNLayer {
 public:
    AtenOnesLayer() {}

    AtenOnesLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenOnesLayer(const AtenOnesLayer& aten_ones_layer) :
        NNLayer(aten_ones_layer) {
        this->_dtype = aten_ones_layer._dtype;
        this->_layout = aten_ones_layer._layout;
        this->_device = aten_ones_layer._device;
        this->_pin_memory = aten_ones_layer._pin_memory;
    }

    virtual ~AtenOnesLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenOnesLayer>(new AtenOnesLayer(*this));
    }

    void setDType(int64_t dtype) { _dtype = dtype; }

    void setLayout(int64_t layout) { _layout = layout; }

    void setDevice(std::string device) { _device = device; }

    void setPinMemory(int pin_memory) { _pin_memory = pin_memory; }

    int64_t getDType() const { return _dtype; }

    int64_t getLayout() const { return _layout; }

    std::string getDevice() const { return _device; }

    int getPinMemory() const { return _pin_memory; }

    void printAttr() {
        DLOG(INFO) << "    AtenOnesAttr   ";
        DLOG(INFO) << "    dtype is       " << _dtype;
        DLOG(INFO) << "    layout is      " << _layout;
        DLOG(INFO) << "    device is      " << _device;
        DLOG(INFO) << "    pin_memory is  " << _pin_memory;
    }

 private:
    int64_t _dtype      = INT64_MIN;
    int64_t _layout     = INT64_MIN;
    std::string _device = "";
    int _pin_memory     = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
