
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
        this->dtype_ = aten_ones_layer.dtype_;
        this->layout_ = aten_ones_layer.layout_;
        this->device_ = aten_ones_layer.device_;
        this->pin_memory_ = aten_ones_layer.pin_memory_;
    }

    virtual ~AtenOnesLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenOnesLayer>(new AtenOnesLayer(*this));
    }

    void setDType(int64_t dtype) { dtype_ = dtype; }

    void setLayout(int64_t layout) { layout_ = layout; }

    void setDevice(std::string device) { device_ = device; }

    void setPinMemory(int pin_memory) { pin_memory_ = pin_memory; }

    int64_t getDType() const { return dtype_; }

    int64_t getLayout() const { return layout_; }

    std::string getDevice() const { return device_; }

    int getPinMemory() const { return pin_memory_; }

    void printAttr() {
        DLOG(INFO) << "    AtenOnesAttr   ";
        DLOG(INFO) << "    dtype is       " << dtype_;
        DLOG(INFO) << "    layout is      " << layout_;
        DLOG(INFO) << "    device is      " << device_;
        DLOG(INFO) << "    pin_memory is  " << pin_memory_;
    }

 private:
    int64_t dtype_      = INT64_MIN;
    int64_t layout_     = INT64_MIN;
    std::string device_ = "";
    int pin_memory_     = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
