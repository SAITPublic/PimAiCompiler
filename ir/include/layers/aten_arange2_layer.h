#pragma once

#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler {
namespace ir {

/*
  len(args) == 6: 
        aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
 */
class AtenArange2Layer : public NNLayer {
 public:
    AtenArange2Layer(std::string name, LayerType type) : NNLayer(name, type) {
    }

    explicit AtenArange2Layer(const AtenArange2Layer& aten_arange_layer) :  NNLayer(aten_arange_layer) {
        this->start_      = aten_arange_layer.start_;
        this->end_        = aten_arange_layer.end_;
        this->dtype_      = aten_arange_layer.dtype_;
        this->layout_     = aten_arange_layer.layout_;
        this->device_     = aten_arange_layer.device_;
        this->pin_memory_ = aten_arange_layer.pin_memory_;
    }

    virtual ~AtenArange2Layer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenArange2Layer> ( new AtenArange2Layer(*this) );
    }

    void setStart(int64_t start) { start_ = start; }

    int64_t getStart() const { return start_; }

    void setEnd(int64_t end) { end_ = end; }

    int64_t getEnd() const { return end_; }

    void setDtype(int64_t dtype) { dtype_ = dtype; }

    int64_t getDtype() const { return dtype_; }

    void setLayout(int64_t layout) { layout_ = layout; }

    int64_t getLayout() const { return layout_; }

    void setDevice(std::string device) { device_ = device; }

    std::string getDevice() const { return device_; }

    void setPinMemory(int pin_memory) { pin_memory_ = pin_memory; }

    int getPinMemory() const { return pin_memory_; }

    void printAttr() {
        DLOG(INFO) << "    AtenArangeAttr          ";
        DLOG(INFO) << "    start is                "<< start_;
        DLOG(INFO) << "    end is                 "<< end_;
        DLOG(INFO) << "    dtype is                "<< dtype_;
        DLOG(INFO) << "    layout is               "<< layout_;
        DLOG(INFO) << "    device is               "<< device_;
        DLOG(INFO) << "    pin_memory is           "<< pin_memory_;
    }

 private:
    int64_t start_      = INT64_MIN;
    int64_t end_        = INT64_MIN;
    int64_t dtype_      = INT64_MIN;
    int64_t layout_     = INT64_MIN;
    std::string device_ = "";
    int pin_memory_     = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
