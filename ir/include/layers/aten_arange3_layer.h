#pragma once

#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler {
namespace ir {

/*
  len(args) == 7: 
        aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)  
 */
class AtenArange3Layer : public NNLayer {
 public:
    AtenArange3Layer(std::string name, LayerType type) : NNLayer(name, type) {
    }

    explicit AtenArange3Layer(const AtenArange3Layer& aten_arange_layer) :  NNLayer(aten_arange_layer) {
        this->_start      = aten_arange_layer._start;
        this->_end        = aten_arange_layer._end;
        this->_step       = aten_arange_layer._step;
        this->_dtype      = aten_arange_layer._dtype;
        this->_layout     = aten_arange_layer._layout;
        this->_device     = aten_arange_layer._device;
        this->_pin_memory = aten_arange_layer._pin_memory;
    }

    virtual ~AtenArange3Layer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenArange3Layer> ( new AtenArange3Layer(*this) );
    }

    void setStart(int64_t start) { _start = start; }

    int64_t getStart() const { return _start; }

    void setEnd(int64_t end) { _end = end; }

    int64_t getEnd() const { return _end; }

    void setStep(int64_t step) { _step = step; }

    int64_t getStep() const { return _step; }

    void setDtype(int64_t dtype) { _dtype = dtype; }

    int64_t getDtype() const { return _dtype; }

    void setLayout(int64_t layout) { _layout = layout; }

    int64_t getLayout() const { return _layout; }

    void setDevice(std::string device) { _device = device; }

    std::string getDevice() const { return _device; }

    void setPinMemory(int pin_memory) { _pin_memory = pin_memory; }

    int getPinMemory() const { return _pin_memory; }

    void printAttr() {
        DLOG(INFO) << "    AtenArangeAttr          ";
        DLOG(INFO) << "    start is                "<< _start;
        DLOG(INFO) << "    end is                  "<< _end;
        DLOG(INFO) << "    step is                 "<< _step;
        DLOG(INFO) << "    dtype is                "<< _dtype;
        DLOG(INFO) << "    layout is               "<< _layout;
        DLOG(INFO) << "    device is               "<< _device;
        DLOG(INFO) << "    pin_memory is           "<< _pin_memory;
    }

 private:
    int64_t _start      = INT64_MIN;
    int64_t _end        = INT64_MIN;
    int64_t _step       = INT64_MIN;
    int64_t _dtype      = INT64_MIN;
    int64_t _layout     = INT64_MIN;
    std::string _device = "";
    int _pin_memory     = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
