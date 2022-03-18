#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

//Tensor to(const Tensor& self, Device device, ScalarType dtype, bool non_blocking,
//                               bool copy, c10::optional<c10::MemoryFormat> optional_memory_format)
class AtenTo1Layer : public NNLayer {
 public:
    AtenTo1Layer() {}

    AtenTo1Layer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenTo1Layer(const AtenTo1Layer& aten_to_layer) : NNLayer(aten_to_layer) {
        this->_dtype = aten_to_layer._dtype;
        this->_non_blocking = aten_to_layer._non_blocking;
        this->_copy = aten_to_layer._copy;
        this->_optional_memory_format = aten_to_layer._optional_memory_format;
    }

    virtual ~AtenTo1Layer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenTo1Layer>(new AtenTo1Layer(*this));
    }

    void setDType(int64_t dtype) { _dtype = dtype; }

    void setNonBlocking(int nonblocking) { _non_blocking = nonblocking; }

    void setCopy(int copy) { _copy = copy; }

    void setOptionalMemoryFormat(int optional_memory_format) {
        _optional_memory_format = optional_memory_format;
    }

    int64_t getDType() const { return _dtype; }

    int getNonBlocking() const { return _non_blocking; }

    int getCopy() const { return _copy; }

    int getOptionalMemoryFormat() { return _optional_memory_format; }

    void printAttr() {
        DLOG(INFO) << "    AtenToAttr                     ";
        DLOG(INFO) << "    dtype is                       " << _dtype;
        DLOG(INFO) << "    non_blocking                   " << _non_blocking;
        DLOG(INFO) << "    copy is                        " << _copy;
        DLOG(INFO) << "    optional_memory_format is      " << _optional_memory_format;
    }

 private:
    int64_t _dtype       = INT64_MIN;
    int    _non_blocking = INT32_MAX;
    int    _copy         = INT32_MAX;
    /* according to pytorch/c10/core/MemoryFormat.h,
       enum MemoryFormat: { Contiguous, Preserve, ChannelsLast, ChannelsLast3d }
       -1 stands for _optional_memory_format = NONE.
     */
    int _optional_memory_format = -1;
};

} // namespace ir
} // namespace nn_compiler
