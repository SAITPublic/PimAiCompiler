#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenTo2Layer : public NNLayer {
 public:
    AtenTo2Layer() {}

    AtenTo2Layer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenTo2Layer(const AtenTo2Layer& aten_to_layer) : NNLayer(aten_to_layer) {
        this->_non_blocking = aten_to_layer._non_blocking;
        this->_copy = aten_to_layer._copy;
        this->_optional_memory_format = aten_to_layer._optional_memory_format;
    }

    virtual ~AtenTo2Layer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenTo2Layer>(new AtenTo2Layer(*this));
    }

    void setNonBlocking(int nonblocking) { _non_blocking = nonblocking; }

    void setCopy(int copy) { _copy = copy; }

    void setOptionalMemoryFormat(int optional_memory_format) {
        _optional_memory_format = optional_memory_format;
    }

    int getNonBlocking() const { return _non_blocking; }

    int getCopy() const { return _copy; }

    int getOptionalMemoryFormat() { return _optional_memory_format; }

    void printAttr() {
        DLOG(INFO) << "    AtenToAttr                     ";
        DLOG(INFO) << "    non_blocking                   " << _non_blocking;
        DLOG(INFO) << "    copy is                        " << _copy;
        DLOG(INFO) << "    optional_memory_format is      " << _optional_memory_format;
    }

 private:
    int    _non_blocking = INT32_MAX;
    int    _copy         = INT32_MAX;
    int _optional_memory_format = -1;
};

} // namespace ir
} // namespace nn_compiler
