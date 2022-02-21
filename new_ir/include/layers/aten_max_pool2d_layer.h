#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenMaxPool2dLayer : public NNLayer {
 public:
    AtenMaxPool2dLayer() {}

    AtenMaxPool2dLayer(std::string name, std::string type)
            : NNLayer(name, type) {
    }

    explicit AtenMaxPool2dLayer(const AtenMaxPool2dLayer &aten_max_pool2d_layer) :
        NNLayer(aten_max_pool2d_layer) {
        this->_kernel_size = aten_max_pool2d_layer._kernel_size;
        this->_pad = aten_max_pool2d_layer._pad;
        this->_stride = aten_max_pool2d_layer._stride;
        this->_dilation = aten_max_pool2d_layer._dilation;
        this->_ceil_mode = aten_max_pool2d_layer._ceil_mode;
    }

    virtual ~AtenMaxPool2dLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenMaxPool2dLayer>(new AtenMaxPool2dLayer(*this));
    }

    void setKernelSize(const std::vector<int64_t> &kernel_size) { _kernel_size = kernel_size; }

    void setPad(const std::vector<int64_t> &pad) { _pad = pad; }

    void setStride(const std::vector<int64_t> &stride) { _stride = stride; }

    void setDilation(const std::vector<int64_t> &dilation) { _dilation = dilation; }

    void setCeilMode(int ceil_mode) { _ceil_mode = ceil_mode; }

    const std::vector<int64_t> &getKernelSize() const { return _kernel_size; }

    const std::vector<int64_t> &getStride() const { return _stride; }

    const std::vector<int64_t> &getDilation() const { return _dilation; }

    const std::vector<int64_t> &getPad() const { return _pad; }

    int getCeilMode() const { return _ceil_mode; }

    void printAttr() {
        Log::IR::I() << "  AtemMaxPool2dAttr";
        Log::IR::I() << "  Kernel size are  " << _kernel_size[0];
        Log::IR::I() << "                   " << _kernel_size[1];
        Log::IR::I() << "  Pad are          " << _pad[0];
        Log::IR::I() << "                   " << _pad[1];
        Log::IR::I() << "  Stride are       " << _stride[0];
        Log::IR::I() << "                   " << _stride[1];
        Log::IR::I() << "  Dilation are     " << _dilation[0];
        Log::IR::I() << "                   " << _dilation[1];
        Log::IR::I() << "  ceil_mode is     " << _ceil_mode;
    }

 private:
    std::vector<int64_t> _kernel_size = {INT64_MIN, INT64_MIN};

    std::vector<int64_t> _stride = {INT64_MIN, INT64_MIN};

    std::vector<int64_t> _pad = {INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN};

    std::vector<int64_t> _dilation = {INT64_MIN, INT64_MIN};

    int _ceil_mode = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
