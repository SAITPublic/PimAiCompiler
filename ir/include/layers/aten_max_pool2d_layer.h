#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenMaxPool2dLayer : public NNLayer
{
   public:
    AtenMaxPool2dLayer() {}

    AtenMaxPool2dLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenMaxPool2dLayer(const AtenMaxPool2dLayer &aten_max_pool2d_layer) : NNLayer(aten_max_pool2d_layer)
    {
        this->kernel_size_ = aten_max_pool2d_layer.kernel_size_;
        this->pad_ = aten_max_pool2d_layer.pad_;
        this->stride_ = aten_max_pool2d_layer.stride_;
        this->dilation_ = aten_max_pool2d_layer.dilation_;
        this->ceil_mode_ = aten_max_pool2d_layer.ceil_mode_;
    }

    virtual ~AtenMaxPool2dLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenMaxPool2dLayer>(new AtenMaxPool2dLayer(*this));
    }

    void setKernelSize(const std::vector<int64_t> &kernel_size) { kernel_size_ = kernel_size; }

    void setPad(const std::vector<int64_t> &pad) { pad_ = pad; }

    void setStride(const std::vector<int64_t> &stride) { stride_ = stride; }

    void setDilation(const std::vector<int64_t> &dilation) { dilation_ = dilation; }

    void setCeilMode(int ceil_mode) { ceil_mode_ = ceil_mode; }

    const std::vector<int64_t> &getKernelSize() const { return kernel_size_; }

    const std::vector<int64_t> &getStride() const { return stride_; }

    const std::vector<int64_t> &getDilation() const { return dilation_; }

    const std::vector<int64_t> &getPad() const { return pad_; }

    int getCeilMode() const { return ceil_mode_; }

    void printAttr()
    {
        DLOG(INFO) << "  AtemMaxPool2dAttr";
        DLOG(INFO) << "  Kernel size are  " << kernel_size_[0];
        DLOG(INFO) << "                   " << kernel_size_[1];
        DLOG(INFO) << "  Pad are          " << pad_[0];
        DLOG(INFO) << "                   " << pad_[1];
        DLOG(INFO) << "  Stride are       " << stride_[0];
        DLOG(INFO) << "                   " << stride_[1];
        DLOG(INFO) << "  Dilation are     " << dilation_[0];
        DLOG(INFO) << "                   " << dilation_[1];
        DLOG(INFO) << "  ceil_mode is     " << ceil_mode_;
    }

   private:
    std::vector<int64_t> kernel_size_ = {INT64_MIN, INT64_MIN};

    std::vector<int64_t> stride_ = {INT64_MIN, INT64_MIN};

    std::vector<int64_t> pad_ = {INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN};

    std::vector<int64_t> dilation_ = {INT64_MIN, INT64_MIN};

    int ceil_mode_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
