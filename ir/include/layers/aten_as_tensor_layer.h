#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenAsTensorLayer : public NNLayer
{
   public:
    AtenAsTensorLayer() {}

    AtenAsTensorLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenAsTensorLayer(const AtenAsTensorLayer& aten_as_tensor_layer) : NNLayer(aten_as_tensor_layer)
    {
        this->dtype_ = aten_as_tensor_layer.dtype_;
        this->device_ = aten_as_tensor_layer.device_;
    }

    virtual ~AtenAsTensorLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenAsTensorLayer>(new AtenAsTensorLayer(*this));
    }

    void setDtype(int64_t dtype) { dtype_ = dtype; }

    int64_t getDtype() const { return dtype_; }

    void setDevice(std::string device) { device_ = device; }

    std::string getDevice() const { return device_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenAsTensorAttr       ";
        DLOG(INFO) << "    dtype is               " << dtype_;
        DLOG(INFO) << "    device is              " << device_;
    }

   private:
    int64_t dtype_ = INT64_MIN;
    std::string device_ = "";
};

}  // namespace ir
}  // namespace nn_compiler
