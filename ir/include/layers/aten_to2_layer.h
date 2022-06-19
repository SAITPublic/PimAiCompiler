#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
// Tensor to(const Tensor & self, ScalarType dtype,
//     bool non_blocking=false, bool copy=false, c10::optional<MemoryFormat> memory_format=c10::nullopt);

class AtenTo2Layer : public NNLayer
{
   public:
    AtenTo2Layer() {}

    AtenTo2Layer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenTo2Layer(const AtenTo2Layer& aten_to_layer) : NNLayer(aten_to_layer)
    {
        this->dtype_ = aten_to_layer.dtype_;
        this->non_blocking_ = aten_to_layer.non_blocking_;
        this->copy_ = aten_to_layer.copy_;
        this->optional_memory_format_ = aten_to_layer.optional_memory_format_;
    }

    virtual ~AtenTo2Layer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenTo2Layer>(new AtenTo2Layer(*this)); }

    void setDType(int64_t dtype) { dtype_ = dtype; }

    void setNonBlocking(int nonblocking) { non_blocking_ = nonblocking; }

    void setCopy(int copy) { copy_ = copy; }

    void setOptionalMemoryFormat(int optional_memory_format) { optional_memory_format_ = optional_memory_format; }

    int64_t getDType() const { return dtype_; }

    int getNonBlocking() const { return non_blocking_; }

    int getCopy() const { return copy_; }

    int getOptionalMemoryFormat() { return optional_memory_format_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenTo2Attr                    ";
        DLOG(INFO) << "    dtype is                       " << dtype_;
        DLOG(INFO) << "    non_blocking                   " << non_blocking_;
        DLOG(INFO) << "    copy is                        " << copy_;
        DLOG(INFO) << "    optional_memory_format is      " << optional_memory_format_;
    }

   private:
    int64_t dtype_ = INT64_MIN;
    int non_blocking_ = INT32_MAX;
    int copy_ = INT32_MAX;
    int optional_memory_format_ = -1;
};

}  // namespace ir
}  // namespace nn_compiler
