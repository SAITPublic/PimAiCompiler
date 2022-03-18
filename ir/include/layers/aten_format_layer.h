#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenFormatLayer : public NNLayer
{
   public:
    AtenFormatLayer() {}

    AtenFormatLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenFormatLayer(const AtenFormatLayer &aten_format_layer) : NNLayer(aten_format_layer)
    {
        this->assembly_format_ = aten_format_layer.assembly_format_;
    }

    virtual ~AtenFormatLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenFormatLayer>(new AtenFormatLayer(*this)); }

    void setAssemblyFormat(std::string assembly_format) { assembly_format_ = assembly_format; }

    std::string getAssemblyFormat() const { return assembly_format_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenFormatAttr      ";
        DLOG(INFO) << "    assembly_format is  " << assembly_format_;
    }

   private:
    std::string assembly_format_;
};

}  // namespace ir
}  // namespace nn_compiler
