#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenCloneLayer : public NNLayer
{
   public:
    AtenCloneLayer() {}

    AtenCloneLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenCloneLayer(const AtenCloneLayer& aten_clone_layer) : NNLayer(aten_clone_layer)
    {
        this->memory_format_ = aten_clone_layer.memory_format_;
    }

    virtual ~AtenCloneLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenCloneLayer>(new AtenCloneLayer(*this)); }

    void setMemoryFormat(int memory_format) { memory_format_ = memory_format; }

    int getMemoryFormat() { return memory_format_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenCloneAttr          ";
        DLOG(INFO) << "    memory_format is       " << memory_format_;
    }

   private:
    int memory_format_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
