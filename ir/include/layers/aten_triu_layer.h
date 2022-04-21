#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{

class AtenTriuLayer : public NNLayer
{
   public:

    AtenTriuLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenTriuLayer(const AtenTriuLayer& aten_triu_layer) : NNLayer(aten_triu_layer)
    {
        this->Diagonal_ = aten_triu_layer.Diagonal_;
    }

    virtual ~AtenTriuLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenTriuLayer>(new AtenTriuLayer(*this)); }

    void setDiagonal(int64_t Diagonal) { Diagonal_ = Diagonal; }

    int64_t getDiagonal() const { return Diagonal_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenTriuAttr     ";
        DLOG(INFO) << "    Diagonal is           " << Diagonal_;
    }

   private:
    int64_t Diagonal_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
