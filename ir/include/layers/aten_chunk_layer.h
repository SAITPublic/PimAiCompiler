/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenChunkLayer : public NNLayer
{
   public:
    AtenChunkLayer() {}

    AtenChunkLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenChunkLayer(const AtenChunkLayer& aten_chunk_layer) : NNLayer(aten_chunk_layer)
    {
        this->chunks_ = aten_chunk_layer.chunks_;
        this->dim_ = aten_chunk_layer.dim_;
    }

    virtual ~AtenChunkLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenChunkLayer>(new AtenChunkLayer(*this)); }

    void setChunks(int chunks) { chunks_ = chunks; }

    int getChunks() { return chunks_; }

    void setDim(int dim) { dim_ = dim; }

    int getDim() { return dim_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenChunkAttr          ";
        DLOG(INFO) << "    chunks is              " << chunks_;
        DLOG(INFO) << "    dim is                 " << dim_;
    }

   private:
    int chunks_ = INT32_MAX;
    int dim_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
