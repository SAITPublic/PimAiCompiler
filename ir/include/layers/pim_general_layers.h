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

#define DECLARE_TORCH_OP_LAYER(op_name)                                                            \
    namespace nn_compiler                                                                          \
    {                                                                                              \
    namespace ir                                                                                   \
    {                                                                                              \
    class op_name##Layer : public NNLayer                                                          \
    {                                                                                              \
       public:                                                                                     \
        op_name##Layer(std::string name, nn_compiler::ir::LayerType type) : NNLayer(name, type) {} \
        explicit op_name##Layer(const op_name##Layer &op) : NNLayer(op) {}                         \
        virtual ~op_name##Layer() {}                                                               \
        virtual std::shared_ptr<NNLayer> clone()                                                   \
        {                                                                                          \
            return std::shared_ptr<op_name##Layer>(new op_name##Layer(*this));                     \
        }                                                                                          \
        void printAttr() { DLOG(INFO) << "     " << #op_name << "Attr   "; }                       \
    };                                                                                             \
    }                                                                                              \
    }

DECLARE_TORCH_OP_LAYER(AtenAbs)
DECLARE_TORCH_OP_LAYER(AtenAnd)
DECLARE_TORCH_OP_LAYER(AtenAny)
DECLARE_TORCH_OP_LAYER(AtenAppend)
DECLARE_TORCH_OP_LAYER(AtenBitwiseNot)
DECLARE_TORCH_OP_LAYER(AtenBmm)
DECLARE_TORCH_OP_LAYER(AtenBool)
DECLARE_TORCH_OP_LAYER(AtenCeil)
DECLARE_TORCH_OP_LAYER(AtenClear)
DECLARE_TORCH_OP_LAYER(AtenCpu)
DECLARE_TORCH_OP_LAYER(AtenCuda)
DECLARE_TORCH_OP_LAYER(AtenCumsum)
DECLARE_TORCH_OP_LAYER(AtenDetach)
DECLARE_TORCH_OP_LAYER(AtenDim)
DECLARE_TORCH_OP_LAYER(AtenDiv)
DECLARE_TORCH_OP_LAYER(AtenEinsum)
DECLARE_TORCH_OP_LAYER(AtenEq)
DECLARE_TORCH_OP_LAYER(AtenEqual)
DECLARE_TORCH_OP_LAYER(AtenFill)
DECLARE_TORCH_OP_LAYER(AtenFloorDivide)
DECLARE_TORCH_OP_LAYER(AtenGe)
DECLARE_TORCH_OP_LAYER(AtenGt)
DECLARE_TORCH_OP_LAYER(AtenIndex)
DECLARE_TORCH_OP_LAYER(AtenInt)
DECLARE_TORCH_OP_LAYER(AtenIntImplicit)
DECLARE_TORCH_OP_LAYER(AtenIs)
DECLARE_TORCH_OP_LAYER(AtenIsInf)
DECLARE_TORCH_OP_LAYER(AtenIsNot)
DECLARE_TORCH_OP_LAYER(AtenItem)
DECLARE_TORCH_OP_LAYER(AtenLe)
DECLARE_TORCH_OP_LAYER(AtenLen)
DECLARE_TORCH_OP_LAYER(AtenList)
DECLARE_TORCH_OP_LAYER(AtenLog)
DECLARE_TORCH_OP_LAYER(AtenLt)
DECLARE_TORCH_OP_LAYER(AtenMaskedSelect)
DECLARE_TORCH_OP_LAYER(AtenMatmul)
DECLARE_TORCH_OP_LAYER(AtenMean)
DECLARE_TORCH_OP_LAYER(AtenMul)
DECLARE_TORCH_OP_LAYER(AtenNeg)
DECLARE_TORCH_OP_LAYER(AtenNe)
DECLARE_TORCH_OP_LAYER(AtenNot)
DECLARE_TORCH_OP_LAYER(AtenPermute)
DECLARE_TORCH_OP_LAYER(AtenPow)
DECLARE_TORCH_OP_LAYER(AtenRelu)
DECLARE_TORCH_OP_LAYER(AtenReshape)
DECLARE_TORCH_OP_LAYER(AtenRemainder)
DECLARE_TORCH_OP_LAYER(AtenRepeat)
DECLARE_TORCH_OP_LAYER(AtenRsqrt)
DECLARE_TORCH_OP_LAYER(AtenTanh)
DECLARE_TORCH_OP_LAYER(AtenTensor)
DECLARE_TORCH_OP_LAYER(AtenTypeAs)
DECLARE_TORCH_OP_LAYER(AtenView)
DECLARE_TORCH_OP_LAYER(AtenWhere)
DECLARE_TORCH_OP_LAYER(AtenZeros)
DECLARE_TORCH_OP_LAYER(AtenZerosLike)

DECLARE_TORCH_OP_LAYER(PrimBlock)
DECLARE_TORCH_OP_LAYER(PrimData)
DECLARE_TORCH_OP_LAYER(PrimDevice)
DECLARE_TORCH_OP_LAYER(PrimDtype)
DECLARE_TORCH_OP_LAYER(PrimInput)
DECLARE_TORCH_OP_LAYER(PrimListConstruct)
DECLARE_TORCH_OP_LAYER(PrimListUnpack)
DECLARE_TORCH_OP_LAYER(PrimOutput)
DECLARE_TORCH_OP_LAYER(PrimRaiseException)
DECLARE_TORCH_OP_LAYER(PrimSetAttr)
DECLARE_TORCH_OP_LAYER(PrimTupleConstruct)
DECLARE_TORCH_OP_LAYER(PrimTupleUnpack)
DECLARE_TORCH_OP_LAYER(PrimType)
DECLARE_TORCH_OP_LAYER(PrimUncheckedCast)
DECLARE_TORCH_OP_LAYER(PrimUninitialized)
