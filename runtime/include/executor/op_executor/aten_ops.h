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

#ifndef NNCOMPILER_ATEN_OP_H
#define NNCOMPILER_ATEN_OP_H

#include <torch/script.h>

#include "ATen/ATen.h"

namespace nn_compiler
{
namespace runtime
{
namespace op_executor
{

at::Tensor atenAbs(const at::Tensor &self);

at::Tensor atenAdd(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha = 1);

at::Tensor atenAdd(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha = 1);

int64_t atenAdd(int64_t &self, int64_t other, int64_t alpha = 1);

at::Tensor &atenAdd_(at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha = 1);

at::Tensor atenAddmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2, const at::Scalar &beta = 1,
                     const at::Scalar &alpha = 1);

bool atenAnd(bool &a, bool &b);

at::Tensor atenAny(const at::Tensor &self);

void atenAppend(c10::List<at::IValue> &list, at::IValue el);

at::Tensor atenArange1(at::Scalar end, const at::TensorOptions &options);

at::Tensor atenArange2(at::Scalar start, at::Scalar end, const at::TensorOptions &options);

at::Tensor atenArange3(at::Scalar start, at::Scalar end, at::Scalar step, const at::TensorOptions &options);

at::Tensor atenArgmax(at::Tensor &self, c10::optional<int64_t> dim = c10::nullopt, bool keepdim = false);

at::Tensor atenAsTensor(at::Tensor &self, at::ScalarType dtype, at::Device device);

at::Tensor atenBatchNorm2d(const at::Tensor &input, const c10::optional<at::Tensor> &weight,
                           const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &running_mean,
                           const c10::optional<at::Tensor> &running_var, bool training, double momentum, double eps,
                           bool cudnn_enabled);

at::Tensor atenBitwiseNot(const at::Tensor &self);

at::Tensor atenBmm(const at::Tensor &self, const at::Tensor &other);

bool atenBool(const at::Tensor &self);

bool atenBool(const int64_t &i);

bool atenBool(const double &d);

at::Tensor atenCat(at::TensorList tensors, int64_t dim = 0);

at::Tensor atenCat(at::TensorList tensors, at::Dimname dim);

at::Tensor atenCeil(const at::Tensor &self);

std::vector<at::Tensor> atenChunk(const at::Tensor &self, int chunks, int dim);

at::Tensor atenClamp(const at::Tensor &self, const at::Scalar &min, const at::Scalar &max);

template <typename T>
void atenClear(at::List<T> &list)
{
    list.clear();
}

at::Tensor atenClone(const at::Tensor &self, c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

at::Tensor atenContiguous(const at::Tensor &self, at::MemoryFormat memory_format);

at::Tensor atenConv2d(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias, at::IntArrayRef stride,
                      at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);

at::Tensor &atenCopy_(at::Tensor &self, const at::Tensor &src, bool non_blocking = false);

at::Tensor atenCpu(const at::Tensor &self);

at::Tensor atenCuda(const at::Tensor &self);

at::Tensor atenCumsum(const at::Tensor &self, int64_t dim, c10::optional<at::ScalarType> dtype = c10::nullopt);

at::Tensor atenCumsum(const at::Tensor &self, at::Dimname dim, c10::optional<at::ScalarType> dtype = c10::nullopt);

int64_t atenDeriveIndex(int64_t index, int64_t start, int64_t step);

at::Tensor atenDetach(const at::Tensor &self);

int64_t atenDim(const at::Tensor &tensor);

at::Tensor atenDiv(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenDiv(const at::Tensor &self, const at::Tensor &other, c10::optional<c10::string_view> rounding_mode);

at::Tensor atenDiv(const at::Tensor &self, const at::Scalar &other);

at::Tensor atenDiv(const at::Tensor &self, const at::Scalar &other, c10::optional<c10::string_view> rounding_mode);

at::Tensor atenDropout(const at::Tensor &input, double p, bool train);

at::Tensor &atenDropout_(at::Tensor &self, double p, bool train);

at::Tensor atenEinsum(std::string equation, at::TensorList tensors);

at::Tensor atenEmbedding(const at::Tensor &weight, const at::Tensor &indices, int64_t padding_idx = -1,
                         bool scale_grad_by_freq = false, bool sparse = false);

at::Tensor atenEq(const at::Tensor &self, const at::Scalar &other);

at::Tensor atenEq(const at::Tensor &self, const at::Tensor &other);

bool atenEqual(const at::Tensor &self, const at::Tensor &other);

bool atenEq(const at::Scalar &self, const at::Scalar &other);

at::Tensor atenExpand(const at::Tensor &self, at::IntArrayRef size, bool implicit = false);

at::Tensor &atenFill(at::Tensor &self, at::Scalar value);

at::Tensor &atenFill(at::Tensor &self, at::Tensor &other);

at::Tensor atenFloorDivide(const at::Tensor &self, at::Scalar value);

at::Tensor atenFloorDivide(const at::Tensor &self, const at::Tensor &other);

static std::string atenFormat(const std::string &fmt)
{
    int index = fmt.find("{}");
    if (index != std::string::npos) {
        DLOG(FATAL) << "Too few arguments for format string:" << fmt;
    }
    return fmt;
}

at::Tensor atenFullLike(const at::Tensor &self, at::Scalar fill_value, const at::TensorOptions &options = {},
                        c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

/**
 * @brief  Replace {} using strings in variadic arguments like python format()
 * @param  fmt formatter string which contains curly braces {}
 * @param  next the first string after fmt
 * @param  args argument list which holds variadic arguments
 * @return A new string after formatting
 */
template <typename T, typename... Types>
std::string atenFormat(std::string &fmt, const T &next, const Types &...args)
{
    int index = fmt.find("{}");
    if (index == std::string::npos) {
        return fmt;
    }
    std::stringstream oss;
    oss << next;
    fmt.replace(index, 2, oss.str());
    return atenFormat(fmt, args...);
}

at::Tensor atenGather(const at::Tensor &self, int64_t dim, const at::Tensor &index, bool sparse_grad);

at::Tensor atenGe(const at::Tensor &self, const at::Scalar &other);

at::Tensor atenGe(const at::Tensor &self, const at::Tensor &other);

at::IValue atenGetItem(const c10::List<at::IValue> &list, int idx);

at::Tensor atenGt(const at::Tensor &self, const at::Scalar &other);

at::Tensor atenGt(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenIndex(const at::Tensor &self, const c10::List<c10::optional<at::Tensor>> &indices);

at::Tensor &atenIndexPut(at::Tensor &self, const c10::List<c10::optional<at::Tensor>> &indices,
                         const at::Tensor &values, bool accumulate);

at::Tensor atenIndexSelect(const at::Tensor &self, int64_t dim, const at::Tensor &index);

int64_t atenInt(const at::Tensor &a);

int64_t atenInt(const bool &b);

int64_t atenInt(const float &f);

int64_t atenInt(const at::IValue &scalar);

int64_t atenIntImplicit(const at::Tensor &a);

bool atenIs(const at::IValue &self, const at::IValue &other);

at::Tensor atenIsInf(const at::Tensor &self);

at::Scalar atenItem(const at::Tensor &self);

at::Tensor atenLayerNorm(const at::Tensor &input, at::IntArrayRef normalized_shape,
                         const c10::optional<at::Tensor> &weight = {}, const c10::optional<at::Tensor> &bias = {},
                         double eps = 1e-05, bool cudnn_enable = true);

std::tuple<at::Tensor, at::Tensor, at::Tensor> atenLayerNorm(const at::Tensor &input, at::IntArrayRef normalized_shape,
                                                             const c10::optional<at::Tensor> &weight,
                                                             const c10::optional<at::Tensor> &bias, double eps);

at::Tensor atenLeakyRelu(const at::Tensor &self, at::Scalar negative_slope);

at::Tensor atenLe(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenLe(const at::Tensor &self, at::Scalar other);

int64_t atenLen(const c10::List<at::IValue> &list);

int64_t atenLen(const at::Tensor &input);

at::Tensor atenLinear(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias = {});

c10::List<std::string> atenList(std::string &str);

c10::List<at::IValue> atenList(const c10::List<at::IValue> &list);

at::Tensor atenLog(const at::Tensor &self);

at::Tensor atenLogSoftmax(const at::Tensor &self, int64_t dim, c10::optional<at::ScalarType> dtype = c10::nullopt);

std::tuple<at::Tensor, at::Tensor, at::Tensor> atenLstm1(const at::Tensor &input, at::TensorList hx,
                                                         at::TensorList params, bool has_biases, int64_t num_layers,
                                                         double dropout, bool train, bool bidirectional,
                                                         bool batch_first);

std::tuple<at::Tensor, at::Tensor, at::Tensor> atenLstm2(const at::Tensor &data, const at::Tensor &batch_sizes,
                                                         at::TensorList hx, at::TensorList params, bool has_biases,
                                                         int64_t num_layers, double dropout, bool train,
                                                         bool bidirectional);

at::Tensor atenLt(const at::Tensor &self, const at::Scalar &other);

at::Tensor atenLt(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenMaskedFill(const at::Tensor &self, const at::Tensor &mask, at::Scalar value);

at::Tensor atenMaskedFill(const at::Tensor &self, const at::Tensor &mask, const at::Tensor &value);

at::Tensor atenMaskedSelect(const at::Tensor &self, const at::Tensor &mask);

at::Tensor atenMatmul(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenMax(const at::Tensor &self);

at::Tensor atenMax(const at::Tensor &self, const at::Tensor &other);

std::tuple<at::Tensor, at::Tensor> atenMax(const at::Tensor &self, int64_t dim, bool keepdim = false);

std::tuple<at::Tensor, at::Tensor> atenMax(const at::Tensor &self, at::Dimname dim, bool keepdim = false);

at::Tensor atenMaxPool2d(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                         at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);

at::Tensor atenMean(const at::Tensor &self, c10::optional<at::ScalarType> dtype = c10::nullopt);

at::Tensor atenMean(const at::Tensor &self, at::IntArrayRef dim, bool keepdim = false,
                    c10::optional<at::ScalarType> dtype = c10::nullopt);

at::Tensor atenMean(const at::Tensor &self, at::DimnameList dim, bool keepdim = false,
                    c10::optional<at::ScalarType> dtype = c10::nullopt);

at::Tensor atenMin(const at::Tensor &self);

at::Tensor atenMin(const at::Tensor &self, const at::Tensor &other);

std::tuple<at::Tensor, at::Tensor> atenMin(const at::Tensor &self, int64_t dim, bool keepdim);

int64_t atenMul(const int64_t &self, const int64_t &other);

double atenMul(const double &self, const double &other);

at::Tensor atenMul(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenMul(const at::Tensor &self, const at::Scalar &other);

at::Tensor atenNe(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenNe(const at::Tensor &self, const at::Scalar &other);

template <typename T>
bool atenNe(const c10::List<T> &a, const c10::List<T> &b)
{
    return a != b;
}

bool atenNe(const std::string a, const std::string b);

bool atenNe(const at::Scalar &self, const at::Scalar &other);

at::Tensor atenNeg(const at::Tensor &self);

at::Tensor atenNorm(const at::Tensor &self, at::Scalar p);

at::Tensor atenNot(const at::Tensor &self);

bool atenNot(const bool &input);

at::Tensor atenOneHot(const at::Tensor &self, int64_t num_classes = -1);

at::Tensor atenOnes(at::IntArrayRef size, const at::TensorOptions &options);

std::tuple<at::Tensor, at::Tensor> atenPackPaddedSequence(const at::Tensor &input, const at::Tensor &lengths,
                                                          bool batch_first);

std::tuple<at::Tensor, at::Tensor> atenPadPackedSequence(const at::Tensor &data, const at::Tensor &batch_sizes,
                                                         bool batch_first, at::Scalar padding_value,
                                                         int64_t total_length);

at::Tensor atenPermute(const at::Tensor &self, at::IntArrayRef dims);

at::Tensor atenPow(const at::Tensor &self, const at::Tensor &exponent);

at::Tensor atenPow(at::Scalar self, const at::Tensor &exponent);

at::Tensor atenPow(const at::Tensor &self, at::Scalar exponent);

at::Tensor atenRelu(const at::Tensor &self);

at::Tensor atenReshape(const at::Tensor &self, at::IntArrayRef shape);

at::Tensor atenRemainder(const at::Tensor &self, const at::Scalar &other);

at::Tensor atenRepeat(const at::Tensor &self, at::IntArrayRef repeats);

at::Tensor atenRsqrt(const at::Tensor &self);

at::Tensor atenSelect(const at::Tensor &self, at::Dimname dim, int64_t index);

at::Tensor atenSelect(const at::Tensor &self, int64_t dim, int64_t index);

template <typename T>
at::List<T> atenSetItem(at::List<T> list, int indice, T item)
{
    list[indice] = item;
    return list;
}

std::vector<int64_t> atenSize(const at::Tensor &tensor);

int64_t atenSize(const at::Tensor &tensor, int64_t dim);

int64_t atenSize(const at::Tensor &self, at::Dimname dim);

at::Tensor atenSlice(const at::Tensor &self, int64_t dim = 0, c10::optional<int64_t> start = c10::nullopt,
                     c10::optional<int64_t> end = c10::nullopt, int64_t step = 1);

at::Tensor atenSoftmax(const at::Tensor &self, int64_t dim, c10::optional<at::ScalarType> dtype = c10::nullopt);

at::Tensor atenSqueeze(const at::Tensor &self, int64_t dim);

at::Tensor atenSub(const at::Tensor &self, at::Scalar other, at::Scalar alpha = 1);

at::Tensor atenSub(const at::Tensor &self, const at::Tensor &other, at::Scalar alpha = 1);

at::Tensor atenSum(const at::Tensor &self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype);

at::Tensor atenTanh(const at::Tensor &self);

at::Tensor atenTensor(int64_t value);

at::Tensor atenTensor(float value);

at::Tensor atenTensor(at::ArrayRef<int64_t> array, const at::TensorOptions &options);

at::Tensor atenTensor(at::ArrayRef<double> array, const at::TensorOptions &options);

at::Tensor atenTo(const at::Tensor &self, c10::optional<at::ScalarType> dtype = {},
                  c10::optional<at::Layout> layout = {}, c10::optional<at::Device> device = {},
                  c10::optional<bool> pin_memory = {}, bool non_blocking = false, bool copy = false,
                  c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

at::Tensor atenTo(const at::Tensor &self, at::Device device, at::ScalarType dtype, bool non_blocking = false,
                  bool copy = false, c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

at::Tensor atenTo(const at::Tensor &self, at::ScalarType dtype, bool non_blocking = false, bool copy = false,
                  c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

at::Tensor atenTo(const at::Tensor &self, const at::Tensor &other, bool non_blocking = false, bool copy = false,
                  c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

std::tuple<at::Tensor, at::Tensor> atenTopk(const at::Tensor &self, int64_t k, int64_t dim, bool largest, bool sorted);

at::Tensor atenTranspose(const at::Tensor &self, int64_t dim0, int64_t dim1);

at::Tensor atenTranspose(const at::Tensor &self, at::Dimname dim0, at::Dimname dim1);

at::Tensor atenTriu(const at::Tensor &self, int64_t diagonal);

at::Tensor atenTypeAs(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenUnsqueeze(const at::Tensor &self, int64_t dim);

at::Tensor atenView(const at::Tensor &self, at::IntArrayRef size);

void atenWarn(const std::string &str);

at::Tensor atenWhere(const at::Tensor &condition, const at::Tensor &self, const at::Tensor &other);

at::Tensor atenWhere(const at::Tensor &condition, at::Scalar self, const at::Tensor &other);

at::Tensor atenWhere(const at::Tensor &condition, const at::Tensor &self, at::Scalar other);

at::Tensor atenWhere(const at::Tensor &condition, at::Scalar self, at::Scalar other);

std::vector<at::Tensor> atenWhere(const at::Tensor &condition);

at::Tensor atenZeros(at::IntArrayRef size, c10::optional<at::DimnameList> names, at::TensorOptions options = {});

at::Tensor atenZeros(at::IntArrayRef size, at::TensorOptions options = {});

at::Tensor atenZeroslike(const at::Tensor &self, at::TensorOptions options = {},
                         c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

void customAtenAddmm(std::string act_type, at::Tensor &self_tensor, at::Tensor &mat1_tensor, at::Tensor &mat2_tensor,
                     at::Scalar beta, at::Scalar alpha, torch::jit::IValue &output_iv, void* stream = nullptr);

void customAtenMatmul(at::Tensor &self_tensor, at::Tensor &other_tensor, torch::jit::IValue &output_iv, void* stream = nullptr);

}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler

#endif  // NNCOMPILER_ATEN_OP_H
