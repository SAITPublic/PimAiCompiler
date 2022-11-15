/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include <iostream>

#include "c10/hip/HIPFunctions.h"
#include "executor/op_executor/aten_ops.h"
#include "executor/utils/utils.h"
#include "pim_runtime_api.h"

using namespace nn_compiler::runtime::utils;

namespace nn_compiler
{
namespace runtime
{
namespace op_executor
{

at::Tensor atenAbs(const at::Tensor &self) { return at::abs(self); }

at::Tensor atenAdd(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
{
    return at::add(self, other, alpha);
}

at::Tensor atenAdd(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
    return at::add(self, other, alpha);
}

int64_t atenAdd(int64_t &self, int64_t other, int64_t alpha) { return self + other * alpha; }

at::Tensor &atenAdd_(at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
{
    return at::native::add_(self, other, alpha);
}

at::Tensor atenAddmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2, const at::Scalar &beta,
                     const at::Scalar &alpha)
{
    return at::addmm(self, mat1, mat2, beta, alpha);
}

bool atenAnd(bool &a, bool &b) { return a & b; }

at::Tensor atenAny(const at::Tensor &self) { return at::any(self); }

void atenAppend(c10::List<at::IValue> &list, at::IValue el) { list.push_back(std::move(el)); }

at::Tensor atenArange1(at::Scalar end, const at::TensorOptions &options) { return at::arange(end, options).cuda(); }

at::Tensor atenArange2(at::Scalar start, at::Scalar end, const at::TensorOptions &options)
{
    return at::arange(start, end, options).cuda();
}

at::Tensor atenArange3(at::Scalar start, at::Scalar end, at::Scalar step, const at::TensorOptions &options)
{
    return at::arange(start, end, step, options).cuda();
}

at::Tensor atenArgmax(at::Tensor &self, c10::optional<int64_t> dim, bool keepdim)
{
    return at::argmax(self, dim, keepdim);
}

// refer to castTensorTo() in torch/csrc/jit/runtime/register_special_ops.cpp
at::Tensor atenAsTensor(at::Tensor &self, at::ScalarType dtype, at::Device device) { return self.to(device, dtype); }

at::Tensor atenBatchNorm2d(const at::Tensor &input, const c10::optional<at::Tensor> &weight,
                           const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &running_mean,
                           const c10::optional<at::Tensor> &running_var, bool training, double momentum, double eps,
                           bool cudnn_enabled)
{
    return at::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}

at::Tensor atenBitwiseNot(const at::Tensor &self) { return at::bitwise_not(self); }

at::Tensor atenBmm(const at::Tensor &self, const at::Tensor &other) { return at::bmm(self, other); }

bool atenBool(const at::Tensor &self) { return self.is_nonzero(); }

bool atenBool(const int64_t &i) { return (bool)i; }

bool atenBool(const double &d) { return (bool)d; }

at::Tensor atenCat(at::TensorList tensors, int64_t dim) { return at::cat(tensors, dim); }

at::Tensor atenCat(at::TensorList tensors, at::Dimname dim) { return at::cat(tensors, dim); }

at::Tensor atenCeil(const at::Tensor &self) { return at::ceil(self); }

std::vector<at::Tensor> atenChunk(const at::Tensor &self, int chunks, int dim) { return at::chunk(self, chunks, dim); }

at::Tensor atenClamp(const at::Tensor &self, const at::Scalar &min, const at::Scalar &max)
{
    return at::clamp(self, min, max);
}

at::Tensor atenClone(const at::Tensor &self, c10::optional<at::MemoryFormat> memory_format)
{
    return at::clone(self, memory_format);
}

at::Tensor atenContiguous(const at::Tensor &self, at::MemoryFormat memory_format)
{
    return at::native::contiguous(self, memory_format);
}

at::Tensor atenConv2d(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias, at::IntArrayRef stride,
                      at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups)
{
    return at::conv2d(input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor &atenCopy_(at::Tensor &self, const at::Tensor &src, bool non_blocking)
{
    return at::native::copy_(self, src, non_blocking);
}

at::Tensor atenCpu(const at::Tensor &self) { return self.cpu(); }

at::Tensor atenCuda(const at::Tensor &self) { return self.cuda(); }

at::Tensor atenCumsum(const at::Tensor &self, int64_t dim, c10::optional<at::ScalarType> dtype)
{
    return at::cumsum(self, dim, dtype);
}

at::Tensor atenCumsum(const at::Tensor &self, at::Dimname dim, c10::optional<at::ScalarType> dtype)
{
    return at::cumsum(self, dim, dtype);
}

int64_t atenDeriveIndex(int64_t index, int64_t start, int64_t step) { return start + index * step; }

at::Tensor atenDetach(const at::Tensor &self) { return at::detach(self); }

int64_t atenDim(const at::Tensor &tensor) { return tensor.dim(); }

at::Tensor atenDiv(const at::Tensor &self, const at::Tensor &other) { return at::div(self, other); }

at::Tensor atenDiv(const at::Tensor &self, const at::Scalar &other) { return at::div(self, other); }

at::Tensor atenDropout(const at::Tensor &input, double p, bool train) { return at::dropout(input, p, train); }

at::Tensor &atenDropout_(at::Tensor &self, double p, bool train) { return at::dropout_(self, p, train); }

at::Tensor atenEinsum(std::string equation, at::TensorList tensors) { return at::einsum(equation, tensors); }

at::Tensor atenEmbedding(const at::Tensor &weight, const at::Tensor &indices, int64_t padding_idx,
                         bool scale_grad_by_freq, bool sparse)
{
    return at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

at::Tensor atenEq(const at::Tensor &self, const at::Scalar &other) { return at::eq(self, other); }

bool atenEq(const at::Scalar &self, const at::Scalar &other)
{
    assert(self.type() == other.type());
    if (self.isIntegral(true)) {
        return (self.to<int>() == other.to<int>());
    } else if (self.isBoolean()) {
        return (self.to<bool>() == other.to<bool>());
    } else if (self.isFloatingPoint()) {
        return (self.to<double>() == other.to<double>());
    }
}

at::Tensor atenExpand(const at::Tensor &self, at::IntArrayRef size, bool implicit)
{
    return at::native::expand(self, size, implicit);
}

at::Tensor atenEq(const at::Tensor &self, const at::Tensor &other) { return at::eq(self, other); }

at::Tensor atenEq(const at::Tensor &self, const at::Scalar other) { return at::eq(self, other); }

bool atenEqual(const at::Tensor &self, const at::Tensor &other) { return at::equal(self, other); }

at::Tensor &atenFill(at::Tensor &self, at::Scalar value) { return at::fill_(self, value); }

at::Tensor &atenFill(at::Tensor &self, at::Tensor &other) { return at::fill_(self, other); }

at::Tensor atenFloorDivide(const at::Tensor &self, at::Scalar value) { return at::floor_divide(self, value); }

at::Tensor atenFloorDivide(const at::Tensor &self, const at::Tensor &other) { return at::floor_divide(self, other); }

at::Tensor atenFullLike(const at::Tensor &self, at::Scalar fill_value, const at::TensorOptions &options,
                        c10::optional<at::MemoryFormat> memory_format)
{
    return at::full_like(self, fill_value, options, memory_format);
}

at::Tensor atenGather(const at::Tensor &self, int64_t dim, const at::Tensor &index, bool sparse_grad)
{
    return at::gather(self, dim, index, sparse_grad);
}

at::Tensor atenGe(const at::Tensor &self, const at::Scalar &other) { return at::ge(self, other); }

at::Tensor atenGe(const at::Tensor &self, const at::Tensor &other) { return at::ge(self, other); }

static int64_t normalizeIndex(int64_t idx, int64_t list_size)
{
    if (idx < 0) {
        // Handle negative indexing
        idx = list_size + idx;
    }
    return idx;
}

at::IValue atenGetItem(const c10::List<at::IValue> &list, int idx)
{
    const int64_t list_size = list.size();
    const int64_t normalized_idx = normalizeIndex(idx, list_size);
    if (normalized_idx < 0 || normalized_idx >= list_size) {
        throw std::out_of_range("list index out of range");
    }
    return list.get(normalized_idx);
}

at::Tensor atenGt(const at::Tensor &self, const at::Scalar &other) { return at::gt(self, other); }

at::Tensor atenGt(const at::Tensor &self, const at::Tensor &other) { return at::gt(self, other); }

at::Tensor atenIndex(const at::Tensor &self, const c10::List<c10::optional<at::Tensor>> &indices)
{
    return at::index(self, indices);
}

at::Tensor &atenIndexPut(at::Tensor &self, const c10::List<c10::optional<at::Tensor>> &indices,
                         const at::Tensor &values, bool accumulate)
{
    return at::index_put_(self, indices, values, accumulate);
}

at::Tensor atenIndexSelect(const at::Tensor &self, int64_t dim, const at::Tensor &index)
{
    return at::index_select(self, dim, index);
}

int64_t atenInt(const at::Tensor &a) { return a.item<int64_t>(); }

int64_t atenInt(const bool &b) { return static_cast<int64_t>(b); }

int64_t atenInt(const float &f) { return static_cast<int64_t>(f); }

int64_t atenInt(const at::IValue &scalar)
{
    if (scalar.isInt()) {
        return scalar.toInt();
    } else {
        return static_cast<int64_t>(scalar.toScalar().toInt());
    }
}

int64_t atenIntImplicit(const at::Tensor &a) { return a.item<int64_t>(); }

bool atenIs(const at::IValue &self, const at::IValue &other) { return self.is(other); }

at::Tensor atenIsInf(const at::Tensor &self) { return at::isinf(self); }

at::Scalar atenItem(const at::Tensor &self) { return at::native::item(self); }

at::Tensor atenLayerNorm(const at::Tensor &input, at::IntArrayRef normalized_shape,
                         const c10::optional<at::Tensor> &weight, const c10::optional<at::Tensor> &bias, double eps,
                         bool cudnn_enable)
{
    return at::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> atenLayerNorm(const at::Tensor &input, at::IntArrayRef normalized_shape,
                                                             const c10::optional<at::Tensor> &weight,
                                                             const c10::optional<at::Tensor> &bias, double eps)
{
    return at::native_layer_norm(input, normalized_shape, weight, bias, eps);
}

at::Tensor atenLeakyRelu(const at::Tensor &self, at::Scalar negative_slope)
{
    return at::leaky_relu(self, negative_slope);
}

at::Tensor atenLe(const at::Tensor &self, const at::Tensor &other) { return at::le(self, other); }

at::Tensor atenLe(const at::Tensor &self, at::Scalar other) { return at::le(self, other); }

int64_t atenLen(const c10::List<at::IValue> &list) { return list.size(); }

int64_t atenLen(const at::Tensor &input) { return input.sizes()[0]; }

at::Tensor atenLinear(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias)
{
    return at::linear(input, weight, bias);
}

c10::List<std::string> atenList(std::string &str)
{
    c10::List<std::string> chars;
    chars.reserve(str.size());
    for (auto c : str) {
        chars.push_back(std::string(1, c));
    }
    return chars;
}

c10::List<at::IValue> atenList(const c10::List<at::IValue> &list) { return list.copy(); }

at::Tensor atenLog(const at::Tensor &self) { return at::log(self); }

at::Tensor atenLogSoftmax(const at::Tensor &self, int64_t dim, c10::optional<at::ScalarType> dtype)
{
    return at::log_softmax(self, dim, dtype);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> atenLstm1(const at::Tensor &input, at::TensorList hx,
                                                         at::TensorList params, bool has_biases, int64_t num_layers,
                                                         double dropout, bool train, bool bidirectional,
                                                         bool batch_first)
{
    dropout = 0.0f;
    return at::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> atenLstm2(const at::Tensor &data, const at::Tensor &batch_sizes,
                                                         at::TensorList hx, at::TensorList params, bool has_biases,
                                                         int64_t num_layers, double dropout, bool train,
                                                         bool bidirectional)
{
    dropout = 0.0f;
    return at::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

at::Tensor atenLt(const at::Tensor &self, const at::Scalar &other) { return at::lt(self, other); }

at::Tensor atenLt(const at::Tensor &self, const at::Tensor &other) { return at::lt(self, other); }

at::Tensor atenMaskedFill(const at::Tensor &self, const at::Tensor &mask, at::Scalar value)
{
    return at::masked_fill(self, mask, value);
}

at::Tensor atenMaskedFill(const at::Tensor &self, const at::Tensor &mask, const at::Tensor &value)
{
    return at::masked_fill(self, mask, value);
}

at::Tensor atenMaskedSelect(const at::Tensor &self, const at::Tensor &mask) { return at::masked_select(self, mask); }

at::Tensor atenMatmul(const at::Tensor &self, const at::Tensor &other) { return at::matmul(self, other); }

at::Tensor atenMax(const at::Tensor &self) { return at::max(self); }

at::Tensor atenMax(const at::Tensor &self, const at::Tensor &other) { return at::max(self, other); }

std::tuple<at::Tensor, at::Tensor> atenMax(const at::Tensor &self, int64_t dim, bool keepdim)
{
    return at::max(self, dim, keepdim);
}

std::tuple<at::Tensor, at::Tensor> atenMax(const at::Tensor &self, at::Dimname dim, bool keepdim)
{
    return at::max(self, dim, keepdim);
}

at::Tensor atenMaxPool2d(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                         at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode)
{
    return at::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor atenMean(const at::Tensor &self, c10::optional<at::ScalarType> dtype) { return at::mean(self, dtype); }

at::Tensor atenMean(const at::Tensor &self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype)
{
    return at::mean(self, dim, keepdim, dtype);
}

at::Tensor atenMean(const at::Tensor &self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype)
{
    return at::mean(self, dim, keepdim, dtype);
}

at::Tensor atenMin(const at::Tensor &self) { return at::min(self); }

at::Tensor atenMin(const at::Tensor &self, const at::Tensor &other) { return at::min(self, other); }

std::tuple<at::Tensor, at::Tensor> atenMin(const at::Tensor &self, int64_t dim, bool keepdim)
{
    return at::min(self, dim, keepdim);
}

int64_t atenMul(const int64_t &self, const int64_t &other) { return self * other; }

double atenMul(const double &self, const double &other) { return self * other; }

at::Tensor atenMul(const at::Tensor &self, const at::Tensor &other) { return at::mul(self, other); }

at::Tensor atenMul(const at::Tensor &self, const at::Scalar &other) { return at::mul(self, other); }

at::Tensor atenNe(const at::Tensor &self, const at::Tensor &other) { return at::ne(self, other); }

at::Tensor atenNe(const at::Tensor &self, const at::Scalar &other) { return at::ne(self, other); }

bool atenNe(const std::string a, const std::string b) { return a != b; }

bool atenNe(const at::Scalar &self, const at::Scalar &other)
{
    if (self.type() != other.type()) {
        return false;
    }
    if (self.isIntegral(true)) {
        return (self.to<int>() != other.to<int>());
    } else if (self.isBoolean()) {
        return (self.to<bool>() != other.to<bool>());
    } else if (self.isFloatingPoint()) {
        return (self.to<double>() != other.to<double>());
    }
}

at::Tensor atenNeg(const at::Tensor &self) { return at::neg(self); }

at::Tensor atenNorm(const at::Tensor &self, at::Scalar p) { return at::norm(self, p); }

at::Tensor atenNot(const at::Tensor &self) { return at::logical_not(self); }

bool atenNot(const bool &input) { return !input; };

at::Tensor atenOneHot(const at::Tensor &self, int64_t num_classes) { return at::one_hot(self, num_classes); }

at::Tensor atenOnes(at::IntArrayRef size, const at::TensorOptions &options) { return at::ones(size, options).cuda(); }

std::tuple<at::Tensor, at::Tensor> atenPackPaddedSequence(const at::Tensor &input, const at::Tensor &lengths,
                                                          bool batch_first)
{
    return at::_pack_padded_sequence(input, lengths, batch_first);
}

std::tuple<at::Tensor, at::Tensor> atenPadPackedSequence(const at::Tensor &data, const at::Tensor &batch_sizes,
                                                         bool batch_first, at::Scalar padding_value,
                                                         int64_t total_length)
{
    return at::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
}

at::Tensor atenPermute(const at::Tensor &self, at::IntArrayRef dims) { return at::native::permute(self, dims); }

at::Tensor atenPow(const at::Tensor &self, const at::Tensor &exponent) { return at::pow(self, exponent); }

at::Tensor atenPow(at::Scalar self, const at::Tensor &exponent) { return at::pow(self, exponent); }

at::Tensor atenPow(const at::Tensor &self, at::Scalar exponent) { return at::pow(self, exponent); }

at::Tensor atenRelu(const at::Tensor &self) { return at::relu(self); }

at::Tensor atenReshape(const at::Tensor &self, at::IntArrayRef shape) { return at::reshape(self, shape); }

at::Tensor atenRemainder(const at::Tensor &self, const at::Scalar &other) { return at::remainder(self, other); }

at::Tensor atenRepeat(const at::Tensor &self, at::IntArrayRef repeats) { return at::native::repeat(self, repeats); }

at::Tensor atenRsqrt(const at::Tensor &self) { return at::rsqrt(self); }

at::Tensor atenSelect(const at::Tensor &self, at::Dimname dim, int64_t index) { return at::select(self, dim, index); }

at::Tensor atenSelect(const at::Tensor &self, int64_t dim, int64_t index) { return at::select(self, dim, index); }

std::vector<int64_t> atenSize(const at::Tensor &tensor) { return tensor.sizes().vec(); }

int64_t atenSize(const at::Tensor &tensor, int64_t dim) { return at::size(tensor, dim); }

int64_t atenSize(const at::Tensor &self, at::Dimname dim) { return at::size(self, dim); }

at::Tensor atenSlice(const at::Tensor &self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end,
                     int64_t step)
{
    return at::slice(self, dim, start, end, step);
}

at::Tensor atenSoftmax(const at::Tensor &self, int64_t dim, c10::optional<at::ScalarType> dtype)
{
    return at::softmax(self, dim, dtype);
}

at::Tensor atenSqueeze(const at::Tensor &self, int64_t dim) { return at::squeeze(self, dim); }

at::Tensor atenSub(const at::Tensor &self, at::Scalar other, at::Scalar alpha) { return at::sub(self, other, alpha); }

at::Tensor atenSub(const at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
{
    return at::sub(self, other, alpha);
}

at::Tensor atenSum(const at::Tensor &self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype)
{
    return at::sum(self, dim, keepdim, dtype);
}

at::Tensor atenTanh(const at::Tensor &self) { return at::tanh(self); }

at::Tensor atenTensor(int64_t value) { return at::tensor(value); }

at::Tensor atenTensor(float value) { return at::tensor(value); }

at::Tensor atenTensor(at::ArrayRef<int64_t> array, const at::TensorOptions &options)
{
    return at::tensor(array, options);
}

at::Tensor atenTensor(at::ArrayRef<double> array, const at::TensorOptions &options)
{
    return at::tensor(array, options);
}

at::Tensor atenTo(const at::Tensor &self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                  c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, bool copy,
                  c10::optional<at::MemoryFormat> memory_format)
{
    return at::native::to(self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
}

at::Tensor atenTo(const at::Tensor &self, at::Device device, at::ScalarType dtype, bool non_blocking, bool copy,
                  c10::optional<at::MemoryFormat> memory_format)
{
    return at::native::to(self, device, dtype, non_blocking, copy, memory_format);
}

at::Tensor atenTo(const at::Tensor &self, at::ScalarType dtype, bool non_blocking, bool copy,
                  c10::optional<at::MemoryFormat> memory_format)
{
    return at::native::to(self, dtype, non_blocking, copy, memory_format);
}

at::Tensor atenTo(const at::Tensor &self, const at::Tensor &other, bool non_blocking, bool copy,
                  c10::optional<at::MemoryFormat> memory_format)
{
    return at::native::to(self, other, non_blocking, copy, memory_format);
}

std::tuple<at::Tensor, at::Tensor> atenTopk(const at::Tensor &self, int64_t k, int64_t dim, bool largest, bool sorted)
{
    return at::topk(self, k, dim, largest, sorted);
}

at::Tensor atenTranspose(const at::Tensor &self, int64_t dim0, int64_t dim1) { return at::transpose(self, dim0, dim1); }

at::Tensor atenTranspose(const at::Tensor &self, at::Dimname dim0, at::Dimname dim1)
{
    return at::transpose(self, dim0, dim1);
}

at::Tensor atenTriu(const at::Tensor &self, int64_t diagonal) { return at::triu(self, diagonal); }

at::Tensor atenTypeAs(const at::Tensor &self, const at::Tensor &other) { return at::native::type_as(self, other); }

at::Tensor atenUnsqueeze(const at::Tensor &self, int64_t dim) { return at::unsqueeze(self, dim); }

at::Tensor atenView(const at::Tensor &self, at::IntArrayRef size) { return at::native::view(self, size); }

void atenWarn(const std::string &str) { LOG(WARNING) << str; }

at::Tensor atenWhere(const at::Tensor &condition, const at::Tensor &self, const at::Tensor &other)
{
    return at::where(condition, self, other);
}

at::Tensor atenWhere(const at::Tensor &condition, at::Scalar self, const at::Tensor &other)
{
    return at::where(condition, self, other);
}

at::Tensor atenWhere(const at::Tensor &condition, const at::Tensor &self, at::Scalar other)
{
    return at::where(condition, self, other);
}

at::Tensor atenWhere(const at::Tensor &condition, at::Scalar self, at::Scalar other)
{
    return at::where(condition, self, other);
}

std::vector<at::Tensor> atenWhere(const at::Tensor &condition) { return at::where(condition); }

at::Tensor atenZeros(at::IntArrayRef size, c10::optional<at::DimnameList> names, at::TensorOptions options)
{
    return at::zeros(size, names, options);
}

at::Tensor atenZeros(at::IntArrayRef size, at::TensorOptions options) { return at::zeros(size, options); }

at::Tensor atenZeroslike(const at::Tensor &self, at::TensorOptions options,
                         c10::optional<at::MemoryFormat> memory_format)
{
    return at::zeros_like(self, options, memory_format);
}

void customAtenAddmm(std::string act_type, at::Tensor &self_tensor, at::Tensor &mat1_tensor, at::Tensor &mat2_tensor,
                     at::Scalar beta, at::Scalar alpha, torch::jit::IValue &output_iv, void *stream)
{
    int dim_i0 = mat1_tensor.dim();
    int dim_i1 = mat2_tensor.dim();
    int dim_self = self_tensor.dim();
    PimActFunc act = PimActFunc::NONE;

    auto mat1_shape = mat1_tensor.sizes().vec();
    auto mat2_shape = mat2_tensor.sizes().vec();
    gemmBroadcast(mat1_shape, mat2_shape);
    auto gemm_type = getGemmType(mat1_shape, mat2_shape);

    if (act_type == "aten::relu") {
        act = PimActFunc::ACT_RELU;
    }

    if (gemm_type == GEMM_TYPE::VM && dim_i0 > 1) {
        float alpha = 1.0f;
        float beta = 0.0f;

        if (!self_tensor.is_contiguous()) self_tensor = self_tensor.contiguous();
        if (!mat1_tensor.is_contiguous()) mat1_tensor = mat1_tensor.contiguous();
        if (!mat2_tensor.is_contiguous()) mat2_tensor = mat2_tensor.contiguous();

        int out_w = mat2_tensor.size(dim_i1 - 1);
        int in_w = mat2_tensor.size(dim_i1 - 2);

        _Float16 *b = (_Float16 *)self_tensor.data_ptr();
        _Float16 *x = (_Float16 *)mat1_tensor.data_ptr();
        _Float16 *A = (_Float16 *)mat2_tensor.data_ptr();
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto output_shape = mat1_tensor.sizes().vec();
        output_shape[dim_i0 - 1] = out_w;
        output_shape[dim_i0 - 2] = 1;
        auto output = at::zeros(output_shape, options);
        _Float16 *y = (_Float16 *)output.data_ptr();

        PimGemmDesc *pim_desc = PimCreateGemmDesc(1, 1, 1, in_w, 1, out_w, PIM_FP16);
        PimBo *dev_op0 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_INPUT, x);
        PimBo *dev_op1 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT, A);
        PimBo *dev_op2 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_BIAS, b);
        PimBo *dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT, y);

        PimExecuteGemm(dev_out, dev_op0, dev_op1, dev_op2, act);

        PimDestroyBo(dev_op0);
        PimDestroyBo(dev_op1);
        PimDestroyBo(dev_op2);
        PimDestroyBo(dev_out);
        PimDestroyGemmDesc(pim_desc);

        output_iv = tensorToIValue(output);
    } else if (gemm_type == GEMM_TYPE::MV && dim_i1 > 1) {
        if (!self_tensor.is_contiguous()) self_tensor = self_tensor.contiguous();
        if (!mat1_tensor.is_contiguous()) mat1_tensor = mat1_tensor.contiguous();
        if (!mat2_tensor.is_contiguous()) mat2_tensor = mat2_tensor.contiguous();

        int out_w = mat1_tensor.size(dim_i0 - 2);
        int in_w = mat1_tensor.size(dim_i0 - 1);

        _Float16 *b = (_Float16 *)self_tensor.data_ptr();
        _Float16 *A = (_Float16 *)mat1_tensor.data_ptr();
        _Float16 *x = (_Float16 *)mat2_tensor.data_ptr();
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto output_shape = mat2_tensor.sizes().vec();
        output_shape[dim_i1 - 1] = 1;
        output_shape[dim_i1 - 2] = out_w;
        auto output = at::zeros(output_shape, options);
        _Float16 *y = (_Float16 *)output.data_ptr();

        PimGemmDesc *pim_desc = PimCreateGemmDesc(1, 1, in_w, 1, out_w, 1, PIM_FP16, W_X_I);
        PimBo *dev_op0 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_INPUT, x);
        PimBo *dev_op1 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT, A);
        PimBo *dev_op2 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_BIAS, b);
        PimBo *dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT, y);

        PimExecuteGemm(dev_out, dev_op0, dev_op1, dev_op2, act, W_X_I);

        PimDestroyBo(dev_op0);
        PimDestroyBo(dev_op1);
        PimDestroyBo(dev_op2);
        PimDestroyBo(dev_out);
        PimDestroyGemmDesc(pim_desc);

        output_iv = tensorToIValue(output);
    } else if (dim_self == 1) {
        self_tensor = self_tensor.unsqueeze(1);
        self_tensor = self_tensor.repeat({1, mat2_tensor.size(dim_i1 - 1)});

        if (dim_i1 != 2) {
            mat2_tensor = mat2_tensor.squeeze(0);
        }

        auto output = atenAddmm(self_tensor, mat1_tensor, mat2_tensor, beta, alpha);
        if (act == PimActFunc::ACT_RELU) {
            output = atenRelu(output);
        }
        // update output
        for (int i = 0; i < dim_i1 - 2; ++i) {
            output = output.unsqueeze(0);
        }
        output_iv = tensorToIValue(output);
    } else {
        // the dims of mat1_shape and mat2_shape are 4
        int num = mat2_shape[0] * mat2_shape[1] * mat2_shape[3];
        const char *env_k = std::getenv("ENABLE_PIM");
        if (num > DIM_OUT_PIM && env_k != nullptr && *env_k == 1) {
            float alpha = 1.0f;
            float beta = 0.0f;

            if (!self_tensor.is_contiguous()) self_tensor = self_tensor.contiguous();
            if (!mat1_tensor.is_contiguous()) mat1_tensor = mat1_tensor.contiguous();
            if (!mat2_tensor.is_contiguous()) mat2_tensor = mat2_tensor.contiguous();

            int n = mat1_shape[0] > mat2_shape[0] ? mat1_shape[0] : mat2_shape[0];
            int c = mat1_shape[1] > mat2_shape[1] ? mat1_shape[1] : mat2_shape[1];
            int h = mat1_shape[2];
            int in_w = mat2_shape[2];
            int out_w = mat2_shape[3];

            _Float16 *b = (_Float16 *)self_tensor.data_ptr();
            _Float16 *x = (_Float16 *)mat1_tensor.data_ptr();
            _Float16 *A = (_Float16 *)mat2_tensor.data_ptr();
            auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
            auto output = at::zeros({n, c, h, out_w}, options);
            _Float16 *y = (_Float16 *)output.data_ptr();

            PimGemmDesc *pim_desc = PimCreateGemmDesc(n, c, h, in_w, h, out_w, PIM_FP16);
            PimBo *dev_op0 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_INPUT, x);
            PimBo *dev_op1 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT, A);
            PimBo *dev_op2 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_BIAS, b);
            PimBo *dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT, y);

            PimExecuteGemm(dev_out, dev_op0, dev_op1, dev_op2, act);

            PimDestroyBo(dev_op0);
            PimDestroyBo(dev_op1);
            PimDestroyBo(dev_op2);
            PimDestroyBo(dev_out);
            PimDestroyGemmDesc(pim_desc);
        } else {
            auto output = atenAddmm(self_tensor, mat1_tensor, mat2_tensor, beta, alpha);
            if (act == PimActFunc::ACT_RELU) {
                output = atenRelu(output);
            }
            output_iv = tensorToIValue(output);
        }
    }
}

void customAtenMatmul(at::Tensor &self_tensor, at::Tensor &other_tensor, torch::jit::IValue &output_iv, void *stream)
{
    int dim_i0 = self_tensor.dim();
    int dim_i1 = other_tensor.dim();

    auto self_shape = self_tensor.sizes().vec();
    auto other_shape = other_tensor.sizes().vec();
    gemmBroadcast(self_shape, other_shape);
    auto gemm_type = getGemmType(self_shape, other_shape);

    if (gemm_type == GEMM_TYPE::VM) {
        if (!self_tensor.is_contiguous()) self_tensor = self_tensor.contiguous();
        if (!other_tensor.is_contiguous()) other_tensor = other_tensor.contiguous();

        int out_w = other_tensor.size(dim_i1 - 1);
        int in_w = other_tensor.size(dim_i1 - 2);

        _Float16 *x = (_Float16 *)self_tensor.data_ptr();
        _Float16 *A = (_Float16 *)other_tensor.data_ptr();
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto output_shape = self_tensor.sizes().vec();
        if (dim_i1 > dim_i0) {
            output_shape = other_tensor.sizes().vec();
            output_shape[dim_i1 - 1] = out_w;
            output_shape[dim_i1 - 2] = 1;
        } else {
            output_shape[dim_i0 - 1] = out_w;
            output_shape[dim_i0 - 2] = 1;
        }
        auto output = at::zeros(output_shape, options);
        _Float16 *y = (_Float16 *)output.data_ptr();

        PimGemmDesc *pim_desc = PimCreateGemmDesc(1, 1, 1, in_w, 1, out_w, PIM_FP16);
        PimBo *dev_op0 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_INPUT, x);
        PimBo *dev_op1 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT, A);
        PimBo *dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT, y);

        PimExecuteGemm(dev_out, dev_op0, dev_op1, nullptr);

        PimDestroyBo(dev_op0);
        PimDestroyBo(dev_op1);
        PimDestroyBo(dev_out);
        PimDestroyGemmDesc(pim_desc);

        output_iv = tensorToIValue(output);
    } else if (gemm_type == GEMM_TYPE::MV) {
        if (!self_tensor.is_contiguous()) self_tensor = self_tensor.contiguous();
        if (!other_tensor.is_contiguous()) other_tensor = other_tensor.contiguous();

        int out_w = self_tensor.size(dim_i0 - 2);
        int in_w = self_tensor.size(dim_i0 - 1);

        _Float16 *A = (_Float16 *)self_tensor.data_ptr();
        _Float16 *x = (_Float16 *)other_tensor.data_ptr();
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto output_shape = other_tensor.sizes().vec();

        if (dim_i0 > dim_i1) {
            output_shape = self_tensor.sizes().vec();
            output_shape[dim_i0 - 2] = out_w;
            output_shape.pop_back();
        } else {
            output_shape[dim_i1 - 1] = 1;
            output_shape[dim_i1 - 2] = out_w;
        }
        auto output = at::zeros(output_shape, options);
        _Float16 *y = (_Float16 *)output.data_ptr();

        PimGemmDesc *pim_desc = PimCreateGemmDesc(1, 1, in_w, 1, out_w, 1, PIM_FP16, W_X_I);
        PimBo *dev_op0 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_INPUT, x);
        PimBo *dev_op1 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT, A);
        PimBo *dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT, y);

        PimExecuteGemm(dev_out, dev_op0, dev_op1, nullptr, PimActFunc::NONE, W_X_I);

        PimDestroyBo(dev_op0);
        PimDestroyBo(dev_op1);
        PimDestroyBo(dev_out);
        PimDestroyGemmDesc(pim_desc);

        output_iv = tensorToIValue(output);
    } else if (gemm_type == GEMM_TYPE::VV) {
        if (!self_tensor.is_contiguous()) self_tensor = self_tensor.contiguous();
        if (!other_tensor.is_contiguous()) other_tensor = other_tensor.contiguous();
        int out_w = 1;
        int in_w = 0;
        for (int i = 0; i < dim_i0; ++i) {
            if (self_tensor.size(i) != 1) {
                in_w = self_tensor.size(i);
                break;
            }
        }

        _Float16 *A = (_Float16 *)self_tensor.data_ptr();
        _Float16 *x = (_Float16 *)other_tensor.data_ptr();
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto output_shape = other_tensor.sizes().vec();

        if (dim_i0 > dim_i1) {
            output_shape = self_tensor.sizes().vec();
            output_shape[dim_i0 - 1] = 1;
            output_shape[dim_i0 - 2] = out_w;
        } else {
            output_shape[dim_i1 - 1] = 1;
            output_shape[dim_i1 - 2] = out_w;
        }

        auto output = at::zeros(output_shape, options);
        _Float16 *y = (_Float16 *)output.data_ptr();

        PimGemmDesc *pim_desc = PimCreateGemmDesc(1, 1, 1, in_w, 1, 1, PIM_FP16);
        PimBo *dev_op0 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_INPUT, x);
        PimBo *dev_op1 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT, A);
        PimBo *dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT, y);

        PimExecuteGemm(dev_out, dev_op0, dev_op1, nullptr, PimActFunc::NONE, I_X_W, stream);

        PimDestroyBo(dev_op0);
        PimDestroyBo(dev_op1);
        PimDestroyBo(dev_out);
        PimDestroyGemmDesc(pim_desc);

        output_iv = tensorToIValue(output);
    } else {
        // the dims of self_shape and other_shape are 4
        int num = other_shape[0] * other_shape[1] * other_shape[3];
        const char *env_k = std::getenv("ENABLE_PIM");
        if (num > DIM_OUT_PIM && env_k != nullptr && *env_k == 1) {
            if (!self_tensor.is_contiguous()) self_tensor = self_tensor.contiguous();
            if (!other_tensor.is_contiguous()) other_tensor = other_tensor.contiguous();
            int n = self_shape[0] > other_shape[0] ? self_shape[0] : other_shape[0];
            int c = self_shape[1] > other_shape[1] ? self_shape[1] : other_shape[1];
            int h = self_shape[2];
            int in_w = other_shape[2];
            int out_w = other_shape[3];

            if (c != self_shape[1]) {
                self_tensor.repeat({1, c, 1, 1});
            } else if (c != other_shape[1]) {
                other_tensor.repeat({1, c, 1, 1});
            }

            if (n != self_shape[0]) {
                self_tensor.repeat({n, 1, 1, 1});
            } else if (n != other_shape[0]) {
                other_tensor.repeat({n, 1, 1, 1});
            }

            _Float16 *x = (_Float16 *)self_tensor.data_ptr();
            _Float16 *A = (_Float16 *)other_tensor.data_ptr();
            auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
            auto output = at::zeros({n, c, h, out_w}, options);
            _Float16 *y = (_Float16 *)output.data_ptr();

            PimGemmDesc *pim_desc = PimCreateGemmDesc(n, c, h, in_w, h, out_w, PIM_FP16);
            PimBo *dev_op0 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_INPUT, x);
            PimBo *dev_op1 = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT, A);
            PimBo *dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT, y);

            PimExecuteGemm(dev_out, dev_op0, dev_op1, nullptr);

            PimDestroyBo(dev_op0);
            PimDestroyBo(dev_op1);
            PimDestroyBo(dev_out);
            PimDestroyGemmDesc(pim_desc);

            output_iv = tensorToIValue(output);
        } else {
            auto output = atenMatmul(self_tensor, other_tensor);
            output_iv = tensorToIValue(output);
        }
    }
}

}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler
