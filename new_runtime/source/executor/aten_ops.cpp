#include <iostream>

#include "c10/hip/HIPFunctions.h"
#include "new_runtime/include/executor/aten_ops.h"

namespace nn_compiler
{
namespace runtime
{
at::Tensor atenAdd(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha) { return at::add(self, other, alpha); }

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

bool atenAnd(bool& a, bool& b) { return a & b; }

at::Tensor atenAny(const at::Tensor &self) {
    return at::any(self);
}

void atenAppend(c10::List<at::IValue> &list, at::IValue el) { list.push_back(std::move(el)); }

at::Tensor atenArange1(at::Scalar end, const at::TensorOptions &options) {
    return at::arange(end, options);
}

at::Tensor atenArange2(at::Scalar start, at::Scalar end, const at::TensorOptions &options) {
    return at::arange(start, end, options);
}

at::Tensor atenArange3(at::Scalar start, at::Scalar end, at::Scalar step, const at::TensorOptions &options) {
    return at::arange(start, end, step, options);
}

// refer to castTensorTo() in torch/csrc/jit/runtime/register_special_ops.cpp
at::Tensor atenAsTensor(at::Tensor &self, at::ScalarType dtype, at::Device device) {
    return self.to(device, dtype);
}

at::Tensor atenBitwiseNot(const at::Tensor &self) { return at::bitwise_not(self); }

at::Tensor atenBmm(const at::Tensor &self, const at::Tensor &other) { return at::bmm(self, other); }

bool atenBool(const at::Tensor &self) {
    return self.is_nonzero();
}

bool atenBool(const int64_t& i) {
    return (bool)i;
}

bool atenBool(const double& d) {
    return (bool)d;
}

at::Tensor atenCat(at::TensorList tensors, int64_t dim) { return at::cat(tensors, dim); }

at::Tensor atenCat(at::TensorList tensors, at::Dimname dim) { return at::cat(tensors, dim); }

at::Tensor atenCeil(const at::Tensor &self) { return at::ceil(self); }

std::vector<at::Tensor> atenChunk(const at::Tensor &self, int chunks, int dim) {
    return at::chunk(self, chunks, dim);
}

at::Tensor atenClamp(const at::Tensor &self, const at::Scalar &min, const at::Scalar &max) {
    return at::clamp(self, min, max);
}

at::Tensor atenContiguous(const at::Tensor &self, at::MemoryFormat memory_format) {
    return at::native::contiguous(self, memory_format);
}

at::Tensor atenConv2d(const at::Tensor &input, const at::Tensor &weight,
                      const at::Tensor &bias, at::IntArrayRef stride,
                      at::IntArrayRef padding, at::IntArrayRef dilation, 
                      int64_t groups) {
    return at::conv2d(input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor &atenCopy_(at::Tensor &self, const at::Tensor &src, bool non_blocking)
{
    return at::native::copy_(self, src, non_blocking);
}

at::Tensor atenCpu(const at::Tensor &self) {
    return self.cpu();
}

at::Tensor atenCuda(const at::Tensor &self) {
    return self.cuda();
}

int64_t atenDeriveIndex(int64_t index, int64_t start, int64_t step) { return start + index * step; }

int64_t atenDim(const at::Tensor &tensor) { return tensor.dim(); }

at::Tensor atenDiv(const at::Tensor &self, const at::Tensor &other) { return at::div(self, other); }

at::Tensor atenDiv(const at::Tensor &self, const at::Scalar &other) { return at::div(self, other); }

at::Tensor atenDropout(const at::Tensor &input, double p, bool train) { return at::dropout(input, p, train); }

at::Tensor &atenDropout_(at::Tensor &self, double p, bool train) { return at::dropout_(self, p, train); }

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

at::Tensor &atenFill(at::Tensor &self, at::Scalar value) {
    return at::fill_(self, value);
}

at::Tensor &atenFill(at::Tensor &self, at::Tensor &other) {
    return at::fill_(self, other);
}

at::Tensor atenFloorDivide(const at::Tensor &self, at::Scalar value) {
    return at::floor_divide(self, value);
}

at::Tensor atenFloorDivide(const at::Tensor &self, const at::Tensor &other) {
    return at::floor_divide(self, other);
}

at::Tensor atenGather(const at::Tensor &self, int64_t dim, const at::Tensor &index, bool sparse_grad) {
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

at::Tensor atenIndex(const at::Tensor &self, const c10::List<c10::optional<at::Tensor>>& indices) {
    return at::index(self, indices);
}

at::Tensor &atenIndexPut(at::Tensor &self, const c10::List<c10::optional<at::Tensor>>& indices,
                         const at::Tensor &values, bool accumulate) {
    return at::index_put_(self, indices, values, accumulate);
}

at::Tensor atenIndexSelect(const at::Tensor &self, int64_t dim, const at::Tensor &index) {
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

bool atenIs(const at::IValue &self, const at::IValue &other) { return self.is(other); }

at::Scalar atenItem(const at::Tensor &self) { return at::native::item(self); }

at::Tensor atenLeakyRelu(const at::Tensor &self, at::Scalar negative_slope) {
    return at::leaky_relu(self, negative_slope);
}

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

at::Tensor atenLogSoftmax(const at::Tensor &self, int64_t dim, c10::optional<at::ScalarType> dtype) {
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
}  // namespace runtime
}  // namespace nn_compiler
