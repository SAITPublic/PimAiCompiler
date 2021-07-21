//
// Created by heguoqiang on 2021/6/23.
//
#include "executor/aten_ops.h"

namespace nnrt
{
int64_t atenDeriveIndex(int64_t start, int64_t index, int64_t step) { return start + index * step; }

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

bool atenIs(const at::IValue &self, const at::IValue &other) { return self.is(other); }

void atenAppend(c10::List<at::IValue> &list, at::IValue el) { list.push_back(std::move(el)); }

int64_t atenDim(const at::Tensor &tensor) { return tensor.dim(); }

int64_t atenLen(const c10::List<at::IValue> &list) { return list.size(); }

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

at::Tensor atenAdd(const at::Tensor &self, at::Scalar other, at::Scalar alpha) { return at::add(self, other, alpha); }

at::Tensor atenAdd(const at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
{
    return at::add(self, other, alpha);
}

at::Tensor &atenAdd_(at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
{
    return at::native::add_(self, other, alpha);
}

at::Tensor &atenAdd_(at::Tensor &self, at::Scalar other, at::Scalar alpha)
{
    return at::native::add_(self, other, alpha);
}

at::Tensor atenAddmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2, const at::Scalar &beta,
                     const at::Scalar &alpha)
{
    return at::addmm(self, mat1, mat2, beta, alpha);
}

at::Tensor atenCat(at::TensorList tensors, int64_t dim) { return at::cat(tensors, dim); }

at::Tensor atenCat(at::TensorList tensors, at::Dimname dim) { return at::cat(tensors, dim); }

at::Tensor atenCeil(const at::Tensor &self) { return at::ceil(self); }

at::Tensor &atenCopy_(at::Tensor &self, const at::Tensor &src, bool non_blocking)
{
    return at::native::copy_(self, src, non_blocking);
}

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

at::Tensor atenGt(const at::Tensor &self, const at::Scalar &other) { return at::gt(self, other); }

at::Tensor atenGt(const at::Tensor &self, const at::Tensor &other) { return at::gt(self, other); }

at::Scalar atenItem(const at::Tensor &self) { return at::native::item(self); }

std::tuple<at::Tensor, at::Tensor, at::Tensor> atenLstm(const at::Tensor &input, at::TensorList hx,
                                                        at::TensorList params, bool has_biases, int64_t num_layers,
                                                        double dropout, bool train, bool bidirectional,
                                                        bool batch_first)
{
    return at::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> atenlstm(const at::Tensor &data, const at::Tensor &batch_sizes,
                                                        at::TensorList hx, at::TensorList params, bool has_biases,
                                                        int64_t num_layers, double dropout, bool train,
                                                        bool bidirectional)
{
    return at::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

at::Tensor atenLt(const at::Tensor &self, const at::Scalar &other) { return at::lt(self, other); }

at::Tensor atenLt(const at::Tensor &self, const at::Tensor &other) { return at::lt(self, other); }

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

at::Tensor atenNe(const at::Tensor &self, const at::Tensor &other) { return at::ne(self, other); }

at::Tensor atenNe(const at::Tensor &self, const at::Scalar &other) { return at::ne(self, other); }

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

at::Tensor atenRelu(const at::Tensor &self) { return at::relu(self); }

at::Tensor atenSelect(const at::Tensor &self, at::Dimname dim, int64_t index) { return at::select(self, dim, index); }

at::Tensor atenSelect(const at::Tensor &self, int64_t dim, int64_t index) { return at::select(self, dim, index); }

std::vector<int64_t> atenSize(const at::Tensor &tensor) { return tensor.sizes().vec(); }

int64_t atenSize(const at::Tensor &tensor, int64_t dim) { return at::size(tensor, dim); }

int64_t atenSize(const at::Tensor &self, at::Dimname dim) { return at::size(self, dim); }

at::Tensor atenSlice(const at::Tensor &self, int64_t dim, int64_t start, int64_t end, int64_t step)
{
    return at::slice(self, dim, start, end, step);
}

at::Tensor atenSub(const at::Tensor &self, at::Scalar other, at::Scalar alpha) { return at::sub(self, other, alpha); }

at::Tensor atenSub(const at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
{
    return at::sub(self, other, alpha);
}

at::Tensor atenTensor(int64_t value) { return at::tensor(value); }

at::Tensor atenTensor(float value) { return at::tensor(value); }

at::Tensor atenTensor(at::ArrayRef<int64_t> array, const at::TensorOptions &options) {
    return at::tensor(array, options);
}

at::Tensor atenTensor(at::ArrayRef<double> array, const at::TensorOptions &options) {
    return at::tensor(array, options);
}

at::Tensor atenTo(const at::Tensor &self, const at::TensorOptions &options, bool non_blocking, bool copy,
                  c10::optional<at::MemoryFormat> memory_format)
{
    return at::native::to(self, options, non_blocking, copy, memory_format);
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

at::Tensor atenTranspose(const at::Tensor &self, int64_t dim0, int64_t dim1) { return at::transpose(self, dim0, dim1); }

at::Tensor atenTranspose(const at::Tensor &self, at::Dimname dim0, at::Dimname dim1)
{
    return at::transpose(self, dim0, dim1);
}

at::Tensor atenUnsqueeze(const at::Tensor &self, int64_t dim) { return at::unsqueeze(self, dim); }

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
}  // namespace nnrt
