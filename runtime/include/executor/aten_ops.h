//
// Created by heguoqiang on 2021/6/23.
//

#ifndef NNCOMPILER_ATEN_OP_H
#define NNCOMPILER_ATEN_OP_H
#include "ATen/ATen.h"

namespace nnrt
{
at::Tensor atenAdd(const at::Tensor &self, at::Scalar other, at::Scalar alpha = 1);

at::Tensor atenAdd(const at::Tensor &self, const at::Tensor &other, at::Scalar alpha = 1);

at::Tensor &atenAdd_(at::Tensor &self, const at::Tensor &other, at::Scalar alpha = 1);

at::Tensor &atenAdd_(at::Tensor &self, at::Scalar other, at::Scalar alpha = 1);

at::Tensor atenAddmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,

                     const at::Scalar &beta = 1, const at::Scalar &alpha = 1);

at::Tensor atenCat(at::TensorList tensors, int64_t dim = 0);

at::Tensor atenCat(at::TensorList tensors, at::Dimname dim);

at::Tensor atenCeil(const at::Tensor &self);

at::Tensor &atenCopy_(at::Tensor &self, const at::Tensor &src, bool non_blocking = false);

at::Tensor atenDiv(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenDiv(const at::Tensor &self, const at::Tensor &other,

                   c10::optional<c10::string_view> rounding_mode);

at::Tensor atenDiv(const at::Tensor &self, const at::Scalar &other);

at::Tensor atenDiv(const at::Tensor &self, const at::Scalar &other,

                   c10::optional<c10::string_view> rounding_mode);

at::Tensor atenDropout(const at::Tensor &input, double p, bool train);

at::Tensor &atenDropout_(at::Tensor &self, double p, bool train);

at::Tensor atenEmbedding(const at::Tensor &weight, const at::Tensor &indices, int64_t padding_idx = -1,

                         bool scale_grad_by_freq = false, bool sparse = false);

at::Tensor atenEq(const at::Tensor &self, const at::Scalar &other);

at::Tensor atenEq(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenExpand(const at::Tensor &self, at::IntArrayRef size, bool implicit = false);

at::Tensor atenGt(const at::Tensor &self, const at::Scalar &other);

at::Tensor atenGt(const at::Tensor &self, const at::Tensor &other);

at::Scalar atenItem(const at::Tensor &self);

std::tuple<at::Tensor, at::Tensor, at::Tensor> atenLstm(const at::Tensor &input, at::TensorList hx,

                                                        at::TensorList params, bool has_biases,

                                                        int64_t num_layers, double dropout, bool train,

                                                        bool bidirectional, bool batch_first);

std::tuple<at::Tensor, at::Tensor, at::Tensor> atenLstm(const at::Tensor &data,

                                                        const at::Tensor &batch_sizes, at::TensorList hx,

                                                        at::TensorList params, bool has_biases,

                                                        int64_t num_layers, double dropout, bool train,

                                                        bool bidirectional);

at::Tensor atenLt(const at::Tensor &self, const at::Scalar &other);

at::Tensor atenLt(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenMatmul(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenMax(const at::Tensor &self);

at::Tensor atenMax(const at::Tensor &self, const at::Tensor &other);

std::tuple<at::Tensor, at::Tensor> atenMax(const at::Tensor &self, int64_t dim,

                                           bool keepdim = false);

std::tuple<at::Tensor, at::Tensor> atenMax(const at::Tensor &self, at::Dimname dim,

                                           bool keepdim = false);

at::Tensor atenNe(const at::Tensor &self, const at::Tensor &other);

at::Tensor atenNe(const at::Tensor &self, const at::Scalar &other);

at::Tensor atenNeg(const at::Tensor &self);

at::Tensor atenRelu(const at::Tensor &self);

at::Tensor atenSelect(const at::Tensor &self, at::Dimname dim, int64_t index);

at::Tensor atenSelect(const at::Tensor &self, int64_t dim, int64_t index);

int64_t atenSize(const at::Tensor &tensor, int64_t dim);

at::Tensor atenSlice(const at::Tensor &self, int64_t dim = 0, int64_t start = 0,

                     int64_t end = 9223372036854775807, int64_t step = 1);

at::Tensor atenSub(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha = 1);

at::Tensor atenTensor(int64_t value);
at::Tensor atenTensor(float value);
at::Tensor atenTo(const at::Tensor &self, const at::TensorOptions &options = {}, bool non_blocking = false,
                  bool copy = false, c10::optional<at::MemoryFormat> memory_format = c10::nullopt);
at::Tensor atenTo(const at::Tensor &self, at::Device device, at::ScalarType dtype, bool non_blocking = false,
                  bool copy = false, c10::optional<at::MemoryFormat> memory_format = c10::nullopt);
at::Tensor atenTo(const at::Tensor &self, at::ScalarType dtype, bool non_blocking = false, bool copy = false,
                  c10::optional<at::MemoryFormat> memory_format = c10::nullopt);
at::Tensor atenTo(const at::Tensor &self, const at::Tensor &other, bool non_blocking = false, bool copy = false,
                  c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

at::Tensor atenTranspose(const at::Tensor &self, int64_t dim0, int64_t dim1);

at::Tensor atenTranspose(const at::Tensor &self, at::Dimname dim0, at::Dimname dim1);

at::Tensor atenUnsqueeze(const at::Tensor &self, int64_t dim);

at::Tensor atenZeros(at::IntArrayRef size, c10::optional<at::DimnameList> names, at::TensorOptions options = {});

at::Tensor atenZeros(at::IntArrayRef size, at::TensorOptions options = {});

at::Tensor atenZeroslike(const at::Tensor &self, at::TensorOptions options = {},
                         c10::optional<at::MemoryFormat> memory_format = c10::nullopt);
}  // namespace nnrt
#endif  // NNCOMPILER_ATEN_OP_H
