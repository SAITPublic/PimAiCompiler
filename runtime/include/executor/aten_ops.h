//
// Created by heguoqiang on 2021/6/23.
//

#ifndef NNCOMPILER_ATEN_OP_H
#define NNCOMPILER_ATEN_OP_H
#include "../nnrt_types.h"
#include "ATen/ATen.h"

namespace nnrt
{
__NNRT_API__ at::Tensor atenAdd(const at::Tensor &self, at::Scalar other, at::Scalar alpha = 1);
__NNRT_API__ at::Tensor atenAdd(const at::Tensor &self, const at::Tensor &other, at::Scalar alpha = 1);
__NNRT_API__ at::Tensor atenAddmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
                                  const at::Scalar &beta = 1, const at::Scalar &alpha = 1);
__NNRT_API__ at::Tensor atenCat(at::TensorList tensors, int64_t dim = 0);
__NNRT_API__ at::Tensor atenCat(at::TensorList tensors, at::Dimname dim);
__NNRT_API__ at::Tensor atenCeil(const at::Tensor &self);
__NNRT_API__ at::Tensor atenDiv(const at::Tensor &self, const at::Tensor &other);
__NNRT_API__ at::Tensor atenDiv(const at::Tensor &self, const at::Tensor &other,
                                c10::optional<c10::string_view> rounding_mode);
__NNRT_API__ at::Tensor atenDiv(const at::Tensor &self, const at::Scalar &other);
__NNRT_API__ at::Tensor atenDiv(const at::Tensor &self, const at::Scalar &other,
                                c10::optional<c10::string_view> rounding_mode);
__NNRT_API__ at::Tensor atenDropout(const at::Tensor &input, double p, bool train);
__NNRT_API__ at::Tensor &atenDropout_(at::Tensor &self, double p, bool train);
__NNRT_API__ at::Tensor atenEmbedding(const at::Tensor &weight, const at::Tensor &indices, int64_t padding_idx = -1,
                                      bool scale_grad_by_freq = false, bool sparse = false);
__NNRT_API__ at::Tensor atenEq(const at::Tensor &self, const at::Scalar &other);
__NNRT_API__ at::Tensor atenEq(const at::Tensor &self, const at::Tensor &other);
__NNRT_API__ at::Tensor atenGt(const at::Tensor &self, const at::Scalar &other);
__NNRT_API__ at::Tensor atenGt(const at::Tensor &self, const at::Tensor &other);
__NNRT_API__ std::tuple<at::Tensor, at::Tensor, at::Tensor> atenLstm(const at::Tensor &input, at::TensorList hx,
                                                                     at::TensorList params, bool has_biases,
                                                                     int64_t num_layers, double dropout, bool train,
                                                                     bool bidirectional, bool batch_first);
__NNRT_API__ std::tuple<at::Tensor, at::Tensor, at::Tensor> atenLstm(const at::Tensor &data,
                                                                     const at::Tensor &batch_sizes, at::TensorList hx,
                                                                     at::TensorList params, bool has_biases,
                                                                     int64_t num_layers, double dropout, bool train,
                                                                     bool bidirectional);
__NNRT_API__ at::Tensor atenLt(const at::Tensor &self, const at::Scalar &other);
__NNRT_API__ at::Tensor atenLt(const at::Tensor &self, const at::Tensor &other);
__NNRT_API__ at::Tensor atenMatmul(const at::Tensor &self, const at::Tensor &other);
}  // namespace nnrt

#endif  // NNCOMPILER_ATEN_OP_H
