//
// Created by heguoqiang on 2021/6/23.
//
#include "executor/aten_op.h"

namespace nnrt
{
at::Tensor atAdd(const at::Tensor &self, at::Scalar other, at::Scalar alpha) { return at::add(self, other, alpha); }

at::Tensor atenAdd(const at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
{
    return at::add(self, other, alpha);
}

at::Tensor atenAddmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2, const at::Scalar &beta,
                     const at::Scalar &alpha)
{
    return at::addmm(self, mat1, mat2, beta, alpha);
}

at::Tensor atenCat(at::TensorList tensors, int64_t dim) { return at::cat(tensors, dim); }

at::Tensor atenCat(at::TensorList tensors, at::Dimname dim) { return at::cat(tensors, dim); }
at::Tensor atenCeil(const at::Tensor &self) { return at::ceil(self); }

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

at::Tensor atenEq(const at::Tensor &self, const at::Tensor &other) { return at::eq(self, other); }

at::Tensor atenGt(const at::Tensor &self, const at::Scalar &other) { return at::gt(self, other); }

at::Tensor atenGt(const at::Tensor &self, const at::Tensor &other) { return at::gt(self, other); }

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
}  // namespace nnrt
