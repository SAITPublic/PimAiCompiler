/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <gtest/gtest.h>
#include "ATen/ATen.h"
#include "glog/logging.h"
// #include "aten_op.h"
// #include "hip/hip_runtime.h"
// #include "prim_op_utils"
// #include "executor/prim_ops.h"

using namespace torch;
using namespace at;
using namespace c10;
// using namespace nnr;

#define ASSERT_EQUAL(t1, t2) ASSERT_TRUE(t1.equal(t2));

// ------ ATen Op ------ //
TEST(NnrUnitTest, addTest)
{
    // result = tensor1 + tensor2 * alpha
    int shape = 4;
    float l_data[]{42, 42, 42, 42};
    float r_value = 2;
    float a_value = 4;
    Scalar r_s = r_value;
    Scalar alpha = a_value;

    auto l = from_blob(l_data, {shape}, kCPU);
    auto l_gpu = l.to(kCUDA);
    auto rt = add(l_gpu, r_s, alpha);
    rt.to(kCPU);

    float r_data[]{r_value, r_value, r_value, r_value};
    auto r = from_blob(r_data, {shape}, kCPU);
    auto r_gpu = r.to(kCUDA);
    auto rt_v = add(l_gpu, r_gpu, alpha);
    rt_v.to(kCPU);

    float tmp = r_value * a_value;
    float bench[]{l_data[0] + tmp, l_data[1] + tmp, l_data[2] + tmp, l_data[3] + tmp};
    int ret = 0;
    for (int i = 0; i < shape; ++i) {
        if (rt.data_ptr<float>()[i] != rt_v.data_ptr<float>()[i] || bench[i] != rt.data_ptr<float>()[i]) {
            ret = -1;
            break;
        }
    }
    EXPECT_TRUE(ret == 0);
}

TEST(NnrUnitTest, addmmTest)
{
    // result = tensor_c * beta  + (tensor_a * tensor_b) * alpha
    int shape_m = 4;
    int shape_n = 4;
    int shape_k = 4;
    float alpha_value = 1.0f;
    float beta_value = 2.0f;

    Scalar alpha = alpha_value;
    Scalar beta = beta_value;

    int result_dim = shape_m * shape_n;
    float tensorData[result_dim];
    float bench[result_dim];
    for (int i = 0; i < result_dim; ++i) {
        tensorData[i] = i;
        bench[i] = i;
    }

    auto tensor_left = from_blob(tensorData, {shape_m, shape_k}, kCPU);
    auto tensor_left_gpu = tensor_left.to(kCUDA);
    auto tensor_right = from_blob(tensorData, {shape_k, shape_n}, kCPU);
    auto tensor_right_gpu = tensor_right.to(kCUDA);
    auto result = from_blob(tensorData, {shape_m, shape_n}, kCPU);
    auto result_gpu = result.to(kCUDA);

    result_gpu = addmm(result_gpu, tensor_left_gpu, tensor_right_gpu, beta, alpha);
    result_gpu.to(kCPU);

    int ret = 0;
    for (int m = 0; m < shape_m; m++) {
        for (int n = 0; n < shape_n; n++) {
            float tmp = 0.f;
            for (int k = 0; k < shape_k; ++k) {
                tmp += tensorData[m * shape_k + k] * tensorData[k * shape_n + n];
            }
            bench[m * shape_n + n] = tmp * alpha_value + bench[m * shape_n + n] * beta_value;
        }
    }

    for (int i = 0; i < result_dim; ++i) {
        if (bench[i] != result_gpu.data_ptr<float>()[i]) {
            ret = -1;
            break;
        }
    }
    EXPECT_TRUE(ret == 0);
}

TEST(NnrUnitTest, catTest)
{
    // result = concat(split(result, dim(n/c/h/w)), dim(n/c/h/w))
    int shape_h = 3;
    int shape_w = 4;
    int dim = 1;
    auto t = randn({shape_h, shape_w}, kCUDA);
    auto t_list = split(t, 1, dim);
    auto r = cat(t_list, dim);
    ASSERT_EQUAL(t, r);
}

TEST(NnrUnitTest, ceilTest)
{
    // smallest integer value not less than arg.
    int shape_w = 5;
    auto t = randn({shape_w}, kCPU);
    auto t_gpu = t.to(kCUDA);
    auto t_ceil = ceil(t_gpu);
    auto t_cpu = ceil(t);
    ASSERT_EQUAL(t_ceil.to(kCPU), t_cpu);
}

TEST(NnrUnitTest, divTest)
{
    // result = tensor_a / tensor_b;
    int shape = 6;
    auto t = randn({shape}, kCPU);
    auto d = randn({shape}, kCPU);
    auto r = div(t, d);

    auto t_gpu = t.to(kCUDA);
    auto d_gpu = d.to(kCUDA);
    auto r_gpu = div(t_gpu, d_gpu);
    ASSERT_EQUAL(r_gpu.to(kCPU), r);

    Scalar d_s = 3.0f;
    auto r_s = div(t, d_s);
    auto r_s_gpu = div(t_gpu, d_s);
    DLOG(INFO) << t;
    DLOG(INFO) << d;
    DLOG(INFO) << r;
    ASSERT_EQUAL(r_gpu.to(kCPU), r);
}

TEST(NnrUnitTest, dropoutTest)
{
    // result = dropout(tensor, probability([0 ~ 1]);
    int shape = 6;
    auto t = randn({shape}, kCPU);
    auto t_gpu = t.to(kCUDA);
    double d_s = 0.25;

    bool train = false;
    auto r = dropout(t, d_s, train);
    auto r_gpu = dropout(t_gpu, d_s, train);
    DLOG(INFO) << t;
    DLOG(INFO) << d_s;
    DLOG(INFO) << r;
    ASSERT_EQUAL(r_gpu.to(kCPU), r);
}

TEST(NnrUnitTest, embeddingTest)
{
    // This module is often used to store word embeddings and retrieve them using indices. The input to the module is a
    // list of indices, and the output is the corresponding word embeddings.
    int shape_w = 3;
    int shape_h = 10;
    int64_t padding_idx = 9;
    bool scale_grad_by_freq = false;
    bool sparse = false;

    auto t = randn({shape_h, shape_w}, kCPU);
    DLOG(INFO) << t;
    auto t_gpu = t.to(kCUDA);

    int shape_indices = 4;
    int64_t tensorData[] = {1, 2, 6, 3};
    auto indices = from_blob(tensorData, {shape_indices}, TensorOptions().dtype(kLong).device(kCPU));
    DLOG(INFO) << indices;
    auto r = embedding(t, indices, padding_idx, scale_grad_by_freq, sparse);
    auto indices_gpu = indices.to(kCUDA);
    auto r_gpu = embedding(t_gpu, indices_gpu, padding_idx, scale_grad_by_freq, sparse);
    DLOG(INFO) << r;
    ASSERT_EQUAL(r_gpu.to(kCPU), r);
}

TEST(NnrUnitTest, lstmTest)
{
    // https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    int seq_len = 10;
    int64_t input_size = 4;
    int batch_size = 1;
    int hidden_size = 5;
    int num_layer = 1;
    bool batch_first = false;
    bool has_bias = false;
    bool train = false;
    bool bidirectional = false;
    double dropout = 0.0;

    auto input = randn({seq_len, batch_size, input_size}, kCPU);
    auto hx = randn({num_layer, batch_size, hidden_size}, kCPU);
    auto hc = randn({num_layer, batch_size, hidden_size}, kCPU);

    std::vector<Tensor> h_tmp;
    h_tmp.push_back(hx);
    h_tmp.push_back(hc);
    c10::ArrayRef<at::Tensor> h_tuple(h_tmp.data(), 2);

    auto param_i = randn({4 * hidden_size, input_size}, kCPU);
    auto param_h = randn({4 * hidden_size, hidden_size}, kCPU);
    std::vector<Tensor> param_tmp;
    param_tmp.push_back(param_i);
    param_tmp.push_back(param_h);
    c10::ArrayRef<at::Tensor> param_tuple(param_tmp.data(), 2);

    auto r = lstm(input, h_tuple, param_tuple, has_bias, num_layer, dropout, train, bidirectional, batch_first);

    // kCUDA
    auto input_gpu = input.to(kCUDA);
    auto hx_gpu = hx.to(kCUDA);
    auto hc_gpu = hc.to(kCUDA);

    std::vector<Tensor> h_tmp_gpu;
    h_tmp_gpu.push_back(hx_gpu);
    h_tmp_gpu.push_back(hc_gpu);
    c10::ArrayRef<at::Tensor> h_tuple_gpu(h_tmp_gpu.data(), 2);

    auto param_i_gpu = param_i.to(kCUDA);
    auto param_h_gpu = param_h.to(kCUDA);
    std::vector<Tensor> param_tmp_gpu;
    param_tmp_gpu.push_back(param_i_gpu);
    param_tmp_gpu.push_back(param_h_gpu);
    c10::ArrayRef<at::Tensor> param_tuple_gpu(param_tmp_gpu.data(), 2);

    auto r_gpu =
        lstm(input_gpu, h_tuple_gpu, param_tuple_gpu, has_bias, num_layer, dropout, train, bidirectional, batch_first);
    DLOG(INFO) << ((std::get<0>(r_gpu))[0]).to(kCPU);
    DLOG(INFO) << std::get<0>(r)[0];
}

TEST(NnrUnitTest, copyTest)
{
    bool non_blocking = false;
    int n = 100;
    auto t = randn({n}, kCPU);
    auto d = randn({n}, kCPU);
    auto g = randn({n}, kCUDA);
    auto r = randn({n}, kCUDA);

    // cpu to cpu
    at::native::copy_(d, t, non_blocking);
    ASSERT_EQUAL(d, t);

    // cpu to gpu
    at::native::copy_(g, d, non_blocking);
    ASSERT_EQUAL(g.to(kCPU), d);

    // gpu to gpu
    at::native::copy_(r, g, non_blocking);
    ASSERT_EQUAL(r, g);
}

TEST(NnrUnitTest, expandTest)
{
    // dim from {n} to {m, n}
    int n = 10;
    int m = 5;
    int dim = 2;
    int64_t s[] = {m, n};
    bool implicit = false;

    auto t = randn({n}, kCPU);
    auto t_gpu = t.to(kCUDA);
    ArrayRef<int64_t> s_v(s, dim);
    DLOG(INFO) << t;

    auto r = at::native::expand(t, s_v, implicit);
    auto r_gpu = at::native::expand(t_gpu, s_v, implicit);
    DLOG(INFO) << r;
    ASSERT_EQUAL(r_gpu.to(kCPU), r);
}

TEST(NnrUnitTest, itemTest)
{
    // from tensor to item
    int n = 1;
    EXPECT_TRUE(n == 1);

    auto t = randn({n}, kCPU);
    auto t_gpu = t.to(kCUDA);
    DLOG(INFO) << t;

    auto r = at::native::item(t);
    auto r_gpu = at::native::item(t_gpu);
    DLOG(INFO) << r;
    DLOG(INFO) << r_gpu;
    EXPECT_TRUE(r.toFloat() == r_gpu.toFloat());
}

TEST(NnrUnitTest, eqTest)
{
    // r = a == b ? 1 : 0;
    int shape = 4;
    float b = 42;
    float tensor1Data[]{b, b, b, b};

    auto tensor1 = from_blob(tensor1Data, {shape}, kCPU);
    auto tensor2 = from_blob(tensor1Data, {shape}, kCPU);
    auto b_scalar = b;

    auto r_s = eq(tensor1, b_scalar);
    auto r = eq(tensor1, tensor2);
    DLOG(INFO) << r_s;
    ASSERT_EQUAL(r_s, r);
}

TEST(NnrUnitTest, gtTest)
{
    // r = a > b ? 1 : 0;
    int shape = 4;
    float b = 42;
    float c = -2;
    float tensor1Data[]{b, b, b, b};
    float tensor2Data[]{b + c, b + c, b + c, b + c};

    auto tensor1 = from_blob(tensor1Data, {shape}, kCPU);
    auto tensor2 = from_blob(tensor2Data, {shape}, kCPU);
    auto b_scalar = b + c;

    auto r_s = gt(tensor1, b_scalar);
    auto r = gt(tensor1, tensor2);
    DLOG(INFO) << r_s;
    DLOG(INFO) << r;
    ASSERT_EQUAL(r_s, r);
}

TEST(NnrUnitTest, ltTest)
{
    // r = a < b ? 1 : 0;
    int shape = 4;
    float b = 42;
    float c = -2;
    float tensor1Data[]{b, b, b, b};
    float tensor2Data[]{b + c, b + c, b + c, b + c};

    auto tensor1 = from_blob(tensor1Data, {shape}, kCPU);
    auto tensor2 = from_blob(tensor2Data, {shape}, kCPU);
    auto b_scalar = b + c;

    auto r_s = lt(tensor1, b_scalar);
    auto r = lt(tensor1, tensor2);
    DLOG(INFO) << r_s;
    DLOG(INFO) << r;
    ASSERT_EQUAL(r_s, r);
}

TEST(NnrUnitTest, neTest)
{
    // r = a != b ? 1 : 0;
    int shape = 4;
    float b = 42;
    float c = 1;
    float tensor1Data[]{b, b, b, b};
    float tensor2Data[]{b + c, b + c, b + c, b + c};

    auto tensor1 = from_blob(tensor1Data, {shape}, kCPU);
    auto tensor2 = from_blob(tensor2Data, {shape}, kCPU);
    auto b_scalar = b + c;

    auto r_s = ne(tensor1, b_scalar);
    auto r = ne(tensor1, tensor2);
    DLOG(INFO) << r_s;
    DLOG(INFO) << r;
    ASSERT_EQUAL(r_s, r);
}

TEST(NnrUnitTest, negTest)
{
    // r = -a;
    int n = 10;
    auto t = randn({n}, kCPU);
    auto t_gpu = t.to(kCUDA);
    DLOG(INFO) << t;
    auto t_n = neg(t);
    auto t_n_gpu = neg(t_gpu);
    DLOG(INFO) << t_n;
    ASSERT_EQUAL(neg(neg(t)), t);
    ASSERT_EQUAL(t_n_gpu.to(kCPU), t_n);
}

TEST(NnrUnitTest, maxTest)
{
    // max(tensor) max value from tensor
    // r_i = t_i > d_i ? t_i : d_i;
    // max(tensor, dim, keepdim) = dim_max, index
    int n = 10;
    int m = 9;
    int64_t dim = 0;
    auto foo = Symbol::dimname("width");
    ASSERT_TRUE(Dimname::isValidName("width"));
    auto dimname = Dimname::fromSymbol(foo);
    DLOG(INFO) << dimname;
    bool keep_dim = true;

    auto t = randn({n}, kCPU);
    auto d = randn({n}, kCPU);
    auto t_d = randn({m, n}, kCPU);
    DLOG(INFO) << t;
    DLOG(INFO) << d;
    DLOG(INFO) << t_d;
    DLOG(INFO) << t;
    auto r = max(t);
    auto r_m = max(t, d);
    auto r_d = max(t_d, dim, keep_dim);
    DLOG(INFO) << r;
    DLOG(INFO) << r_m;
    DLOG(INFO) << std::get<0>(r_d);
    DLOG(INFO) << std::get<1>(r_d);
}

TEST(NnrUnitTest, reluTest)
{
    // r = relu(tensor)
    int n = 10;
    auto t = randn({n}, kCPU);
    auto t_gpu = t.to(kCUDA);
    auto r = relu(t);
    auto r_gpu = relu(t_gpu);
    ASSERT_EQUAL(r_gpu.to(kCPU), r);
}

TEST(NnrUnitTest, selectTest)
{
    int n = 10;
    int m = 9;
    int64_t dim = 0;
    int64_t index = 1;
    
    auto t = randn({m, n}, kCPU);
    auto t_gpu = t.to(kCUDA);
    DLOG(INFO) << t;
    auto r = select(t, dim, index);
    auto r_gpu = select(t_gpu, dim, index);
    DLOG(INFO) << r;
    ASSERT_EQUAL(r_gpu.to(kCPU), r);
}

TEST(NnrUnitTest, sizeTest)
{
    // size of tensor dim
    // dim 0 = 9, dim 1 = 10
    int n = 10;
    int m = 9;
    int64_t dim = 0;

    auto t = randn({m, n}, kCPU);
    auto t_gpu = t.to(kCUDA);
    int64_t r = size(t, dim);
    int64_t r_gpu = size(t_gpu, dim);
    DLOG(INFO) << r;
    ASSERT_TRUE(r == r_gpu);
}

TEST(NnrUnitTest, sliceTest)
{
    int n = 10;
    int m = 9;
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 6;
    int64_t step = 2;
    auto t = randn({m, n}, kCPU);
    auto t_gpu = t.to(kCUDA);
    DLOG(INFO) << t;
    auto r = slice(t, dim, start, end, step);
    auto r_gpu = slice(t_gpu, dim, start, end, step);
    DLOG(INFO) << r;
    ASSERT_EQUAL(r_gpu.to(kCPU), r);
}

TEST(NnrUnitTest, subTest)
{
    // result = tensor1 - tensor2 * alpha
    int n = 10;
    int m = 9;
    float alpha_value = 4;
    auto l = randn({m, n}, kCPU);
    auto l_gpu = l.to(kCUDA);
    auto r = randn({m, n}, kCPU);
    auto r_gpu = r.to(kCUDA);
    Scalar alpha = alpha_value;

    auto re = sub(l, r, alpha);
    auto re_gpu = sub(l_gpu, r_gpu, alpha);
    ASSERT_EQUAL(re_gpu.to(kCPU), re);
}

TEST(NnrUnitTest, tensorTest)
{
    // from value to tensor
    int64_t a = 8;
    float b = 10.f;
    auto r_a = tensor(a);
    DLOG(INFO) << r_a;
    auto r_b = tensor(b);
    DLOG(INFO) << r_b;
}

TEST(NnrUnitTest, toTest)
{
    // from tensor to another tensor
    int n = 10;
    int m = 9;
    auto t = randn({m, n}, kCPU);
    DLOG(INFO) << t;
    bool non_blocking = false;
    bool copy = false;
    ScalarType dtype(kFloat);
    c10::optional<MemoryFormat> optional_memory_format(MemoryFormat::Preserve);
    at::Device device(kCPU);
    auto r = at::native::to(t, device, dtype, non_blocking, copy, optional_memory_format);
    DLOG(INFO) << r;
    auto d = randn({m, n}, kCPU);
    auto rt = at::native::to(t, d, non_blocking, copy, optional_memory_format);
    DLOG(INFO) << d;
    DLOG(INFO) << rt;
    ASSERT_EQUAL(r, rt);
}

TEST(NnrUnitTest, transposeTest)
{
    // r = transpose(a)
    int n = 10;
    int m = 9;
    auto t = randn({m, n}, kCUDA);
    int64_t dim0 = 0;
    int64_t dim1 = 1;
    ASSERT_EQUAL(transpose(transpose(t, dim0, dim1), dim0, dim1), t);
}

TEST(NnrUnitTest, unsqueezeTest)
{
    // r = unsqueeze(a)
    int n = 10;
    int m = 9;
    int64_t dim = 1;
    auto t = randn({m, n}, kCUDA);
    DLOG(INFO) << t;
    auto r = unsqueeze(t, dim);
    DLOG(INFO) << r;
}

TEST(NnrUnitTest, zeroTest)
{
    // r = zero()
    int n = 10;
    int m = 9;
    int dim = 2;
    int64_t s[] = {m, n};
    ArrayRef<int64_t> s_v(s, dim);
    auto options = TensorOptions().dtype(kFloat).layout(kStrided).device(kCUDA).requires_grad(true);
    auto r = zeros(s_v, options);
    DLOG(INFO) << r;
}

TEST(NnrUnitTest, zeros_likeTest)
{
    // tensor = 0
    int n = 10;
    int m = 9;
    auto t = randn({m, n});
    DLOG(INFO) << t;

    auto options = TensorOptions().dtype(kFloat).layout(kStrided).device(kCPU);
    c10::optional<MemoryFormat> memory_format(MemoryFormat::Preserve);
    auto r = zeros_like(t, options, memory_format);
    auto r_gpu = r.to(kCPU);
    DLOG(INFO) << r;
    ASSERT_EQUAL(r_gpu.to(kCPU), r);
}

TEST(NnrUnitTest, matmulTest)
{
    // result = tensor_left * tensor_right
    int shape_m = 4;
    int shape_n = 4;
    int shape_k = 4;

    int result_dim = shape_m * shape_n;
    float tensorData[result_dim];
    for (int i = 0; i < result_dim; ++i) {
        tensorData[i] = i;
    }

    auto tensor_left = from_blob(tensorData, {shape_m, shape_k}, kCPU);
    auto tensor_left_gpu = tensor_left.to(kCUDA);
    auto tensor_right = from_blob(tensorData, {shape_k, shape_n}, kCPU);
    auto tensor_right_gpu = tensor_right.to(kCUDA);

    auto result = matmul(tensor_left_gpu, tensor_right_gpu);
    result.to(kCPU);

    int ret = 0;
    float bench[result_dim];
    for (int m = 0; m < shape_m; m++) {
        for (int n = 0; n < shape_n; n++) {
            float tmp = 0.f;
            for (int k = 0; k < shape_k; ++k) {
                tmp += tensorData[m * shape_k + k] * tensorData[k * shape_n + n];
            }
            bench[m * shape_n + n] = tmp;
        }
    }

    for (int i = 0; i < result_dim; ++i) {
        if (bench[i] != result.data_ptr<float>()[i]) {
            ret = -1;
            break;
        }
    }
    EXPECT_TRUE(ret == 0);
}

// ------ prim Op ------ //
