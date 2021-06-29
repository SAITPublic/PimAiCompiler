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
    float tensor1Data[]{42, 0, 0, 0};
    float tensor2_value = 2;
    float alpha_value = 4;

    auto tensor1 = from_blob(tensor1Data, {shape}, kCPU);
    auto tensor1_gpu = tensor1.to(kCUDA);

    Scalar scalar_cpu = tensor2_value;
    Scalar alpha = alpha_value;

    auto result_tensor_scalar = add(tensor1_gpu, scalar_cpu, alpha);
    result_tensor_scalar.to(kCPU);

    float tensor2Data[]{tensor2_value, tensor2_value, tensor2_value, tensor2_value};
    auto tensor2 = from_blob(tensor2Data, {shape}, kCPU);
    auto tensor2_gpu = tensor2.to(kCUDA);
    auto result_tensor_tensor = add(tensor1_gpu, tensor2_gpu, alpha);
    result_tensor_tensor.to(kCPU);

    float bench[]{tensor1Data[0] + tensor2_value * alpha_value, tensor1Data[1] + tensor2_value * alpha_value,
                  tensor1Data[2] + tensor2_value * alpha_value, tensor1Data[3] + tensor2_value * alpha_value};
    int ret = 0;
    for (int i = 0; i < shape; ++i) {
        if (result_tensor_scalar.data_ptr<float>()[i] != result_tensor_tensor.data_ptr<float>()[i] ||
            bench[i] != result_tensor_scalar.data_ptr<float>()[i]) {
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
    // std::cout << t << std::endl;
    // std::cout << d << std::endl;
    // std::cout << r << std::endl;
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
    // std::cout << t << std::endl;
    // std::cout << d_s << std::endl;
    // std::cout << r << std::endl;
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
    // std::cout << t << std::endl;
    auto t_gpu = t.to(kCUDA);

    int shape_indices = 4;
    int64_t tensorData[] = {1, 2, 6, 3};
    auto indices = from_blob(tensorData, {shape_indices}, TensorOptions().dtype(kLong).device(kCPU));
    // std::cout << indices << std::endl;
    auto r = embedding(t, indices, padding_idx, scale_grad_by_freq, sparse);
    auto indices_gpu = indices.to(kCUDA);
    auto r_gpu = embedding(t_gpu, indices_gpu, padding_idx, scale_grad_by_freq, sparse);
    // std::cout <<  r << std::endl;
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

    auto r_gpu =  lstm(input_gpu, h_tuple_gpu, param_tuple_gpu, has_bias, num_layer, dropout, train, bidirectional, batch_first);
    // ASSERT_EQUAL(((std::get<0>(r_gpu))[0]).to(kCPU), std::get<0>(r)[0]);
    std::cout << ((std::get<0>(r_gpu))[0]).to(kCPU) << std::endl;
    std::cout << std::get<0>(r)[0] << std::endl;
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
    // std::cout << t << std::endl;

    auto r = at::native::expand(t, s_v, implicit);
    auto r_gpu = at::native::expand(t_gpu, s_v, implicit);
    // std::cout << r << std::endl;
    ASSERT_EQUAL(r_gpu.to(kCPU), r);
}

TEST(NnrUnitTest, itemTest)
{
    // from tensor to item
    int n = 1;
    EXPECT_TRUE(n == 1);

    auto t = randn({n}, kCPU);
    auto t_gpu = t.to(kCUDA);
    // std::cout << t << std::endl;

    auto r = at::native::item(t);
    auto r_gpu = at::native::item(t_gpu);
    //  std::cout << r << std::endl;
    // std::cout << r_gpu << std::endl;
    // ASSERT_EQUAL(r_gpu, r);
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
    // std::cout << r_s << std::endl;
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
    // std::cout << r_s << std::endl;
    // std::cout << r << std::endl;
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
    // std::cout << r_s << std::endl;
    // std::cout << r << std::endl;
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
    // std::cout << r_s << std::endl;
    // std::cout << r << std::endl;
    ASSERT_EQUAL(r_s, r);
}

TEST(NnrUnitTest, negTest)
{
    // r = -a;
    int n = 10;
    auto t = randn({n}, kCPU);
    auto t_gpu = t.to(kCUDA);
    // std::cout << t << std::endl;
    auto t_n = neg(t);
    auto t_n_gpu = neg(t_gpu);
    // std::cout << t_n << std::endl;
    ASSERT_EQUAL(neg(neg(t)), t);
    ASSERT_EQUAL(t_n_gpu.to(kCPU), t_n);
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
