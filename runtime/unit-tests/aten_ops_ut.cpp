/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <gtest/gtest.h>
#include <torch/script.h>

#include "executor/op_executor/aten_ops.h"
#include "glog/logging.h"

using namespace torch;
using namespace at;
using namespace c10;
using namespace nn_compiler::runtime::op_executor;

#define ASSERT_EQUAL(t1, t2) ASSERT_TRUE(t1.equal(t2));

// ------ ATen Op ------ //
// only the ops implemented by ourselves need to test, and ops implemented by pytorch do not need to test.

TEST(NNCompilerUnitTest, atenCatOpTest)
{
    // result = concat(split(result, dim(n/c/h/w)), dim(n/c/h/w))
    int shape_h = 3;
    int shape_w = 4;
    int dim = 1;
    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
    auto t = at::randn({shape_h, shape_w}, options);
    auto t_list = split(t, 1, dim);
    auto r = atenCat(t_list, dim);
    ASSERT_EQUAL(t, r);
}

TEST(NNCompilerUnitTest, atenCopyOpTest)
{
    bool non_blocking = false;
    int n = 100;
    auto t = at::randn({n}, kCPU);
    auto d = at::randn({n}, kCPU);
    auto g = at::randn({n}, kCUDA);
    auto r = at::randn({n}, kCUDA);

    // cpu to cpu
    atenCopy_(d, t, non_blocking);
    ASSERT_EQUAL(d, t);

    // cpu to gpu
    atenCopy_(g, d, non_blocking);
    ASSERT_EQUAL(g.to(kCPU), d);

    // gpu to gpu
    atenCopy_(r, g, non_blocking);
    ASSERT_EQUAL(r, g);
}

TEST(NNCompilerUnitTest, atenDeriveOpTest)
{
    // r = start + index * step
    int64_t start = 10;
    int64_t index = 9;
    int64_t step = 8;
    int64_t t = start + index * step;
    int64_t r = atenDeriveIndex(index, start, step);
    EXPECT_TRUE(t == r);
}

TEST(NNCompilerUnitTest, atenGetItemOpTest)
{
#define LIST_CONSTRUCT_TEST(name)  \
    std::vector<torch::IValue> iv; \
    iv.clear();                    \
    for (auto& item : vars) {      \
        iv.push_back({item});      \
    }                              \
    at::ListTypePtr type = at::ListType::of##name##s();

    std::vector<int> vars = {1, 2, 3, 4};
    LIST_CONSTRUCT_TEST(Int);

    c10::List<torch::jit::IValue> vals(type->getElementType());
    vals.reserve(iv.size());
    for (size_t i = 0; i < iv.size(); ++i) {
        vals.emplace_back(std::move(iv[i]));
    }
    for (int i = 0; i < iv.size(); i++) {
        auto r = atenGetItem(vals, i);
        EXPECT_TRUE(r.toInt() == vars[i]);
    }
}

TEST(NNCompilerUnitTest, atenAppendAndLenOpTest)
{
#define LIST_CONSTRUCT_TEST(name)  \
    std::vector<torch::IValue> iv; \
    iv.clear();                    \
    for (auto& item : vars) {      \
        iv.push_back({item});      \
    }                              \
    at::ListTypePtr type = at::ListType::of##name##s();

    std::vector<int> vars = {1, 2, 3, 4};
    LIST_CONSTRUCT_TEST(Int);

    c10::List<torch::jit::IValue> vals(type->getElementType());
    vals.reserve(iv.size());
    for (size_t i = iv.size() - iv.size(); i < iv.size(); ++i) {
        vals.emplace_back(std::move(iv[i]));
    }
    int t_i = 5;
    torch::IValue tmp(t_i);
    atenAppend(vals, tmp);
    auto r = atenGetItem(vals, 4);
    EXPECT_TRUE(r.toInt() == t_i);

    int64_t rt = atenLen(vals);
    EXPECT_TRUE(rt == 5);
}

TEST(NNCompilerUnitTest, atenDimOpTest)
{
    int n = 10;
    int m = 9;
    auto t = at::randn({m, n});
    int64_t r = atenDim(t);
    EXPECT_TRUE(r == 2);
}

TEST(NNCompilerUnitTest, atenIntOpTest)
{
    int n = 1;
    int m = 1;
    auto t = at::randn({m, n});
    int64_t r = atenInt(t);
    EXPECT_TRUE(r == t.item<int64_t>());
    bool i = false;
    r = atenInt(i);
    EXPECT_TRUE(r == 0);
    float f = 2.f;
    r = atenInt(f);
    EXPECT_TRUE(r == 2);
    at::IValue scalar(n);
    r = atenInt(scalar);
    EXPECT_TRUE(r == n);
}

TEST(NNCompilerUnitTest, atenFormatOpTest)
{
    std::string t("{}abc{}");
    std::string d("d");
    std::string a("__");
    auto r = atenFormat(t, d, a);
    EXPECT_TRUE(r == "dabc__");
}

TEST(NNCompilerUnitTest, atenListOpTest)
{
    std::string t("abc");
    std::string data[3] = {"a", "b", "c"};
    auto r = atenList(t);
    int result = 0;
    for (auto idx = 0; idx < r.size(); idx++) {
        if (r.get(idx) != data[idx]) {
            LOG(INFO) << r.get(idx);
            LOG(INFO) << data[idx];
            result++;
            break;
        }
    }
    EXPECT_TRUE(result == 0);
}
