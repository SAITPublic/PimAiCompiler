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

#include <gtest/gtest.h>
#include <stdint.h>
#include <torch/script.h>

#include "executor/op_executor/prim_ops.h"
#include "glog/logging.h"
#include "ut_utils.h"
#include "utils/utils.h"

using std::vector;
using namespace nn_compiler::runtime;
using namespace nn_compiler::runtime::utils;
using namespace nn_compiler::runtime::op_executor;

TEST(NNCompilerUnitTest, primDeviceOpTest)
{
    vector<c10::DeviceType> devices{torch::kCUDA, torch::kCPU};
    int sz = 10;
    for (auto& dev : devices) {
        // create a tensor
        torch::Tensor input_tensor = torch::randn({sz, sz}, dev);
        EXPECT_TRUE(primDevice(input_tensor).type() == dev);
    }

    // test kHIP with ERROR
    // C++ exception with description "Could not run 'aten::empty.memory_format' with arguments from the 'HIP' backend.
}

TEST(NNCompilerUnitTest, primDtypeOpTest)
{
    // for kBool, kInt64: C++ exception with description ""normal_kernel_cpu/normal_kernel_cuda" not implemented for
    // 'Bool' and 'Long'
    vector<c10::ScalarType> types{torch::kHalf, torch::kFloat32};
    int sz = 10;
    for (auto& tp : types) {
        c10::TensorOptions opt;
        torch::Tensor input_tensor = torch::randn({sz, sz}, opt.device(c10::kCUDA).dtype(tp));
        EXPECT_TRUE(primDtype(input_tensor) == static_cast<int64_t>(input_tensor.scalar_type()));
    }
}

TEST(NNCompilerUnitTest, primTupleIndexOpTest)
{
    int sz = 20;
    std::vector<torch::Tensor> tuple_tensors = {torch::ones({sz, sz}), 2 * torch::ones({sz / 2, sz / 2}),
                                                torch::zeros({sz / 4, sz / 4})};

    std::vector<torch::IValue> vec;
    for (int i = 0; i < tuple_tensors.size(); i++) {
        auto a = tuple_tensors[i];
        auto iv = torch::jit::IValue(a);
        vec.push_back(iv);
    }

    for (int i = 0; i < tuple_tensors.size(); i++) {
        ASSERT_EQUAL(primTupleIndex(vec, i).toTensor(), vec.at(i).toTensor());
    }
}

TEST(NNCompilerUnitTest, primDataOpTest)
{
    vector<c10::ScalarType> types{torch::kHalf, torch::kFloat32};
    int sz = 10;
    for (auto& tp : types) {
        c10::TensorOptions opt;
        torch::Tensor input_tensor = torch::randn({sz, sz}, opt.device(c10::kCUDA).dtype(tp));
        ASSERT_EQUAL(primData(input_tensor), input_tensor);
    }
}

TEST(NNCompilerUnitTest, primTupleConstructOpTest)
{
    int sz = 10;
    vector<torch::IValue> ivs;

    // case 1: tensor, tensor --> (tensor, tensor)
    {
        vector<torch::Tensor> tensors = {torch::randn({sz, sz}), torch::randn({sz, sz})};
        // create a ivalue[] from
        ivs = {tensors[0], tensors[1]};
        primTupleConstruct(ivs, 2);
        // check the results
        auto elements = ivs[0].toTuple()->elements();
        for (int i = 0; i < elements.size(); i++) {
            ASSERT_EQUAL(elements[i].toTensor(), tensors[i]);
        }
    }

    // case 2: int, int, int --> (int, int, int)
    {
        int intvals[3] = {3, 2, 1};
        ivs.clear();
        ivs = {intvals[0], intvals[1], intvals[2]};
        primTupleConstruct(ivs, 3);
        auto elements = ivs[0].toTuple()->elements();
        for (int i = 0; i < elements.size(); i++) {
            EXPECT_TRUE(elements[i].isInt());
            EXPECT_TRUE(elements[i].toInt() == intvals[i]);
        }
    }

    // case 3: tensor, (tensor, tensor) --> (tensor, (tensor, tensor))
    {
        torch::Tensor item1 = torch::randn({sz, sz});
        vector<torch::Tensor> tensors = {torch::randn({sz, sz}), torch::randn({sz, sz})};
        auto item2 = c10::ivalue::Tuple::create(tensors[0], tensors[1]);
        ivs.clear();
        ivs = {item1, item2};
        primTupleConstruct(ivs, 2);
        auto elements = ivs[0].toTuple()->elements();

        // check item1
        EXPECT_TRUE(elements[0].isTensor());
        ASSERT_EQUAL(elements[0].toTensor(), item1);

        // check item2
        EXPECT_TRUE(elements[1].isTuple());
        auto tuple = elements[1].toTuple();
        for (int i = 0; i < tuple->elements().size(); i++) {
            ASSERT_EQUAL(tuple->elements().at(i).toTensor(), tensors[i]);
        }
    }
}

TEST(NNCompilerUnitTest, primTupleUnpackOpTest)
{
    int sz = 10;
    std::vector<torch::Tensor> tensors = {torch::randn({sz, sz}), torch::rand({sz / 2, sz / 2}),
                                          torch::rand({sz / 3, sz / 3})};
    auto tuple = c10::ivalue::Tuple::create(tensors[0], tensors[1], tensors[2]);

    auto ret = primTupleUnpack(tuple);
    for (int i = 0; i < ret.size(); i++) {
        EXPECT_TRUE(ret.at(i).isTensor());
        ASSERT_EQUAL(ret.at(i).toTensor(), tensors.at(i));
    }
}

TEST(NNCompilerUnitTest, primConstantOpTest)
{
    // case 1: scalar type
    {
        int64_t sint64 = 128;
        double sdouble = 128.5;
        EXPECT_TRUE(sint64 == primScalarConstant<int64_t>(&sint64));
        EXPECT_TRUE(sdouble == primScalarConstant<double>(&sdouble));
    }

    // case 2: Tensor type
    {
        int sz = 10;
        vector<c10::ScalarType> types{torch::kFloat32, torch::kHalf};
        for (auto& tp : types) {
            torch::Tensor tensor = torch::randn({sz, sz}, tp);
            std::vector<int64_t> shape;
            for (auto& item : tensor.sizes()) shape.push_back(item);
            if (tp == torch::kHalf) {
                ASSERT_EQUAL(createPtTensor(tensor.data_ptr(), shape, DataType::FLOAT16), tensor);
            } else if (tp == torch::kFloat32) {
                ASSERT_EQUAL(createPtTensor(tensor.data_ptr(), shape, DataType::FLOAT32), tensor);
            }
        }
    }

    // case 3: Str type
    {
        char* str = "PIMRuntime";
        EXPECT_TRUE(strcmp(primStrConstsnt(static_cast<void*>(str)).c_str(), str) == 0);
    }
}

TEST(NNCompilerUnitTest, primListConstructAndUnpackOpTest)
{
#define LIST_CONSTRUCT_TEST(name) \
    vector<torch::IValue> iv;     \
    iv.clear();                   \
    for (auto& item : vars) {     \
        iv.push_back({item});     \
    }                             \
    primListConstruct(iv, iv.size(), at::ListType::of##name##s());

    // case 1: int, int --> int[]
    {
        // Pack
        std::vector<int> vars = {1, 2, 3, 4};
        LIST_CONSTRUCT_TEST(Int);

        for (int i = 0; i < vars.size(); i++) {
            EXPECT_TRUE(iv[0].toIntList().get(i) == vars[i]);
        }

        // Unpack
        primListUnpack(iv, vars.size());
        for (int i = 0; i < iv.size(); i++) {
            EXPECT_TRUE(iv[i].toInt() == vars[i]);
        }
    }

    // case 2: tensor, tensor --> tensor[]
    {
        // Pack
        int sz = 20;
        std::vector<torch::Tensor> vars = {torch::randn({sz, sz}, torch::kHalf),
                                           torch::randn({sz / 2, sz / 2}, torch::kHalf),
                                           torch::randn({sz / 2, sz / 2}, torch::kHalf)};

        LIST_CONSTRUCT_TEST(Tensor);
        for (int i = 0; i < vars.size(); i++) {
            ASSERT_EQUAL(iv[0].toTensorList().get(i), vars[i]);
        }

        // Unpack
        primListUnpack(iv, vars.size());
        for (int i = 0; i < iv.size(); i++) {
            ASSERT_EQUAL(iv[i].toTensor(), vars[i]);
        }
    }
}

TEST(NNCompilerUnitTest, primUncheckedCastOpTest)
{
    int sz = 10;
    torch::Tensor tensor = torch::randn({sz, sz});
    ASSERT_EQUAL(tensor, primUncheckedCast(tensor));
}
