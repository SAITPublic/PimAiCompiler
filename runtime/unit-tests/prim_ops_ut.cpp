#include <vector>
#include <gtest/gtest.h>
#include <torch/script.h>
#include "executor/prim_ops.h"
#include "ut_utils.h"

using std::vector;

TEST(NnrUnitTest, primDeviceTest)
{
    vector<c10::DeviceType> devices{torch::kCUDA, torch::kCPU};
    int sz = 10;
    for (auto& dev : devices) {
        // create a tensor
        torch::Tensor input_tensor = torch::randn({sz, sz}, dev);
        EXPECT_TRUE(nnrt::primDevice(input_tensor).type() == dev);
    }

    // test kHIP with ERROR
    // C++ exception with description "Could not run 'aten::empty.memory_format' with arguments from the 'HIP' backend.
}

TEST(NnrUnitTest, primDtypeTest)
{
    // for kBool, kInt64: C++ exception with description ""normal_kernel_cpu/normal_kernel_cuda" not implemented for
    // 'Bool' and 'Long'
    vector<c10::ScalarType> types{torch::kHalf, torch::kFloat32};
    int sz = 10;
    for (auto& tp : types) {
        c10::TensorOptions opt;
        torch::Tensor input_tensor = torch::randn({sz, sz}, opt.device(c10::kCUDA).dtype(tp));
        EXPECT_TRUE(nnrt::primDtype(input_tensor) == static_cast<int64_t>(input_tensor.scalar_type()));
    }
}

TEST(NnrUnitTest, primTupleIndexTest)
{
    int sz = 20;
    std::vector<torch::Tensor> tuple_tensors = {torch::ones({sz, sz}), 2 * torch::ones({sz / 2, sz / 2}),
                                                torch::zeros({sz / 4, sz / 4})};

    for (int i = 0; i < tuple_tensors.size(); i++) {
        ASSERT_EQUAL(nnrt::primTupleIndex(tuple_tensors, i), tuple_tensors.at(i));
    }
}

TEST(NnrUnitTest, primData)
{
    vector<c10::ScalarType> types{torch::kHalf, torch::kFloat32};
    int sz = 10;
    for (auto& tp : types) {
        c10::TensorOptions opt;
        torch::Tensor input_tensor = torch::randn({sz, sz}, opt.device(c10::kCUDA).dtype(tp));
        ASSERT_EQUAL(nnrt::primData(input_tensor), input_tensor);
    }
}

TEST(NnrUnitTest, primTupleConstruct)
{
    int sz = 10;
    vector<torch::IValue> ivs;

    // case 1: tensor, tensor --> (tensor, tensor)
    {
        vector<torch::Tensor> tensors = {torch::randn({sz, sz}), torch::randn({sz, sz})};
        // create a ivalue[] from
        ivs = {tensors[0], tensors[1]};
        nnrt::primTupleConstruct(ivs, 2);
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
        nnrt::primTupleConstruct(ivs, 3);
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
        nnrt::primTupleConstruct(ivs, 2);
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

TEST(NnrUnitTest, primTupleUnpack)
{
    int sz = 10;
    std::vector<torch::Tensor> tensors = {torch::randn({sz, sz}), torch::rand({sz / 2, sz / 2}),
                                          torch::rand({sz / 3, sz / 3})};
    auto tuple = c10::ivalue::Tuple::create(tensors[0], tensors[1], tensors[2]);

    auto ret = nnrt::primTupleUnpack(tuple);
    for (int i = 0; i < ret.size(); i++) {
        EXPECT_TRUE(ret.at(i).isTensor());
        ASSERT_EQUAL(ret.at(i).toTensor(), tensors.at(i));
    }
}
