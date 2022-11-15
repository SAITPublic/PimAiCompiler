/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include <gtest/gtest.h>

#include "executor/op_executor/custom_ops.h"
#include "pim_runtime_api.h"
#include "ut_utils.h"

using namespace nn_compiler::runtime::op_executor;

TEST(NNCompilerUnitTest, customGemvTest)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);

    {
        torch::Tensor self = torch::rand({1, 1, 1, 1024}, options);
        torch::Tensor other = torch::rand({1024, 4096}, options);

        int dim_i0 = self.dim();
        int dim_i1 = other.dim();

        int m = 1;
        int n = other.size(dim_i1 - 1);
        int k = other.size(dim_i1 - 2);

        _Float16* x = (_Float16*)self.data_ptr();
        _Float16* A = (_Float16*)other.data_ptr();

        auto output = at::zeros({1, 1, 1, n}, options);
        _Float16* y = (_Float16*)output.data_ptr();
        rocblas_gemv_template_xAy(nullptr, x, A, y, m, n, k, alpha, beta);

        auto output_torch = at::matmul(self, other);
        ASSERT_ALLCLOSE_TOLERANCES(output.cpu(), output_torch.cpu(), 1e-3, 1e-5);
    }

    {
        torch::Tensor self = torch::rand({4096, 1024}, options);
        torch::Tensor other = torch::rand({1, 1, 1024, 1}, options);

        int dim_i0 = self.dim();
        int dim_i1 = other.dim();

        int m = self.size(dim_i0 - 2);
        int n = 1;
        int k = self.size(dim_i0 - 1);

        _Float16* A = (_Float16*)self.data_ptr();
        _Float16* x = (_Float16*)other.data_ptr();
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto output = at::zeros({1, 1, m, 1}, options);
        _Float16* y = (_Float16*)output.data_ptr();
        rocblas_gemv_template_Axy(nullptr, A, x, y, m, n, k, alpha, beta);

        auto output_torch = at::matmul(self, other);
        ASSERT_ALLCLOSE_TOLERANCES(output.cpu(), output_torch.cpu(), 1e-3, 1e-5);
    }
}

TEST(NNCompilerUnitTest, pimCustomGemvTest)
{
    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);

    {
        torch::Tensor self = torch::rand({1, 1, 1, 320}, options);
        torch::Tensor other = torch::rand({320, 1280}, options);

        int dim_i0 = self.dim();
        int dim_i1 = other.dim();

        int n = 1, c = 1, h = 1;
        int in_w = other.size(dim_i1 - 2);
        int out_w = other.size(dim_i1 - 1);

        _Float16* x = (_Float16*)self.data_ptr();
        _Float16* A = (_Float16*)other.data_ptr();

        auto output = at::zeros({n, c, 1, out_w}, options);
        _Float16* y = (_Float16*)output.data_ptr();

        PimInitialize(RT_TYPE_HIP, PIM_FP16);

        PimGemmDesc* pim_desc = PimCreateGemmDesc(n, c, h, in_w, h, out_w, PIM_FP16);
        PimBo* dev_in = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_INPUT, x);
        PimBo* dev_wei = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT, A);
        PimBo* dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT, y);

        PimExecuteGemm(dev_out, dev_in, dev_wei, nullptr);

        PimDestroyBo(dev_in);
        PimDestroyBo(dev_wei);
        PimDestroyBo(dev_out);
        PimDestroyGemmDesc(pim_desc);

        PimDeinitialize();

        auto output_torch = at::matmul(self, other);
        ASSERT_ALLCLOSE_TOLERANCES(output.cpu(), output_torch.cpu(), 1e-3, 1e-5);
    }

    {
        torch::Tensor self = torch::rand({1280, 320}, options);
        torch::Tensor other = torch::rand({1, 1, 320, 1}, options);

        int dim_i0 = self.dim();
        int dim_i1 = other.dim();

        int out_w = self.size(dim_i0 - 2);
        int in_w = self.size(dim_i0 - 1);
        int n = 1, c = 1, h = 1;

        _Float16* A = (_Float16*)self.data_ptr();
        _Float16* x = (_Float16*)other.data_ptr();

        auto output = at::zeros({n, c, out_w, 1}, options);
        _Float16* y = (_Float16*)output.data_ptr();

        PimInitialize(RT_TYPE_HIP, PIM_FP16);

        PimGemmDesc* pim_desc = PimCreateGemmDesc(n, c, in_w, h, out_w, h, PIM_FP16, W_X_I);
        PimBo* dev_in = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_INPUT, x);
        PimBo* dev_wei = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_WEIGHT, A);
        PimBo* dev_out = PimCreateBo(pim_desc, MEM_TYPE_DEVICE, GEMM_OUTPUT, y);

        PimExecuteGemm(dev_out, dev_in, dev_wei, nullptr, PimActFunc::NONE, W_X_I);

        PimDestroyBo(dev_in);
        PimDestroyBo(dev_wei);
        PimDestroyBo(dev_out);
        PimDestroyGemmDesc(pim_desc);

        PimDeinitialize();

        auto output_torch = at::matmul(self, other);
        ASSERT_ALLCLOSE_TOLERANCES(output.cpu(), output_torch.cpu(), 1e-3, 1e-5);
    }
}
