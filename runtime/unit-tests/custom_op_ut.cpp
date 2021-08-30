#include <gtest/gtest.h>
#include "executor/custom_ops.hpp"
#include "ut_utils.h"


TEST(NnrUnitTest, customGemv)
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
