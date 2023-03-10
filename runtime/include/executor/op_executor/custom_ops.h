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

#ifndef __CUSTOM_OPS__
#define __CUSTOM_OPS__

#include "common/include/common.hpp"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"

namespace nn_compiler
{
namespace runtime
{
namespace op_executor
{
struct rocblas_reduce_sum {
    template <typename T>
    __forceinline__ __device__ void operator()(T &__restrict__ a, const T &__restrict__ b)
    {
        a += b;
    }
};

template <int k, typename REDUCE, typename T>
struct rocblas_reduction_s {
    __forceinline__ __device__ void operator()(int tx, T *x)
    {
        // Reduce the lower half with the upper half
        if (tx < k) REDUCE{}(x[tx], x[tx + k]);
        __syncthreads();

        // Recurse down with k / 2
        rocblas_reduction_s<k / 2, REDUCE, T>{}(tx, x);
    }
};

// leaf node for terminating recursion
template <typename REDUCE, typename T>
struct rocblas_reduction_s<0, REDUCE, T> {
    __forceinline__ __device__ void operator()(int tx, T *x) {}
};

template <int NB, typename REDUCE, typename T>
__attribute__((flatten)) __device__ void rocblas_reduction(int tx, T *x)
{
    static_assert(NB > 1 && !(NB & (NB - 1)), "NB must be a power of 2");
    __syncthreads();
    rocblas_reduction_s<NB / 2, REDUCE, T>{}(tx, x);
}

template <int NB, typename T>
__attribute__((flatten)) __device__ void rocblas_sum_reduce(int tx, T *x)
{
    rocblas_reduction<NB, rocblas_reduce_sum>(tx, x);
}

template <int NB_X, typename T, typename U>
__device__ void gemvt_kernel_calc_xAy(int m, int n, U alpha, const T *x, int lda, const T *A, int incx, U beta, T *y)
{
    int tx = hipThreadIdx_x;

    if (tx < m) A += tx * incx;

    U res;
    res = 0.0;

    int col = 0;
    __shared__ U sdata[NB_X];

    // partial sums
    int m_full = (m / NB_X) * NB_X;

    for (int i = 0; i < m_full; i += NB_X) res += x[i + tx] * A[i * incx];

    if (tx + m_full < m) res += x[m_full + tx] * A[m_full * incx];

    sdata[tx] = res;

    // tree reduction of partial sums,
    if (NB_X > 16) {
        rocblas_sum_reduce<NB_X>(tx, sdata);
    } else {
        __syncthreads();

        if (tx == 0) {
            for (int i = 1; i < m && i < NB_X; i++) sdata[0] += sdata[i];
        }

        __syncthreads();
    }

    if (tx == 0) {
        if (beta != 0.f) {
            y[col] = alpha * sdata[0] + beta * y[col];
        } else {
            y[col] = alpha * sdata[0];
        }
    }
}

template <int NB_X, typename U, typename V, typename W>
__global__ void gemvt_kernel_xAy(int m, int n, U alpha_device_host, const V *xa, int lda, const V *Aa, int incx,
                                 U beta_device_host, W *ya)
{
    const V *x = xa;
    const V *A = Aa + hipBlockIdx_y;
    W *y = ya + hipBlockIdx_y;

    auto alpha = alpha_device_host;
    auto beta = beta_device_host;
    gemvt_kernel_calc_xAy<NB_X>(m, n, alpha, x, lda, A, incx, beta, y);
}

template <typename U, typename V, typename W>
void rocblas_gemv_template_xAy(hipStream_t p_stream, const V *x, const V *A, W *y, int m, int n, int k, U alpha, U beta)
{
    if (m != 1) {
        // found gemm, unsupported
        return;
    }
    static constexpr int NB = 256;
    dim3 gemvt_grid(1, n);
    dim3 gemvt_threads(NB);

    hipLaunchKernelGGL((gemvt_kernel_xAy<NB>), gemvt_grid, gemvt_threads, 0, p_stream, k, n, alpha,
                       x,  // k
                       k,
                       A,  // kxn
                       n, beta, y);
}

// Ax = y
template <int NB_X, typename T, typename U>
__device__ void gemvt_kernel_calc_Axy(int m, int n, U alpha, const T *A, int lda, const T *x, int incx, U beta, T *y)
{
    int tx = hipThreadIdx_x;
    if (tx < n) A += tx;

    U res;
    res = 0.0;

    int col = 0;
    __shared__ U sdata[NB_X];

    // partial sums
    int m_full = (n / NB_X) * NB_X;

    for (int i = 0; i < m_full; i += NB_X) res += A[i] * x[(tx + i) * incx];

    if (tx + m_full < n) res += A[m_full] * x[(tx + m_full) * incx];

    sdata[tx] = res;

    // tree reduction of partial sums,
    if (NB_X > 16) {
        rocblas_sum_reduce<NB_X>(tx, sdata);
    } else {
        __syncthreads();

        if (tx == 0) {
            for (int i = 1; i < n && i < NB_X; i++) sdata[0] += sdata[i];
        }

        __syncthreads();
    }

    if (tx == 0) {
        if (beta != 0) {
            y[col] = alpha * sdata[0] + beta * y[col];
        } else {
            y[col] = alpha * sdata[0];
        }
    }
}

template <int NB_X, typename U, typename V, typename W>
__global__ void gemvt_kernel_Axy(int m, int n, U alpha_device_host, const V *Aa, int lda, const V *xa, int incx,
                                 U beta_device_host, W *ya)
{
    const V *A = Aa + hipBlockIdx_y * n;
    const V *x = xa;
    W *y = ya + hipBlockIdx_y;

    auto alpha = alpha_device_host;
    auto beta = beta_device_host;
    gemvt_kernel_calc_Axy<NB_X>(m, n, alpha, A, lda, x, incx, beta, y);
}

template <typename U, typename V, typename W>
void rocblas_gemv_template_Axy(hipStream_t p_stream, const V *A, const V *x, W *y, int m, int n, int k, U alpha, U beta)
{
    if (n != 1) {
        // found gemm, unsupported
        return;
    }
    static constexpr int NB = 256;
    dim3 gemvt_grid(1, m);
    dim3 gemvt_threads(NB);

    hipLaunchKernelGGL((gemvt_kernel_Axy<NB>), gemvt_grid, gemvt_threads, 0, p_stream, m, k, alpha,
                       A,  // k
                       k,
                       x,  // kxn
                       n, beta, y);
}

template <int NB_X, typename T, typename U>
__device__ void addmv_kernel_calc_xAy(int m, int n, U alpha, const T *b, const T *x, int lda, const T *A, int incx,
                                      U beta, T *y)
{
    int tx = hipThreadIdx_x;

    if (tx < m) A += tx * incx;

    U res;
    res = 0.0;

    int col = 0;
    __shared__ U sdata[NB_X];

    // partial sums
    int m_full = (m / NB_X) * NB_X;

    for (int i = 0; i < m_full; i += NB_X) res += x[i + tx] * A[i * incx];

    if (tx + m_full < m) res += x[m_full + tx] * A[m_full * incx];

    sdata[tx] = res;

    // tree reduction of partial sums,
    if (NB_X > 16) {
        rocblas_sum_reduce<NB_X>(tx, sdata);
    } else {
        __syncthreads();

        if (tx == 0) {
            for (int i = 1; i < m && i < NB_X; i++) sdata[0] += sdata[i];
        }

        __syncthreads();
    }

    if (tx == 0) {
        if (beta != 0) {
            y[col] = alpha * sdata[0] + beta * y[col] + b[col];
        } else {
            y[col] = alpha * sdata[0] + b[col];
        }
    }
}

template <int NB_X, typename U, typename V, typename W>
__global__ void addmv_kernel_xAy(int m, int n, U alpha_device_host, const V *ba, const V *xa, int lda, const V *Aa,
                                 int incx, U beta_device_host, W *ya, bool relu)
{
    const V *b = ba + hipBlockIdx_y;
    const V *x = xa;
    const V *A = Aa + hipBlockIdx_y;
    W *y = ya + hipBlockIdx_y;

    auto alpha = alpha_device_host;
    auto beta = beta_device_host;
    addmv_kernel_calc_xAy<NB_X>(m, n, alpha, b, x, lda, A, incx, beta, y);

    if (relu) {
        if (hipThreadIdx_x == 0) {
            y[0] = y[0] > 0.f ? y[0] : 0.f;
        }
    }
}

template <typename U, typename V, typename W>
void rocblas_addmv_template_xAy(hipStream_t p_stream, const V *b, const V *x, const V *A, W *y, int m, int n, int k,
                                U alpha, U beta, bool relu)
{
    if (m != 1) {
        // found gemm, unsupported
        return;
    }
    static constexpr int NB = 256;
    dim3 addmv_grid(1, n);
    dim3 addmv_threads(NB);

    hipLaunchKernelGGL((addmv_kernel_xAy<NB>), addmv_grid, addmv_threads, 0, p_stream, k, n, alpha, b,
                       x,  // k
                       k,
                       A,  // kxn
                       n, beta, y, relu);
}

template <int NB_X, typename T, typename U>
__device__ void addmv_kernel_calc_Axy(int m, int n, U alpha, const T *b, const T *A, int lda, const T *x, int incx,
                                      U beta, T *y)
{
    int tx = hipThreadIdx_x;
    if (tx < n) A += tx;

    U res;
    res = 0.0;

    int col = 0;
    __shared__ U sdata[NB_X];

    // partial sums
    int m_full = (n / NB_X) * NB_X;

    for (int i = 0; i < m_full; i += NB_X) res += A[i] * x[(tx + i) * incx];

    if (tx + m_full < n) res += A[m_full] * x[(tx + m_full) * incx];

    sdata[tx] = res;

    // tree reduction of partial sums,
    if (NB_X > 16) {
        rocblas_sum_reduce<NB_X>(tx, sdata);
    } else {
        __syncthreads();

        if (tx == 0) {
            for (int i = 1; i < n && i < NB_X; i++) sdata[0] += sdata[i];
        }

        __syncthreads();
    }

    if (tx == 0) {
        if (beta != 0) {
            y[col] = alpha * sdata[0] + beta * y[col] + b[col];
        } else {
            y[col] = alpha * sdata[0] + b[col];
        }
    }
}

template <int NB_X, typename U, typename V, typename W>
__global__ void addmv_kernel_Axy(int m, int n, U alpha_device_host, const V *ba, const V *Aa, int lda, const V *xa,
                                 int incx, U beta_device_host, W *ya, bool relu)
{
    const V *b = ba + hipBlockIdx_y;
    const V *A = Aa + hipBlockIdx_y * n;
    const V *x = xa;
    W *y = ya + hipBlockIdx_y;

    auto alpha = alpha_device_host;
    auto beta = beta_device_host;
    addmv_kernel_calc_Axy<NB_X>(m, n, alpha, b, A, lda, x, incx, beta, y);

    if (relu) {
        if (hipThreadIdx_x == 0) {
            y[0] = y[0] > 0.f ? y[0] : 0.f;
        }
    }
}

template <typename U, typename V, typename W>
void rocblas_addmv_template_Axy(hipStream_t p_stream, const V *b, const V *A, const V *x, W *y, int m, int n, int k,
                                U alpha, U beta, bool relu)
{
    if (n != 1) {
        // found gemm, unsupported
        return;
    }
    static constexpr int NB = 256;
    dim3 addmv_grid(1, m);
    dim3 addmv_threads(NB);

    hipLaunchKernelGGL((addmv_kernel_Axy<NB>), addmv_grid, addmv_threads, 0, p_stream, m, k, alpha, b,
                       A,  // k
                       k,
                       x,  // kxn
                       n, beta, y, relu);
}

template <int block_x, int ele_per_thread, typename U>
__global__ void add_kernel(int m, int n, U *A, U *B, U *C)
{
    if (hipBlockIdx_y >= m || (hipBlockIdx_x * block_x + hipThreadIdx_x) * ele_per_thread >= n) {
        return;
    }
    U *index_a = A + hipBlockIdx_y * n + (hipBlockIdx_x * block_x + hipThreadIdx_x) * ele_per_thread;
    U *index_b = B + hipBlockIdx_y * n + (hipBlockIdx_x * block_x + hipThreadIdx_x) * ele_per_thread;
    U *index_c = C + hipBlockIdx_y * n + (hipBlockIdx_x * block_x + hipThreadIdx_x) * ele_per_thread;

    reinterpret_cast<half2 *>(index_c)[0] =
        reinterpret_cast<half2 *>(index_a)[0] + reinterpret_cast<half2 *>(index_b)[0];
    reinterpret_cast<half2 *>(index_c)[1] =
        reinterpret_cast<half2 *>(index_a)[1] + reinterpret_cast<half2 *>(index_b)[1];
}

template <int block_x, int ele_per_thread, typename U>
__global__ void add_kernel_unsym(int m, int n, U *A, U *B, U *C)
{
    if (hipBlockIdx_y >= m || (hipBlockIdx_x * block_x + hipThreadIdx_x) * ele_per_thread >= n) {
        return;
    }
    U *index_a = A + hipBlockIdx_y * n + (hipBlockIdx_x * block_x + hipThreadIdx_x) * ele_per_thread;
    U *index_b = B + (hipBlockIdx_x * block_x + hipThreadIdx_x) * ele_per_thread;
    U *index_c = C + hipBlockIdx_y * n + (hipBlockIdx_x * block_x + hipThreadIdx_x) * ele_per_thread;

    reinterpret_cast<half2 *>(index_c)[0] =
        reinterpret_cast<half2 *>(index_a)[0] + reinterpret_cast<half2 *>(index_b)[0];
    reinterpret_cast<half2 *>(index_c)[1] =
        reinterpret_cast<half2 *>(index_a)[1] + reinterpret_cast<half2 *>(index_b)[1];
}

template <typename U, typename W>
void custom_add(hipStream_t p_stream, U *A, U *B, U *C, int m, int n, W alpha, bool sym, int a_m_s, int a_n_s,
                int b_m_s, int b_n_s)
{
    static constexpr int block_x = 256;
    static constexpr int per_thread = 4;
    int num_n = (n + block_x - 1) / block_x / per_thread;
    if (num_n == 0) {
        num_n = 1;
    }
    dim3 add_grid(num_n, m);
    dim3 add_threads(block_x);

    if (sym) {
        hipLaunchKernelGGL((add_kernel<block_x, per_thread>), add_grid, add_threads, 0, p_stream, m, n, A, B, C);
    } else {
        hipLaunchKernelGGL((add_kernel_unsym<block_x, per_thread>), add_grid, add_threads, 0, p_stream, m, n, A, B, C);
    }
}

template <int NB_X, typename V, typename W>
__global__ void bmm_kernel_xAy(int m, int n, int k, const V *xa, const V *Aa, W *ya)
{
    int index = hipBlockIdx_x * NB_X + hipThreadIdx_x;
    const V *x = xa;
    const V *A = Aa + index;
    W *y = ya + index;

    W sum = 0.0;
    for (int i = 0; i < k; ++i) {
        sum += x[i] * A[i * n];
    }
    y[0] = sum;
}

template <typename V, typename W>
void rocblas_bmm_template_xAy(hipStream_t p_stream, const V *x, const V *A, W *y, int m, int n, int k)
{
    if (m != 1) {
        // found gemm, unsupported
        return;
    }
    static constexpr int NB = 256;
    int num_n = (n + NB - 1) / NB;
    dim3 gemvt_grid(num_n);
    dim3 gemvt_threads(NB);

    hipLaunchKernelGGL((bmm_kernel_xAy<NB>), gemvt_grid, gemvt_threads, 0, p_stream, m, n, k,
                       x,  // k
                       A,  // kxn
                       y);
}

}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler

#endif  // __CUSTOM_OPS__
