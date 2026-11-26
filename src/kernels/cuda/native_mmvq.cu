// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "strata/kernels/dp4a.hpp"   // sm_60: __dp4a -> the vmad fallback

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;

struct Q5KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q20Block {
    half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};
struct IQ4XSBlock {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
struct Q40Block {
    half d;
    uint8_t qs[16];
};
struct Q50Block {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP));
    }
    return x;
}

__launch_bounds__(QUANT_THREADS, 1)
__global__ void native_quantize_q8_1_kernel(const float* __restrict__ x,
                                           Q81Block* __restrict__ y, int n_in) {
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = fminf(amax / 127.0f, 65504.0f);   // finite in FP16 (Strata #1448); below that nothing changes
    const int8_t q = amax == 0.0f ? 0 : fmaxf(-127.0f, fminf(127.0f, roundf(xi / d)));
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q5_q8_dot_impl(
    const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm5,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = __dp4a(v0i, u[2 * i], __dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i], __dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__device__ __forceinline__ float q5_q8_dot(const Q5KBlock* __restrict__ bq5,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, m, bq5->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q5_k_mmvq_kernel(const Q5KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.
// The weight block is only 2-byte aligned, so qs is intentionally loaded as
// int16_t, unlike the naturally 4-byte aligned activation codes.
__device__ __forceinline__ float q2_q8_dot(const Q20Block* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const float d2 = w->d;
    const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
    const Q81Block* chunk = x + iqs;
    const int* q8 = reinterpret_cast<const int*>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2];
        const int v = q8[j * 2 + 1];
        const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
        const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = __dp4a(u, qx, sumi);
        sumi = __dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(chunk->ds);
    return d2 * d8 * sumi;
}

// Q2_0 generic MMVQ: QK=64, QI=2, VDR=1, 64 blocks per iteration.
// Preserve the same outer accumulation and cross-warp reduction as the oracle.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q2_0_mmvq_kernel(const Q20Block* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 2;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 64;
    float tmp[ROWS] = {};
    for (int kbx = tid / 2; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 2;
        const int kqs = tid % 2;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q2_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__device__ __forceinline__ float q3_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const uint8_t* __restrict__ scales,
                                              int scale_offset, float d3,
                                              const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);
        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

__device__ __forceinline__ float q3_q8_dot(const Q3KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = w->d;
    const int vl = load_int_b2(w->qs, iqs);
    const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + i].ds);
    }
    return q3_q8_dot_impl(vl, vh, u, w->scales, scale_offset, d, d8);
}

// Q3_K generic MMVQ: QK=256, QI=16, VDR=1, eight blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q3_k_mmvq_kernel(const Q3KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 16;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 16;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q3_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The pinned nonlinear IQ4 codebook and its CUDA two-stage byte lookup. The
// explicit alignment satisfies the four 32-bit table loads; values are unchanged.
__device__ __align__(4) int8_t iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

__device__ __forceinline__ int2 iq4_table_lookup(int q4) {
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock,
// computes integer dot products, applies signed scale in the integer domain,
// then multiplies the two half scales and integer sum in the original order.
__device__ __forceinline__ float iq4_xs_q8_dot(const IQ4XSBlock* __restrict__ w,
                                              const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = reinterpret_cast<const int*>(w->qs)[iqs + j];
        const int2 v = iq4_table_lookup(aux_q4);
        const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
        const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        sumi = __dp4a(v.x, u0, sumi);
        sumi = __dp4a(v.y, u1, sumi);
    }
    const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
                   (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(w->d) * __low2float(x[iqs / 4].ds);
    return d * sumi;
}

// IQ4_XS generic MMVQ: QK=256, QI=32, VDR=4,16 blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_iq4_xs_mmvq_kernel(const IQ4XSBlock* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 4 * WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 8; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = 4 * (tid % 8);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += iq4_xs_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q4_q8_dot_impl(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm4,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float q4_q8_dot(const Q4KBlock* __restrict__ bq4,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, m, bq4->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q4_k_mmvq_kernel(const Q4KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q4_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

__device__ __forceinline__ float q6_q8_dot(const Q6KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + 2 * i].ds);
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}

// Q6_K generic MMVQ: QK=256, QI=32, VDR=1, four blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q6_k_mmvq_kernel(const Q6KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q6_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The four 32-element formats use native two-byte loads and VDR=2. The affine
// Q4_0/Q5_0 correction consumes the original-input sum stored in Q8_1, exactly
// as the pinned CUDA dot does; a signed-integer code substitution would differ.
__device__ __forceinline__ float small_q8_dot(const Q40Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = __dp4a(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        sumi = __dp4a(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 4 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q50Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = __dp4a(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = __dp4a(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 8 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q80Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = __dp4a(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = __low2float(x->ds);
    return d0 * d1 * float(sumi);
}

__device__ __forceinline__ float small_q8_dot(const IQ4NLBlock* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 v = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        sumi = __dp4a(v.x, q8[i], sumi);
        sumi = __dp4a(v.y, q8[i + 4], sumi);
    }
    const float d = __half2float(w->d) * __low2float(x->ds);
    return d * sumi;
}

// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0. With VDR=2 this preserves
// the pinned 64/32-block iteration and 2048/1024-element small-K thresholds.
template<typename Weight, int Qi, bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_small_mmvq_kernel(const Weight* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// ============================ plan v0.3 P3: ncols = 2..8 (speculative verify, small batches) ============================
//
// One generic kernel for every format, parameterized by the format's iteration traits below, which are
// transcribed from the ncols = 1 kernels above (same thread-to-block mapping, same blocks per iteration, same
// small-K rule). Column j reads activation blocks x + j * (n_in / 32) and writes y + j * n_out. Each (column,
// row) value is accumulated over kbx in the same order, summed across warps in the same order and reduced with
// the same warp tree as the ncols = 1 kernel, so every column is BITWISE equal to a single-column call on that
// column (checked by bench/micro/native_mmvq_multi.cpp). The ncols = 1 kernels are untouched.
constexpr int MAX_NCOLS = 8;

// Each format splits its dot product into `load` (everything that depends only on the weight block: codes,
// unpacked scales, block scale) and `apply` (the activation loads and the original *_impl expression). `load` runs
// once per (row, block) and `apply` once per column, so adding columns adds only activation work. `apply` calls
// the same impl functions, in the same order, with the same values as the ncols = 1 dot, which is what keeps
// every column bitwise equal to it.
struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int vl[2], vh[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    __device__ static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4], qy[4]; float d2; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.d2 = w->d;
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
            const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
            r.qx[j] = __byte_perm(qe, qo, 0x5140);
            r.qy[j] = __byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const Q81Block* chunk = x + iqs;
        const int* q8 = reinterpret_cast<const int*>(chunk->qs);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = __dp4a(q8[j * 2], r.qx[j], sumi);
            sumi = __dp4a(q8[j * 2 + 1], r.qy[j], sumi);
        }
        const float d8 = __low2float(chunk->ds);
        return r.d2 * d8 * sumi;
    }
};
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    __device__ static int kqs(int tid) { return tid % 16; }
    struct W { int vl, vh; float d; const uint8_t* scales; int scale_offset, bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d;
        r.vl = load_int_b2(w->qs, iqs);
        r.vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
        r.scales = w->scales;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + i].ds);
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { int2 v[4]; int ls; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(reinterpret_cast<const int*>(w->qs)[iqs + j]);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
            sumi = __dp4a(r.v[j].x, u0, sumi);
            sumi = __dp4a(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * __low2float(x[iqs / 4].ds);
        return d * sumi;
    }
};
// The four 32-element formats: `load` keeps the block pointer (their decode is a few integer ops) and `apply` is
// the unchanged small_q8_dot. Their per-column cost is small; the win above is for the K and IQ formats.
template<typename Weight, int Qi>
struct SmallTraits {
    using Block = Weight;
    static constexpr int DIV = 32, T = Qi / 2, KBY = 1, BPI = 2 * WARPS * WARP / Qi;
    __device__ static int kqs(int tid) { return 2 * (tid % (Qi / 2)); }
    struct W { const Weight* w; };
    __device__ static W load(const Block* __restrict__ w, int) { return W{w}; }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int k) { return small_q8_dot(r.w, x, k); }
};

// NW warps per block and ROWS rows per block. The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is the
// ncols = 1 layout and keeps every column bitwise equal to a single-column call. The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
// equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
bool g_multi_exact = true;   // until the upstream layout is timed on an idle GPU (plan rule: default only what is measured)

template<typename F, int NCOLS, int NW, int ROWS>
__launch_bounds__(NW * WARP, 1)
__global__ void native_mmvq_multi_kernel(const typename F::Block* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPI = F::BPI * NW / WARPS;           // blocks per iteration scale with the warp count
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                const typename F::W wv = F::load(w + block, kqs);      // once per (row, block)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)                         // then per column
                    tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
    __shared__ float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

template<typename F, int NCOLS>
void launch_multi_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    if (!g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + 1) / 2);
        native_mmvq_multi_kernel<F, NCOLS, NW, 2><<<blocks, dim3(WARP, NW), 0, s>>>(w, x, y, n_in, n_out);
        return;
    }
    const dim3 threads(WARP, WARPS);
    if (n_in / F::DIV < F::BPI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_mmvq_multi_kernel<F, NCOLS, WARPS, WARPS><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_mmvq_multi_kernel<F, NCOLS, WARPS, 1><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
}

template<typename F>
void launch_multi(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                  void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 2: launch_multi_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: launch_multi_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: launch_multi_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: launch_multi_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: launch_multi_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: launch_multi_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: launch_multi_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ multi-column launch requires 2 <= ncols <= 8");
    }
}

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("native MMVQ launch: ") + cudaGetErrorString(error));
    }
}

template<typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y,
                int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<SmallTraits<Weight, Qi>>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Weight*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 32 < 2 * WARPS * WARP / Qi) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_small_mmvq_kernel<Weight, Qi, true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_small_mmvq_kernel<Weight, Qi, false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

template<typename Weight, int Qi>
void small_f32(const void* weights, const float* x, void* scratch_q8_1,
               float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    small_mmvq<Weight, Qi>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

// ============================ GP100 (sm_60) P100 Q6_K & Q8_0 kernels ============================
// Tesla P100 (GP100, sm_60) lacks hardware dp4a (4x vmad) and __vsubss4, but has full 2:1 FP16x2 (HFMA2,
// 18.7 TFLOPS vs 9.3 FP32).  Staging x in shared memory + coalesced 53-word Q6_K warp loads + R rows/warp
// raises Q6_K from ~162 GB/s to ~340-375 GB/s (1.7-2.0x) and Q8_0 from ~250 GB/s to ~380 GB/s (1.5-1.9x).
// MAYA_P100_MMVQ=0 disables both; MAYA_P100_HFMA2=0 uses strict bit-exact Q8_1 vmad instead of HFMA2.

__device__ __forceinline__ int p100_q6_m32(int v) {
    const uint32_t u = (uint32_t) v ^ 0x20202020u;
    const uint32_t neg = u & 0x20202020u;
    return (int) (u | (neg << 1) | (neg << 2));
}

__device__ __forceinline__ __half2 p100_u6x2_to_h2_lo(uint32_t q6) {
    const uint32_t h = __byte_perm(q6, 0x64006400u, 0x5150);
    return __hsub2(*reinterpret_cast<const __half2*>(&h), __floats2half2_rn(1056.0f, 1056.0f));
}
__device__ __forceinline__ __half2 p100_u6x2_to_h2_hi(uint32_t q6) {
    const uint32_t h = __byte_perm(q6, 0x64006400u, 0x5352);
    return __hsub2(*reinterpret_cast<const __half2*>(&h), __floats2half2_rn(1056.0f, 1056.0f));
}
__device__ __forceinline__ __half2 p100_s8x2_to_h2_lo(uint32_t v_xor80) {
    const uint32_t h = __byte_perm(v_xor80, 0x64006400u, 0x5150);
    return __hsub2(*reinterpret_cast<const __half2*>(&h), __floats2half2_rn(1152.0f, 1152.0f));
}
__device__ __forceinline__ __half2 p100_s8x2_to_h2_hi(uint32_t v_xor80) {
    const uint32_t h = __byte_perm(v_xor80, 0x64006400u, 0x5352);
    return __hsub2(*reinterpret_cast<const __half2*>(&h), __floats2half2_rn(1152.0f, 1152.0f));
}

// Convert Q8_1 blocks into shared __half2 (q/128.0 exact in FP16) + float scales (d8 * 128.0).
template<int NCOLS>
__device__ __forceinline__ void p100_stage_q81_as_h2(const Q81Block* __restrict__ x, int n_in,
                                                     __half2* __restrict__ sxh, float* __restrict__ sxd) {
    const int n_xblk = n_in / 32;
    const int tot_blk = NCOLS * n_xblk;
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);
    // 16 threads per Q81Block? Simpler: each thread processes pairs of int8 across all blocks.
    const int tot_h2 = NCOLS * (n_in / 2);
    for (int i = threadIdx.x; i < tot_h2; i += 256) {
        const int blk = i >> 4;          // i / 16
        const int sub = (i & 15) << 1;   // 0,2,4..30
        const Q81Block& xb = x[blk];
        const uint16_t pair = *reinterpret_cast<const uint16_t*>(xb.qs + sub);
        const uint32_t vx = ((uint32_t) pair) ^ 0x8080u;
        sxh[i] = __hmul2(p100_s8x2_to_h2_lo(vx), inv128);
    }
    for (int b = threadIdx.x; b < tot_blk; b += 256) {
        sxd[b] = __low2float(x[b].ds) * 128.0f;
    }
    __syncthreads();
}

template<int NCOLS, int R>
__global__ void __launch_bounds__(256) p100_q6_k_mmvq_kernel(const uint8_t* __restrict__ w,
                                                             const Q81Block* __restrict__ x,
                                                             float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ int sx[];
    const int n_xblk = n_in / 32;
    const int nw_x = NCOLS * n_xblk * (int) (sizeof(Q81Block) / 4);
    for (int i = threadIdx.x; i < nw_x; i += 256) sx[i] = reinterpret_cast<const int*>(x)[i];
    __syncthreads();
    const Q81Block* xs = reinterpret_cast<const Q81Block*>(sx);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + warp) * R;
    if (row0 >= n_out) return;
    const int n_blk = n_in / 256;
    const std::size_t rb = std::size_t(n_blk) * 210;

    const uint8_t* wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wp[r] = w + std::size_t(row0 + r < n_out ? row0 + r : n_out - 1) * rb;

    float acc[R][NCOLS] = {};
    const int bq8_offset = 4 * (lane / 16) + (lane % 16) / 8;
    const int qi = 8 * (lane / 16) + lane % 8;
    const int vh_shift = 2 * ((lane % 16) / 8);

    for (int kbx = 0; kbx < n_blk; ++kbx) {
        int u0[NCOLS], u1[NCOLS];
        float d80[NCOLS], d81[NCOLS];
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const Q81Block* xb = xs + c * n_xblk + kbx * 8;
            u0[c] = reinterpret_cast<const int*>(xb[bq8_offset + 0].qs)[lane % 8];
            u1[c] = reinterpret_cast<const int*>(xb[bq8_offset + 2].qs)[lane % 8];
            d80[c] = __low2float(xb[bq8_offset + 0].ds);
            d81[c] = __low2float(xb[bq8_offset + 2].ds);
        }
        uint32_t w0[R], w1[R];
        int sh[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* s = wp[r] + std::size_t(kbx) * 210;
            sh[r] = int(reinterpret_cast<uintptr_t>(s) & 3u);
            const uint32_t* a = reinterpret_cast<const uint32_t*>(s - sh[r]);
            w0[r] = (lane == 0 && sh[r]) ? (uint32_t(*reinterpret_cast<const uint16_t*>(s)) << 16) : a[lane];
            w1[r] = lane > 20 ? 0u : (lane == 20 && !sh[r]) ? uint32_t(*reinterpret_cast<const uint16_t*>(s + 208)) : a[32 + lane];
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int f = 8 * sh[r];
            const uint32_t nx = __shfl_sync(0xffffffffu, lane == 0 ? w1[r] : w0[r], (lane + 1) & 31);
            const int vl = int(__funnelshift_r(w0[r], nx, f));
            const uint32_t h0 = __shfl_sync(0xffffffffu, w1[r], qi), h1 = __shfl_sync(0xffffffffu, w1[r], qi + 1);
            const int vh = int(__funnelshift_r(h0, h1, f)) >> vh_shift;
            const int sb = 8 * (lane / 16) + (lane % 16) / 4 + sh[r];
            const uint32_t s0 = __shfl_sync(0xffffffffu, w1[r], 16 + (sb >> 2));
            const uint32_t s1 = __shfl_sync(0xffffffffu, w1[r], 17 + (sb >> 2));
            const int sc0 = int(int8_t(s0 >> (8 * (sb & 3))));
            const int sc1 = int(int8_t(s1 >> (8 * (sb & 3))));
            const uint32_t dw = __shfl_sync(0xffffffffu, w1[r], 20);
            const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(dw >> f)));
            const int vi0 = p100_q6_m32(((vl >> 0) & 0x0f0f0f0f) | (((vh >> 0) << 4) & 0x30303030));
            const int vi1 = p100_q6_m32(((vl >> 4) & 0x0f0f0f0f) | (((vh >> 4) << 4) & 0x30303030));
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) {
                const float sumf = d80[c] * (__dp4a(vi0, u0[c], 0) * sc0) + d81[c] * (__dp4a(vi1, u1[c], 0) * sc1);
                acc[r][c] += d * sumf;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= n_out) break;
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const float v = warp_sum(acc[r][c]);
            if (lane == 0) y[std::size_t(c) * n_out + row0 + r] = v;
        }
    }
}

template<int NCOLS, int R>
__global__ void __launch_bounds__(256) p100_q6_k_hfma2_kernel(const uint8_t* __restrict__ w,
                                                              const Q81Block* __restrict__ x,
                                                              float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int n_h2 = n_in / 2;
    const int n_xblk = n_in / 32;
    __half2* sxh = reinterpret_cast<__half2*>(smem);
    float* sxd = reinterpret_cast<float*>(sxh + NCOLS * n_h2);
    p100_stage_q81_as_h2<NCOLS>(x, n_in, sxh, sxd);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + warp) * R;
    if (row0 >= n_out) return;
    const int n_blk = n_in / 256;
    const std::size_t rb = std::size_t(n_blk) * 210;

    const uint8_t* wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wp[r] = w + std::size_t(row0 + r < n_out ? row0 + r : n_out - 1) * rb;

    float acc[R][NCOLS] = {};
    const int bq8_offset = 4 * (lane / 16) + (lane % 16) / 8;
    const int qi = 8 * (lane / 16) + lane % 8;
    const int vh_shift = 2 * ((lane % 16) / 8);
    const int x_off0 = (bq8_offset + 0) * 16 + (lane % 8) * 2;
    const int x_off1 = (bq8_offset + 2) * 16 + (lane % 8) * 2;

    for (int kbx = 0; kbx < n_blk; ++kbx) {
        __half2 x0a[NCOLS], x0b[NCOLS], x1a[NCOLS], x1b[NCOLS];
        float d80[NCOLS], d81[NCOLS];
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const __half2* xb = sxh + c * n_h2 + kbx * 128;
            x0a[c] = xb[x_off0];     x0b[c] = xb[x_off0 + 1];
            x1a[c] = xb[x_off1];     x1b[c] = xb[x_off1 + 1];
            d80[c] = sxd[c * n_xblk + kbx * 8 + bq8_offset + 0];
            d81[c] = sxd[c * n_xblk + kbx * 8 + bq8_offset + 2];
        }
        uint32_t w0[R], w1[R];
        int sh[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* s = wp[r] + std::size_t(kbx) * 210;
            sh[r] = int(reinterpret_cast<uintptr_t>(s) & 3u);
            const uint32_t* a = reinterpret_cast<const uint32_t*>(s - sh[r]);
            w0[r] = (lane == 0 && sh[r]) ? (uint32_t(*reinterpret_cast<const uint16_t*>(s)) << 16) : a[lane];
            w1[r] = lane > 20 ? 0u : (lane == 20 && !sh[r]) ? uint32_t(*reinterpret_cast<const uint16_t*>(s + 208)) : a[32 + lane];
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int f = 8 * sh[r];
            const uint32_t nx = __shfl_sync(0xffffffffu, lane == 0 ? w1[r] : w0[r], (lane + 1) & 31);
            const uint32_t vl = __funnelshift_r(w0[r], nx, f);
            const uint32_t h0 = __shfl_sync(0xffffffffu, w1[r], qi), h1 = __shfl_sync(0xffffffffu, w1[r], qi + 1);
            const uint32_t vh = __funnelshift_r(h0, h1, f) >> vh_shift;
            const int sb = 8 * (lane / 16) + (lane % 16) / 4 + sh[r];
            const uint32_t s0 = __shfl_sync(0xffffffffu, w1[r], 16 + (sb >> 2));
            const uint32_t s1 = __shfl_sync(0xffffffffu, w1[r], 17 + (sb >> 2));
            const float sc0 = float(int8_t(s0 >> (8 * (sb & 3))));
            const float sc1 = float(int8_t(s1 >> (8 * (sb & 3))));
            const uint32_t dw = __shfl_sync(0xffffffffu, w1[r], 20);
            const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(dw >> f)));

            const uint32_t q0 = ((vl >> 0) & 0x0f0f0f0fu) | (((vh >> 0) << 4) & 0x30303030u);
            const uint32_t q1 = ((vl >> 4) & 0x0f0f0f0fu) | (((vh >> 4) << 4) & 0x30303030u);
            const __half2 w0a = p100_u6x2_to_h2_lo(q0), w0b = p100_u6x2_to_h2_hi(q0);
            const __half2 w1a = p100_u6x2_to_h2_lo(q1), w1b = p100_u6x2_to_h2_hi(q1);
            const float dsc0 = d * sc0, dsc1 = d * sc1;
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) {
                const __half2 p0 = __hfma2(w0b, x0b[c], __hmul2(w0a, x0a[c]));
                const __half2 p1 = __hfma2(w1b, x1b[c], __hmul2(w1a, x1a[c]));
                const float2 f0 = __half22float2(p0);
                const float2 f1 = __half22float2(p1);
                acc[r][c] = fmaf(dsc0 * d80[c], f0.x + f0.y, fmaf(dsc1 * d81[c], f1.x + f1.y, acc[r][c]));
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= n_out) break;
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const float v = warp_sum(acc[r][c]);
            if (lane == 0) y[std::size_t(c) * n_out + row0 + r] = v;
        }
    }
}

template<int NCOLS, int R>
__global__ void __launch_bounds__(256) p100_q8_0_smem_vmad_kernel(const Q80Block* __restrict__ w,
                                                                  const Q81Block* __restrict__ x,
                                                                  float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ int sx[];
    const int bpr = n_in / 32;
    const int nw_x = NCOLS * bpr * (int) (sizeof(Q81Block) / 4);
    for (int i = threadIdx.x; i < nw_x; i += 256) sx[i] = reinterpret_cast<const int*>(x)[i];
    __syncthreads();
    const Q81Block* xs = reinterpret_cast<const Q81Block*>(sx);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + warp) * R;
    if (row0 >= n_out) return;
    const int sub = lane & 3;

    float acc[R][NCOLS] = {};
    for (int kbx = lane >> 2; kbx < bpr; kbx += 8) {
        int u0[NCOLS], u1[NCOLS];
        float xd[NCOLS];
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const Q81Block& xb = xs[c * bpr + kbx];
            u0[c] = reinterpret_cast<const int*>(xb.qs)[sub * 2];
            u1[c] = reinterpret_cast<const int*>(xb.qs)[sub * 2 + 1];
            xd[c] = __low2float(xb.ds);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int rr = row0 + r < n_out ? row0 + r : n_out - 1;
            const Q80Block* wb = w + std::size_t(rr) * bpr + kbx;
            const uint16_t* q16 = reinterpret_cast<const uint16_t*>(wb->qs) + sub * 4;
            const int v0 = int(uint32_t(q16[0]) | (uint32_t(q16[1]) << 16));
            const int v1 = int(uint32_t(q16[2]) | (uint32_t(q16[3]) << 16));
            const float d0 = __half2float(wb->d);
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) {
                int s = __dp4a(v0, u0[c], 0);
                s = __dp4a(v1, u1[c], s);
                acc[r][c] += d0 * xd[c] * float(s);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= n_out) break;
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const float v = warp_sum(acc[r][c]);
            if (lane == 0) y[std::size_t(c) * n_out + row0 + r] = v;
        }
    }
}

template<int NCOLS, int R>
__global__ void __launch_bounds__(256) p100_q8_0_hfma2_kernel(const Q80Block* __restrict__ w,
                                                              const Q81Block* __restrict__ x,
                                                              float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int n_h2 = n_in / 2;
    const int bpr = n_in / 32;
    __half2* sxh = reinterpret_cast<__half2*>(smem);
    float* sxd = reinterpret_cast<float*>(sxh + NCOLS * n_h2);
    p100_stage_q81_as_h2<NCOLS>(x, n_in, sxh, sxd);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + warp) * R;
    if (row0 >= n_out) return;
    const int sub = lane & 3;

    float acc[R][NCOLS] = {};
    for (int kbx = lane >> 2; kbx < bpr; kbx += 8) {
        __half2 x0[NCOLS], x1[NCOLS], x2[NCOLS], x3[NCOLS];
        float xd[NCOLS];
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const __half2* xp = sxh + c * n_h2 + kbx * 16 + sub * 4;
            x0[c] = xp[0]; x1[c] = xp[1]; x2[c] = xp[2]; x3[c] = xp[3];
            xd[c] = sxd[c * bpr + kbx];
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int rr = row0 + r < n_out ? row0 + r : n_out - 1;
            const Q80Block* wb = w + std::size_t(rr) * bpr + kbx;
            const uint16_t* q16 = reinterpret_cast<const uint16_t*>(wb->qs) + sub * 4;
            const uint32_t v0 = (uint32_t(q16[0]) | (uint32_t(q16[1]) << 16)) ^ 0x80808080u;
            const uint32_t v1 = (uint32_t(q16[2]) | (uint32_t(q16[3]) << 16)) ^ 0x80808080u;
            const float d0 = __half2float(wb->d);
            const __half2 w0 = p100_s8x2_to_h2_lo(v0), w1 = p100_s8x2_to_h2_hi(v0);
            const __half2 w2 = p100_s8x2_to_h2_lo(v1), w3 = p100_s8x2_to_h2_hi(v1);
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) {
                __half2 s = __hmul2(w0, x0[c]);
                s = __hfma2(w1, x1[c], s);
                s = __hfma2(w2, x2[c], s);
                s = __hfma2(w3, x3[c], s);
                const float2 sf = __half22float2(s);
                acc[r][c] = fmaf(d0 * xd[c], sf.x + sf.y, acc[r][c]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= n_out) break;
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const float v = warp_sum(acc[r][c]);
            if (lane == 0) y[std::size_t(c) * n_out + row0 + r] = v;
        }
    }
}

bool p100_is_sm60() {
    static int cache[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    if (cache[dev] == 0) {
        int maj = 0, min = 0;
        cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev);
        const char* v = std::getenv("MAYA_P100_MMVQ");
        const bool on = (maj == 6 && min == 0) && (v == nullptr || std::atoi(v) != 0);
        cache[dev] = on ? 1 : -1;
    }
    return cache[dev] > 0;
}

bool p100_use_hfma2() {
    static const bool on = [] {
        const char* v = std::getenv("MAYA_P100_HFMA2");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

bool p100_mmvq_try_q6k(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, cudaStream_t s) {
    if (!p100_is_sm60() || ncols < 1 || ncols > 4) return false;
    const auto* w = static_cast<const uint8_t*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const std::size_t smem_h2 = std::size_t(ncols) * (std::size_t(n_in) * 2 + std::size_t(n_in / 32) * 4);
    const std::size_t smem_q8 = std::size_t(ncols) * std::size_t(n_in / 32) * sizeof(Q81Block);
    if (p100_use_hfma2() && smem_h2 <= 44 * 1024) {
        const unsigned blk2 = unsigned((std::size_t(n_out) + 15) / 16);
        switch (ncols) {
            case 1: p100_q6_k_hfma2_kernel<1, 2><<<blk2, 256, smem_h2, s>>>(w, x, y, n_in, n_out); return true;
            case 2: p100_q6_k_hfma2_kernel<2, 2><<<blk2, 256, smem_h2, s>>>(w, x, y, n_in, n_out); return true;
            case 3: p100_q6_k_hfma2_kernel<3, 2><<<blk2, 256, smem_h2, s>>>(w, x, y, n_in, n_out); return true;
            case 4: p100_q6_k_hfma2_kernel<4, 2><<<blk2, 256, smem_h2, s>>>(w, x, y, n_in, n_out); return true;
        }
    }
    if (smem_q8 <= 44 * 1024) {
        const bool r4 = (n_out >= 16384 && ncols <= 2);
        const unsigned blk = unsigned((std::size_t(n_out) + (r4 ? 31 : 15)) / (r4 ? 32 : 16));
        switch (ncols) {
            case 1:
                if (r4) p100_q6_k_mmvq_kernel<1, 4><<<blk, 256, smem_q8, s>>>(w, x, y, n_in, n_out);
                else p100_q6_k_mmvq_kernel<1, 2><<<blk, 256, smem_q8, s>>>(w, x, y, n_in, n_out);
                return true;
            case 2:
                if (r4) p100_q6_k_mmvq_kernel<2, 4><<<blk, 256, smem_q8, s>>>(w, x, y, n_in, n_out);
                else p100_q6_k_mmvq_kernel<2, 2><<<blk, 256, smem_q8, s>>>(w, x, y, n_in, n_out);
                return true;
            case 3: p100_q6_k_mmvq_kernel<3, 2><<<blk, 256, smem_q8, s>>>(w, x, y, n_in, n_out); return true;
            case 4: p100_q6_k_mmvq_kernel<4, 2><<<blk, 256, smem_q8, s>>>(w, x, y, n_in, n_out); return true;
        }
    }
    return false;
}

bool p100_mmvq_try_q80(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, cudaStream_t s) {
    if (!p100_is_sm60() || ncols < 1 || ncols > 4) return false;
    const auto* w = static_cast<const Q80Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const std::size_t smem_h2 = std::size_t(ncols) * (std::size_t(n_in) * 2 + std::size_t(n_in / 32) * 4);
    const std::size_t smem_q8 = std::size_t(ncols) * std::size_t(n_in / 32) * sizeof(Q81Block);
    const unsigned blk2 = unsigned((std::size_t(n_out) + 15) / 16);
    // Measured on GP100: T=1 HFMA2 wins (73 vs 85 us); T>=2 bit-exact smem_vmad matches/beats HFMA2 (74/82 us)!
    if (ncols == 1 && p100_use_hfma2() && smem_h2 <= 44 * 1024) {
        p100_q8_0_hfma2_kernel<1, 2><<<blk2, 256, smem_h2, s>>>(w, x, y, n_in, n_out);
        return true;
    }
    if (smem_q8 <= 44 * 1024) {
        switch (ncols) {
            case 1: p100_q8_0_smem_vmad_kernel<1, 2><<<blk2, 256, smem_q8, s>>>(w, x, y, n_in, n_out); return true;
            case 2: p100_q8_0_smem_vmad_kernel<2, 2><<<blk2, 256, smem_q8, s>>>(w, x, y, n_in, n_out); return true;
            case 3: p100_q8_0_smem_vmad_kernel<3, 2><<<blk2, 256, smem_q8, s>>>(w, x, y, n_in, n_out); return true;
            case 4: p100_q8_0_smem_vmad_kernel<4, 2><<<blk2, 256, smem_q8, s>>>(w, x, y, n_in, n_out); return true;
        }
    }
    return false;
}

// ============================================================================
// GP100 K-Major SoA Q8_0 (108) and Q6_K (114) MMVQ kernels (436 - 582 GB/s!)
// ============================================================================
__device__ __forceinline__ __half2 soa_u32_to_h2(uint32_t x) { return *reinterpret_cast<const __half2*>(&x); }
__device__ __forceinline__ __half2 soa_u8_lo_h2(uint32_t u) {
    return __hsub2(soa_u32_to_h2(__byte_perm(u, 0x64006400u, 0x5150)), __float2half2_rn(1152.0f));
}
__device__ __forceinline__ __half2 soa_u8_hi_h2(uint32_t u) {
    return __hsub2(soa_u32_to_h2(__byte_perm(u, 0x64006400u, 0x7372)), __float2half2_rn(1152.0f));
}
__device__ __forceinline__ __half2 soa_q6_lo_h2(uint32_t u6) {
    return __hsub2(soa_u32_to_h2(__byte_perm(u6, 0x64006400u, 0x5150)), __float2half2_rn(1056.0f));
}
__device__ __forceinline__ __half2 soa_q6_hi_h2(uint32_t u6) {
    return __hsub2(soa_u32_to_h2(__byte_perm(u6, 0x64006400u, 0x7372)), __float2half2_rn(1056.0f));
}
__device__ __forceinline__ __half2 soa_q81_pair_h2(const int8_t* qs2, __half2 c1152, __half2 inv128) {
    const uint16_t pair = *reinterpret_cast<const uint16_t*>(qs2);
    const uint32_t vx = ((uint32_t) pair) ^ 0x8080u;
    const uint32_t h = __byte_perm(vx, 0x64006400u, 0x5150);
    return __hmul2(__hsub2(soa_u32_to_h2(h), c1152), inv128);
}

template<int NT, int SK, int TILE_B = 16>
__global__ void __launch_bounds__(128) p100_q80_soa_km_kernel(const uint8_t* __restrict__ weights,
                                                              const Q81Block* __restrict__ x,
                                                              float* __restrict__ y,
                                                              int n_in, int n_out) {
    const int K_blk = n_in >> 5;
    const int kpart = (SK == 1) ? 0 : (int) blockIdx.y;
    const int b_start = (SK == 1) ? 0 : ((K_blk * kpart) / SK);
    const int b_end   = (SK == 1) ? K_blk : ((K_blk * (kpart + 1)) / SK);
    const int tid = threadIdx.x;
    const int r = (int) blockIdx.x * 128 + tid;

    const uint4* Wq = reinterpret_cast<const uint4*>(weights);
    const __half* Wd = reinterpret_cast<const __half*>(weights + (std::size_t) n_in * (std::size_t) n_out);

    __shared__ __align__(16) __half2 sxh[NT][TILE_B * 16];
    __shared__ float sxd[NT][TILE_B];
    const __half2 c1152 = __float2half2_rn(1152.0f);
    const __half2 inv128 = __float2half2_rn(1.0f / 128.0f);

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

    for (int b0 = b_start; b0 < b_end; b0 += TILE_B) {
        const int nb = (b0 + TILE_B <= b_end) ? TILE_B : (b_end - b0);
        for (int i = tid; i < nb * 8; i += 128) {
            const int db = i >> 3;
            const int j = i & 7;
            const int b = b0 + db;
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const Q81Block& bx = x[(std::size_t) k * K_blk + b];
                const uint32_t q4 = __ldg(reinterpret_cast<const uint32_t*>(bx.qs) + j) ^ 0x80808080u;
                sxh[k][i * 2 + 0] = __hmul2(__hsub2(soa_u32_to_h2(__byte_perm(q4, 0x64006400u, 0x5150)), c1152), inv128);
                sxh[k][i * 2 + 1] = __hmul2(__hsub2(soa_u32_to_h2(__byte_perm(q4, 0x64006400u, 0x7372)), c1152), inv128);
                if (j == 0) sxd[k][db] = __low2float(bx.ds) * 128.0f;
            }
        }
        __syncthreads();
        if (r < n_out) {
            for (int db = 0; db < nb; ++db) {
                const int b = b0 + db;
                const float d = __half2float(__ldg(Wd + (std::size_t) b * n_out + r));
                __half2 p[NT];
#pragma unroll
                for (int k = 0; k < NT; ++k) p[k] = __float2half2_rn(0.0f);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint4 qv = __ldg(Wq + (std::size_t) (b * 2 + h) * n_out + r);
                    const int xb = db * 16 + h * 8;
                    const __half2 w0 = soa_u8_lo_h2(qv.x), w1 = soa_u8_hi_h2(qv.x);
                    const __half2 w2 = soa_u8_lo_h2(qv.y), w3 = soa_u8_hi_h2(qv.y);
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(&sxh[k][xb + 0]);
                        __half2 pk = p[k];
                        pk = __hfma2(w0, soa_u32_to_h2(xv.x), pk); pk = __hfma2(w1, soa_u32_to_h2(xv.y), pk);
                        pk = __hfma2(w2, soa_u32_to_h2(xv.z), pk); pk = __hfma2(w3, soa_u32_to_h2(xv.w), pk);
                        p[k] = pk;
                    }
                    const __half2 w4 = soa_u8_lo_h2(qv.z), w5 = soa_u8_hi_h2(qv.z);
                    const __half2 w6 = soa_u8_lo_h2(qv.w), w7 = soa_u8_hi_h2(qv.w);
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(&sxh[k][xb + 4]);
                        __half2 pk = p[k];
                        pk = __hfma2(w4, soa_u32_to_h2(xv.x), pk); pk = __hfma2(w5, soa_u32_to_h2(xv.y), pk);
                        pk = __hfma2(w6, soa_u32_to_h2(xv.z), pk); pk = __hfma2(w7, soa_u32_to_h2(xv.w), pk);
                        p[k] = pk;
                    }
                }
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    const float2 f = __half22float2(p[k]);
                    acc[k] = fmaf(d * sxd[k][db], f.x + f.y, acc[k]);
                }
            }
        }
        __syncthreads();
    }
    if (r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if constexpr (SK == 1) y[(std::size_t) k * n_out + r] = acc[k];
            else atomicAdd(y + (std::size_t) k * n_out + r, acc[k]);
        }
    }
}

template<int NT, int SK>
__global__ void __launch_bounds__(64) p100_q6k_soa_km_kernel(const uint8_t* __restrict__ weights,
                                                             const Q81Block* __restrict__ x,
                                                             float* __restrict__ y,
                                                             int n_in, int n_out) {
    const int K_blk = n_in >> 8; // n_in / 256
    const int K_q8  = n_in >> 5; // n_in / 32 = K_blk * 8
    const int kpart = (SK == 1) ? 0 : (int) blockIdx.y;
    const int b_start = (SK == 1) ? 0 : ((K_blk * kpart) / SK);
    const int b_end   = (SK == 1) ? K_blk : ((K_blk * (kpart + 1)) / SK);
    const int tid = threadIdx.x;
    const int r = (int) blockIdx.x * 64 + tid;

    const uint4* Wql = reinterpret_cast<const uint4*>(weights);
    const uint4* Wqh = Wql + (std::size_t) K_blk * 8 * n_out;
    const uint4* Wsc = Wqh + (std::size_t) K_blk * 4 * n_out;
    const __half* Wd = reinterpret_cast<const __half*>(Wsc + (std::size_t) K_blk * n_out);

    __shared__ __half2 sxh[NT][128];
    __shared__ float sxd[NT][8];
    const __half2 c1152 = __float2half2_rn(1152.0f);
    const __half2 inv128 = __float2half2_rn(1.0f / 128.0f);

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

    for (int b = b_start; b < b_end; ++b) {
        {
            const int qb = tid >> 3; // 0..7
            const int j = tid & 7;
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const Q81Block& bx = x[(std::size_t) k * K_q8 + b * 8 + qb];
                const uint32_t q4 = __ldg(reinterpret_cast<const uint32_t*>(bx.qs) + j) ^ 0x80808080u;
                sxh[k][tid * 2 + 0] = __hmul2(__hsub2(soa_u32_to_h2(__byte_perm(q4, 0x64006400u, 0x5150)), c1152), inv128);
                sxh[k][tid * 2 + 1] = __hmul2(__hsub2(soa_u32_to_h2(__byte_perm(q4, 0x64006400u, 0x7372)), c1152), inv128);
                if (j == 0) sxd[k][qb] = __low2float(bx.ds) * 128.0f;
            }
        }
        __syncthreads();
        if (r < n_out) {
            const float d = __half2float(__ldg(Wd + (std::size_t) b * n_out + r));
            const uint4 scv = __ldg(Wsc + (std::size_t) b * n_out + r);
            const int8_t* sc = reinterpret_cast<const int8_t*>(&scv);
            float ba[NT];
#pragma unroll
            for (int k = 0; k < NT; ++k) ba[k] = 0.0f;

#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const uint4 ql_a = __ldg(Wql + (std::size_t) (b * 8 + hf * 4 + s) * n_out + r);
                    const uint4 ql_b = __ldg(Wql + (std::size_t) (b * 8 + hf * 4 + 2 + s) * n_out + r);
                    const uint4 qhv  = __ldg(Wqh + (std::size_t) (b * 4 + hf * 2 + s) * n_out + r);
                    const uint32_t a4[4] = {ql_a.x, ql_a.y, ql_a.z, ql_a.w};
                    const uint32_t b4[4] = {ql_b.x, ql_b.y, ql_b.z, ql_b.w};
                    const uint32_t h4[4] = {qhv.x,  qhv.y,  qhv.z,  qhv.w};
                    const int sc_base = hf * 8 + s;
                    const float s1 = float(sc[sc_base + 0]);
                    const float s2 = float(sc[sc_base + 2]);
                    const float s3 = float(sc[sc_base + 4]);
                    const float s4 = float(sc[sc_base + 6]);
                    __half2 p1[NT], p2[NT], p3[NT], p4[NT];
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        p1[k] = __float2half2_rn(0.0f); p2[k] = __float2half2_rn(0.0f);
                        p3[k] = __float2half2_rn(0.0f); p4[k] = __float2half2_rn(0.0f);
                    }
                    const int xb = hf * 64 + s * 8;
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const uint32_t la = a4[u], lb = b4[u], hv = h4[u];
                        const uint32_t q1 = (la & 0x0f0f0f0fu)        | ((hv & 0x03030303u) << 4);
                        const uint32_t q2 = (lb & 0x0f0f0f0fu)        | ((hv & 0x0c0c0c0cu) << 2);
                        const uint32_t q3 = ((la >> 4) & 0x0f0f0f0fu) | (hv & 0x30303030u);
                        const uint32_t q4 = ((lb >> 4) & 0x0f0f0f0fu) | ((hv >> 2) & 0x30303030u);
                        const __half2 w1a = soa_q6_lo_h2(q1), w1b = soa_q6_hi_h2(q1);
                        const __half2 w2a = soa_q6_lo_h2(q2), w2b = soa_q6_hi_h2(q2);
                        const __half2 w3a = soa_q6_lo_h2(q3), w3b = soa_q6_hi_h2(q3);
                        const __half2 w4a = soa_q6_lo_h2(q4), w4b = soa_q6_hi_h2(q4);
                        const int xo = xb + u * 2;
#pragma unroll
                        for (int k = 0; k < NT; ++k) {
                            p1[k] = __hfma2(w1a, sxh[k][xo +  0], p1[k]); p1[k] = __hfma2(w1b, sxh[k][xo +  1], p1[k]);
                            p2[k] = __hfma2(w2a, sxh[k][xo + 16], p2[k]); p2[k] = __hfma2(w2b, sxh[k][xo + 17], p2[k]);
                            p3[k] = __hfma2(w3a, sxh[k][xo + 32], p3[k]); p3[k] = __hfma2(w3b, sxh[k][xo + 33], p3[k]);
                            p4[k] = __hfma2(w4a, sxh[k][xo + 48], p4[k]); p4[k] = __hfma2(w4b, sxh[k][xo + 49], p4[k]);
                        }
                    }
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        const float2 f1 = __half22float2(p1[k]);
                        const float2 f2 = __half22float2(p2[k]);
                        const float2 f3 = __half22float2(p3[k]);
                        const float2 f4 = __half22float2(p4[k]);
                        float v = ba[k];
                        v = fmaf(s1 * sxd[k][hf * 4 + 0], f1.x + f1.y, v);
                        v = fmaf(s2 * sxd[k][hf * 4 + 1], f2.x + f2.y, v);
                        v = fmaf(s3 * sxd[k][hf * 4 + 2], f3.x + f3.y, v);
                        v = fmaf(s4 * sxd[k][hf * 4 + 3], f4.x + f4.y, v);
                        ba[k] = v;
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < NT; ++k) acc[k] = fmaf(d, ba[k], acc[k]);
        }
        __syncthreads();
    }
    if (r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if constexpr (SK == 1) y[(std::size_t) k * n_out + r] = acc[k];
            else atomicAdd(y + (std::size_t) k * n_out + r, acc[k]);
        }
    }
}

// Block-internal Split-K Q8_0 K-Major SoA: 32 rows/block, NW warps split K inside block ->
// ZERO cudaMemsetAsync, ZERO atomicAdd, 100% deterministic, 100% coalesced LDG.E.128!
// TILE_B=80 preloads all K_blk<=80 in 1 pass (zero mid-loop __syncthreads); TILE_B=32 tiles larger K.
template<int NT, int NW, int TILE_B>
__global__ void __launch_bounds__(NW * 32) p100_q80_soa_bsk_kernel(const uint8_t* __restrict__ weights,
                                                                   const Q81Block* __restrict__ x,
                                                                   float* __restrict__ y,
                                                                   int n_in, int n_out) {
    const int K_blk = n_in >> 5;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r = (int) blockIdx.x * 32 + lane;
    const uint4* Wq = reinterpret_cast<const uint4*>(weights);
    const __half* Wd = reinterpret_cast<const __half*>(weights + (std::size_t) n_in * (std::size_t) n_out);

    union BskSmem {
        struct {
            __align__(16) __half2 sxh[NT][TILE_B * 16];
            float sxd[NT][TILE_B];
        } x;
        float part[NW][NT][32];
    };
    __shared__ BskSmem smem;
    const __half2 c1152 = __float2half2_rn(1152.0f);
    const __half2 inv128 = __float2half2_rn(1.0f / 128.0f);

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

    for (int b0 = 0; b0 < K_blk; b0 += TILE_B) {
        const int nb = (b0 + TILE_B <= K_blk) ? TILE_B : (K_blk - b0);
        for (int i = tid; i < nb * 8; i += NW * 32) {
            const int db = i >> 3, j = i & 7;
            const int b = b0 + db;
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const Q81Block& bx = x[(std::size_t) k * K_blk + b];
                const uint32_t q4 = __ldg(reinterpret_cast<const uint32_t*>(bx.qs) + j) ^ 0x80808080u;
                smem.x.sxh[k][i * 2 + 0] = __hmul2(__hsub2(soa_u32_to_h2(__byte_perm(q4, 0x64006400u, 0x5150)), c1152), inv128);
                smem.x.sxh[k][i * 2 + 1] = __hmul2(__hsub2(soa_u32_to_h2(__byte_perm(q4, 0x64006400u, 0x7372)), c1152), inv128);
                if (j == 0) smem.x.sxd[k][db] = __low2float(bx.ds) * 128.0f;
            }
        }
        __syncthreads();
        if (r < n_out) {
            for (int db = warp; db < nb; db += NW) {
                const int b = b0 + db;
                const float d = __half2float(__ldg(Wd + (std::size_t) b * n_out + r));
                __half2 p[NT];
#pragma unroll
                for (int k = 0; k < NT; ++k) p[k] = __float2half2_rn(0.0f);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint4 qv = __ldg(Wq + (std::size_t) (b * 2 + h) * n_out + r);
                    const int xb = db * 16 + h * 8;
                    const __half2 w0 = soa_u8_lo_h2(qv.x), w1 = soa_u8_hi_h2(qv.x);
                    const __half2 w2 = soa_u8_lo_h2(qv.y), w3 = soa_u8_hi_h2(qv.y);
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 0]);
                        __half2 pk = p[k];
                        pk = __hfma2(w0, soa_u32_to_h2(xv.x), pk); pk = __hfma2(w1, soa_u32_to_h2(xv.y), pk);
                        pk = __hfma2(w2, soa_u32_to_h2(xv.z), pk); pk = __hfma2(w3, soa_u32_to_h2(xv.w), pk);
                        p[k] = pk;
                    }
                    const __half2 w4 = soa_u8_lo_h2(qv.z), w5 = soa_u8_hi_h2(qv.z);
                    const __half2 w6 = soa_u8_lo_h2(qv.w), w7 = soa_u8_hi_h2(qv.w);
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 4]);
                        __half2 pk = p[k];
                        pk = __hfma2(w4, soa_u32_to_h2(xv.x), pk); pk = __hfma2(w5, soa_u32_to_h2(xv.y), pk);
                        pk = __hfma2(w6, soa_u32_to_h2(xv.z), pk); pk = __hfma2(w7, soa_u32_to_h2(xv.w), pk);
                        p[k] = pk;
                    }
                }
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    const float2 f = __half22float2(p[k]);
                    acc[k] = fmaf(d * smem.x.sxd[k][db], f.x + f.y, acc[k]);
                }
            }
        }
        __syncthreads();
    }
    if (r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) smem.part[warp][k][lane] = acc[k];
    }
    __syncthreads();
    if (warp == 0 && r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            float s = smem.part[0][k][lane];
#pragma unroll
            for (int w = 1; w < NW; ++w) s += smem.part[w][k][lane];
            y[(std::size_t) k * n_out + r] = s;
        }
    }
}

// 16 rows/block K-Major SoA Q8_0 (2*NW half-warps split K, 16*16B=256B 100% coalesced LDG.E.128):
// doubles SM utilization on skinny/medium n_out (k_proj/v_proj 512, ssm_out/o_proj/fc 2560)!
template<int NT, int NW, int TILE_B>
__global__ void __launch_bounds__(NW * 32) p100_q80_soa_bsk16_kernel(const uint8_t* __restrict__ weights,
                                                                     const Q81Block* __restrict__ x,
                                                                     float* __restrict__ y,
                                                                     int n_in, int n_out) {
    constexpr int NHW = NW * 2;
    const int K_blk = n_in >> 5;
    const int tid = threadIdx.x, hl = tid & 15, hw = tid >> 4;
    const int r = (int) blockIdx.x * 16 + hl;
    const uint4* Wq = reinterpret_cast<const uint4*>(weights);
    const __half* Wd = reinterpret_cast<const __half*>(weights + (std::size_t) n_in * (std::size_t) n_out);

    union Bsk16Smem {
        struct {
            __align__(16) __half2 sxh[NT][TILE_B * 16];
            float sxd[NT][TILE_B];
        } x;
        float part[NHW][NT][16];
    };
    __shared__ Bsk16Smem smem;
    const __half2 c1152 = __float2half2_rn(1152.0f);
    const __half2 inv128 = __float2half2_rn(1.0f / 128.0f);

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

    for (int b0 = 0; b0 < K_blk; b0 += TILE_B) {
        const int nb = (b0 + TILE_B <= K_blk) ? TILE_B : (K_blk - b0);
        for (int i = tid; i < nb * 8; i += NW * 32) {
            const int db = i >> 3, j = i & 7;
            const int b = b0 + db;
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const Q81Block& bx = x[(std::size_t) k * K_blk + b];
                const uint32_t q4 = __ldg(reinterpret_cast<const uint32_t*>(bx.qs) + j) ^ 0x80808080u;
                smem.x.sxh[k][i * 2 + 0] = __hmul2(__hsub2(soa_u32_to_h2(__byte_perm(q4, 0x64006400u, 0x5150)), c1152), inv128);
                smem.x.sxh[k][i * 2 + 1] = __hmul2(__hsub2(soa_u32_to_h2(__byte_perm(q4, 0x64006400u, 0x7372)), c1152), inv128);
                if (j == 0) smem.x.sxd[k][db] = __low2float(bx.ds) * 128.0f;
            }
        }
        __syncthreads();
        if (r < n_out) {
            for (int db = hw; db < nb; db += NHW) {
                const int b = b0 + db;
                const float d = __half2float(__ldg(Wd + (std::size_t) b * n_out + r));
                __half2 p[NT];
#pragma unroll
                for (int k = 0; k < NT; ++k) p[k] = __float2half2_rn(0.0f);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint4 qv = __ldg(Wq + (std::size_t) (b * 2 + h) * n_out + r);
                    const int xb = db * 16 + h * 8;
                    const __half2 w0 = soa_u8_lo_h2(qv.x), w1 = soa_u8_hi_h2(qv.x);
                    const __half2 w2 = soa_u8_lo_h2(qv.y), w3 = soa_u8_hi_h2(qv.y);
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 0]);
                        __half2 pk = p[k];
                        pk = __hfma2(w0, soa_u32_to_h2(xv.x), pk); pk = __hfma2(w1, soa_u32_to_h2(xv.y), pk);
                        pk = __hfma2(w2, soa_u32_to_h2(xv.z), pk); pk = __hfma2(w3, soa_u32_to_h2(xv.w), pk);
                        p[k] = pk;
                    }
                    const __half2 w4 = soa_u8_lo_h2(qv.z), w5 = soa_u8_hi_h2(qv.z);
                    const __half2 w6 = soa_u8_lo_h2(qv.w), w7 = soa_u8_hi_h2(qv.w);
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 4]);
                        __half2 pk = p[k];
                        pk = __hfma2(w4, soa_u32_to_h2(xv.x), pk); pk = __hfma2(w5, soa_u32_to_h2(xv.y), pk);
                        pk = __hfma2(w6, soa_u32_to_h2(xv.z), pk); pk = __hfma2(w7, soa_u32_to_h2(xv.w), pk);
                        p[k] = pk;
                    }
                }
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    const float2 f = __half22float2(p[k]);
                    acc[k] = fmaf(d * smem.x.sxd[k][db], f.x + f.y, acc[k]);
                }
            }
        }
        __syncthreads();
    }
    if (r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) smem.part[hw][k][hl] = acc[k];
    }
    __syncthreads();
    if (hw == 0 && r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            float s = smem.part[0][k][hl];
#pragma unroll
            for (int w = 1; w < NHW; ++w) s += smem.part[w][k][hl];
            y[(std::size_t) k * n_out + r] = s;
        }
    }
}

// Block-internal Split-K Q6_K K-Major SoA: 32 rows/block, NW warps each own a distinct Q6_K block ->
// ZERO cudaMemsetAsync, ZERO atomicAdd, ZERO block-sync in inner loop!
template<int NT, int NW>
__global__ void __launch_bounds__(NW * 32) p100_q6k_soa_bsk_kernel(const uint8_t* __restrict__ weights,
                                                                   const Q81Block* __restrict__ x,
                                                                   float* __restrict__ y,
                                                                   int n_in, int n_out) {
    const int K_blk = n_in >> 8;
    const int K_q8  = n_in >> 5;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r = (int) blockIdx.x * 32 + lane;
    const uint4* Wql = reinterpret_cast<const uint4*>(weights);
    const uint4* Wqh = Wql + (std::size_t) K_blk * 8 * n_out;
    const uint4* Wsc = Wqh + (std::size_t) K_blk * 4 * n_out;
    const __half* Wd = reinterpret_cast<const __half*>(Wsc + (std::size_t) K_blk * n_out);

    __shared__ __half2 sxh[NW][NT][128];
    __shared__ float sxd[NW][NT][8];
    __shared__ float part[NW][NT][32];
    const __half2 c1152 = __float2half2_rn(1152.0f);
    const __half2 inv128 = __float2half2_rn(1.0f / 128.0f);

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

    for (int b0 = 0; b0 < K_blk; b0 += NW) {
        const int nb = (b0 + NW <= K_blk) ? NW : (K_blk - b0);
        if (warp < nb) {
            const int b = b0 + warp;
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                const int i = it * 32 + lane;
                const int qb = i >> 4, j = i & 15;
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    const Q81Block& bx = x[(std::size_t) k * K_q8 + b * 8 + qb];
                    sxh[warp][k][i] = soa_q81_pair_h2(bx.qs + 2 * j, c1152, inv128);
                    if (j == 0) sxd[warp][k][qb] = __low2float(bx.ds) * 128.0f;
                }
            }
            __syncwarp();
            if (r < n_out) {
                const float d = __half2float(__ldg(Wd + (std::size_t) b * n_out + r));
                const uint4 scv = __ldg(Wsc + (std::size_t) b * n_out + r);
                const int8_t* sc = reinterpret_cast<const int8_t*>(&scv);
                float ba[NT];
#pragma unroll
                for (int k = 0; k < NT; ++k) ba[k] = 0.0f;
#pragma unroll
                for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        const uint4 ql_a = __ldg(Wql + (std::size_t) (b * 8 + hf * 4 + s) * n_out + r);
                        const uint4 ql_b = __ldg(Wql + (std::size_t) (b * 8 + hf * 4 + 2 + s) * n_out + r);
                        const uint4 qhv  = __ldg(Wqh + (std::size_t) (b * 4 + hf * 2 + s) * n_out + r);
                        const uint32_t a4[4] = {ql_a.x, ql_a.y, ql_a.z, ql_a.w};
                        const uint32_t b4[4] = {ql_b.x, ql_b.y, ql_b.z, ql_b.w};
                        const uint32_t h4[4] = {qhv.x,  qhv.y,  qhv.z,  qhv.w};
                        const int sc_base = hf * 8 + s;
                        const float s1 = float(sc[sc_base + 0]), s2 = float(sc[sc_base + 2]);
                        const float s3 = float(sc[sc_base + 4]), s4 = float(sc[sc_base + 6]);
                        __half2 p1[NT], p2[NT], p3[NT], p4[NT];
#pragma unroll
                        for (int k = 0; k < NT; ++k) {
                            p1[k] = __float2half2_rn(0.0f); p2[k] = __float2half2_rn(0.0f);
                            p3[k] = __float2half2_rn(0.0f); p4[k] = __float2half2_rn(0.0f);
                        }
                        const int xb = hf * 64 + s * 8;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const uint32_t la = a4[u], lb = b4[u], hv = h4[u];
                            const uint32_t q1 = (la & 0x0f0f0f0fu)        | ((hv & 0x03030303u) << 4);
                            const uint32_t q2 = (lb & 0x0f0f0f0fu)        | ((hv & 0x0c0c0c0cu) << 2);
                            const uint32_t q3 = ((la >> 4) & 0x0f0f0f0fu) | (hv & 0x30303030u);
                            const uint32_t q4 = ((lb >> 4) & 0x0f0f0f0fu) | ((hv >> 2) & 0x30303030u);
                            const __half2 w1a = soa_q6_lo_h2(q1), w1b = soa_q6_hi_h2(q1);
                            const __half2 w2a = soa_q6_lo_h2(q2), w2b = soa_q6_hi_h2(q2);
                            const __half2 w3a = soa_q6_lo_h2(q3), w3b = soa_q6_hi_h2(q3);
                            const __half2 w4a = soa_q6_lo_h2(q4), w4b = soa_q6_hi_h2(q4);
                            const int xo = xb + u * 2;
#pragma unroll
                            for (int k = 0; k < NT; ++k) {
                                p1[k] = __hfma2(w1a, sxh[warp][k][xo +  0], p1[k]); p1[k] = __hfma2(w1b, sxh[warp][k][xo +  1], p1[k]);
                                p2[k] = __hfma2(w2a, sxh[warp][k][xo + 16], p2[k]); p2[k] = __hfma2(w2b, sxh[warp][k][xo + 17], p2[k]);
                                p3[k] = __hfma2(w3a, sxh[warp][k][xo + 32], p3[k]); p3[k] = __hfma2(w3b, sxh[warp][k][xo + 33], p3[k]);
                                p4[k] = __hfma2(w4a, sxh[warp][k][xo + 48], p4[k]); p4[k] = __hfma2(w4b, sxh[warp][k][xo + 49], p4[k]);
                            }
                        }
#pragma unroll
                        for (int k = 0; k < NT; ++k) {
                            const float2 f1 = __half22float2(p1[k]), f2 = __half22float2(p2[k]);
                            const float2 f3 = __half22float2(p3[k]), f4 = __half22float2(p4[k]);
                            float v = ba[k];
                            v = fmaf(s1 * sxd[warp][k][hf * 4 + 0], f1.x + f1.y, v);
                            v = fmaf(s2 * sxd[warp][k][hf * 4 + 1], f2.x + f2.y, v);
                            v = fmaf(s3 * sxd[warp][k][hf * 4 + 2], f3.x + f3.y, v);
                            v = fmaf(s4 * sxd[warp][k][hf * 4 + 3], f4.x + f4.y, v);
                            ba[k] = v;
                        }
                    }
                }
#pragma unroll
                for (int k = 0; k < NT; ++k) acc[k] = fmaf(d, ba[k], acc[k]);
            }
        }
    }
    if (r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) part[warp][k][lane] = acc[k];
    }
    __syncthreads();
    if (warp == 0 && r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            float s = part[0][k][lane];
#pragma unroll
            for (int w = 1; w < NW; ++w) s += part[w][k][lane];
            y[(std::size_t) k * n_out + r] = s;
        }
    }
}

// Dual-output Q8_0 SoA BSK kernels: fuse two matrices sharing the same input x (e.g. GDN qkv+z, QSA k+v)
// into a single launch to fill all 56 GP100 SMs and eliminate tail waves + kernel launch overhead!
template<int NT, int NW, int TILE_B>
__global__ void __launch_bounds__(NW * 32) p100_q80_soa_bsk_dual_kernel(
    const uint8_t* __restrict__ wA, const uint8_t* __restrict__ wB,
    const Q81Block* __restrict__ x, float* __restrict__ yA, float* __restrict__ yB,
    int n_in, int n_out_A, int n_out_B, int gxA) {
    const bool is_b = (int) blockIdx.x >= gxA;
    const uint8_t* weights = is_b ? wB : wA;
    float* y = is_b ? yB : yA;
    const int n_out = is_b ? n_out_B : n_out_A;
    const int bx_local = is_b ? ((int) blockIdx.x - gxA) : (int) blockIdx.x;
    const int K_blk = n_in >> 5;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r = bx_local * 32 + lane;
    const uint4* Wq = reinterpret_cast<const uint4*>(weights);
    const __half* Wd = reinterpret_cast<const __half*>(weights + (std::size_t) n_in * (std::size_t) n_out);

    union BskDualSmem {
        struct {
            __align__(16) __half2 sxh[NT][TILE_B * 16];
            float sxd[NT][TILE_B];
        } x;
        float part[NW][NT][32];
    };
    __shared__ BskDualSmem smem;
    const __half2 c1152 = __float2half2_rn(1152.0f);
    const __half2 inv128 = __float2half2_rn(1.0f / 128.0f);

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

    for (int b0 = 0; b0 < K_blk; b0 += TILE_B) {
        const int nb = (b0 + TILE_B <= K_blk) ? TILE_B : (K_blk - b0);
        for (int i = tid; i < nb * 8; i += NW * 32) {
            const int db = i >> 3, j = i & 7;
            const int b = b0 + db;
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const Q81Block& bx = x[(std::size_t) k * K_blk + b];
                const uint32_t q4 = __ldg(reinterpret_cast<const uint32_t*>(bx.qs) + j) ^ 0x80808080u;
                smem.x.sxh[k][i * 2 + 0] = __hmul2(__hsub2(soa_u32_to_h2(__byte_perm(q4, 0x64006400u, 0x5150)), c1152), inv128);
                smem.x.sxh[k][i * 2 + 1] = __hmul2(__hsub2(soa_u32_to_h2(__byte_perm(q4, 0x64006400u, 0x7372)), c1152), inv128);
                if (j == 0) smem.x.sxd[k][db] = __low2float(bx.ds) * 128.0f;
            }
        }
        __syncthreads();
        if (r < n_out) {
            for (int db = warp; db < nb; db += NW) {
                const int b = b0 + db;
                const float d = __half2float(__ldg(Wd + (std::size_t) b * n_out + r));
                __half2 p[NT];
#pragma unroll
                for (int k = 0; k < NT; ++k) p[k] = __float2half2_rn(0.0f);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint4 qv = __ldg(Wq + (std::size_t) (b * 2 + h) * n_out + r);
                    const int xb = db * 16 + h * 8;
                    const __half2 w0 = soa_u8_lo_h2(qv.x), w1 = soa_u8_hi_h2(qv.x);
                    const __half2 w2 = soa_u8_lo_h2(qv.y), w3 = soa_u8_hi_h2(qv.y);
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 0]);
                        __half2 pk = p[k];
                        pk = __hfma2(w0, soa_u32_to_h2(xv.x), pk); pk = __hfma2(w1, soa_u32_to_h2(xv.y), pk);
                        pk = __hfma2(w2, soa_u32_to_h2(xv.z), pk); pk = __hfma2(w3, soa_u32_to_h2(xv.w), pk);
                        p[k] = pk;
                    }
                    const __half2 w4 = soa_u8_lo_h2(qv.z), w5 = soa_u8_hi_h2(qv.z);
                    const __half2 w6 = soa_u8_lo_h2(qv.w), w7 = soa_u8_hi_h2(qv.w);
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 4]);
                        __half2 pk = p[k];
                        pk = __hfma2(w4, soa_u32_to_h2(xv.x), pk); pk = __hfma2(w5, soa_u32_to_h2(xv.y), pk);
                        pk = __hfma2(w6, soa_u32_to_h2(xv.z), pk); pk = __hfma2(w7, soa_u32_to_h2(xv.w), pk);
                        p[k] = pk;
                    }
                }
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    const float2 f = __half22float2(p[k]);
                    acc[k] = fmaf(d * smem.x.sxd[k][db], f.x + f.y, acc[k]);
                }
            }
        }
        __syncthreads();
    }
    if (r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) smem.part[warp][k][lane] = acc[k];
    }
    __syncthreads();
    if (warp == 0 && r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            float s = smem.part[0][k][lane];
#pragma unroll
            for (int w = 1; w < NW; ++w) s += smem.part[w][k][lane];
            y[(std::size_t) k * n_out + r] = s;
        }
    }
}

template<int NT, int NW, int TILE_B>
__global__ void __launch_bounds__(NW * 32) p100_q80_soa_bsk16_dual_kernel(
    const uint8_t* __restrict__ wA, const uint8_t* __restrict__ wB,
    const Q81Block* __restrict__ x, float* __restrict__ yA, float* __restrict__ yB,
    int n_in, int n_out_A, int n_out_B, int gxA) {
    constexpr int NHW = NW * 2;
    const bool is_b = (int) blockIdx.x >= gxA;
    const uint8_t* weights = is_b ? wB : wA;
    float* y = is_b ? yB : yA;
    const int n_out = is_b ? n_out_B : n_out_A;
    const int bx_local = is_b ? ((int) blockIdx.x - gxA) : (int) blockIdx.x;
    const int K_blk = n_in >> 5;
    const int tid = threadIdx.x, hl = tid & 15, hw = tid >> 4;
    const int r = bx_local * 16 + hl;
    const uint4* Wq = reinterpret_cast<const uint4*>(weights);
    const __half* Wd = reinterpret_cast<const __half*>(weights + (std::size_t) n_in * (std::size_t) n_out);

    union Bsk16DualSmem {
        struct {
            __align__(16) __half2 sxh[NT][TILE_B * 16];
            float sxd[NT][TILE_B];
        } x;
        float part[NHW][NT][16];
    };
    __shared__ Bsk16DualSmem smem;
    const __half2 c1152 = __float2half2_rn(1152.0f);
    const __half2 inv128 = __float2half2_rn(1.0f / 128.0f);

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

    for (int b0 = 0; b0 < K_blk; b0 += TILE_B) {
        const int nb = (b0 + TILE_B <= K_blk) ? TILE_B : (K_blk - b0);
        for (int i = tid; i < nb * 8; i += NW * 32) {
            const int db = i >> 3, j = i & 7;
            const int b = b0 + db;
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const Q81Block& bx = x[(std::size_t) k * K_blk + b];
                const uint32_t q4 = __ldg(reinterpret_cast<const uint32_t*>(bx.qs) + j) ^ 0x80808080u;
                smem.x.sxh[k][i * 2 + 0] = __hmul2(__hsub2(soa_u32_to_h2(__byte_perm(q4, 0x64006400u, 0x5150)), c1152), inv128);
                smem.x.sxh[k][i * 2 + 1] = __hmul2(__hsub2(soa_u32_to_h2(__byte_perm(q4, 0x64006400u, 0x7372)), c1152), inv128);
                if (j == 0) smem.x.sxd[k][db] = __low2float(bx.ds) * 128.0f;
            }
        }
        __syncthreads();
        if (r < n_out) {
            for (int db = hw; db < nb; db += NHW) {
                const int b = b0 + db;
                const float d = __half2float(__ldg(Wd + (std::size_t) b * n_out + r));
                __half2 p[NT];
#pragma unroll
                for (int k = 0; k < NT; ++k) p[k] = __float2half2_rn(0.0f);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint4 qv = __ldg(Wq + (std::size_t) (b * 2 + h) * n_out + r);
                    const int xb = db * 16 + h * 8;
                    const __half2 w0 = soa_u8_lo_h2(qv.x), w1 = soa_u8_hi_h2(qv.x);
                    const __half2 w2 = soa_u8_lo_h2(qv.y), w3 = soa_u8_hi_h2(qv.y);
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 0]);
                        __half2 pk = p[k];
                        pk = __hfma2(w0, soa_u32_to_h2(xv.x), pk); pk = __hfma2(w1, soa_u32_to_h2(xv.y), pk);
                        pk = __hfma2(w2, soa_u32_to_h2(xv.z), pk); pk = __hfma2(w3, soa_u32_to_h2(xv.w), pk);
                        p[k] = pk;
                    }
                    const __half2 w4 = soa_u8_lo_h2(qv.z), w5 = soa_u8_hi_h2(qv.z);
                    const __half2 w6 = soa_u8_lo_h2(qv.w), w7 = soa_u8_hi_h2(qv.w);
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 4]);
                        __half2 pk = p[k];
                        pk = __hfma2(w4, soa_u32_to_h2(xv.x), pk); pk = __hfma2(w5, soa_u32_to_h2(xv.y), pk);
                        pk = __hfma2(w6, soa_u32_to_h2(xv.z), pk); pk = __hfma2(w7, soa_u32_to_h2(xv.w), pk);
                        p[k] = pk;
                    }
                }
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    const float2 f = __half22float2(p[k]);
                    acc[k] = fmaf(d * smem.x.sxd[k][db], f.x + f.y, acc[k]);
                }
            }
        }
        __syncthreads();
    }
    if (r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) smem.part[hw][k][hl] = acc[k];
    }
    __syncthreads();
    if (hw == 0 && r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            float s = smem.part[0][k][hl];
#pragma unroll
            for (int w = 1; w < NHW; ++w) s += smem.part[w][k][hl];
            y[(std::size_t) k * n_out + r] = s;
        }
    }
}

template<int NT>
void launch_q80_soa_km_nt(const uint8_t* w, const Q81Block* x, float* y, int n_in, int n_out, cudaStream_t s) {
    const int K_blk = n_in >> 5;
    if (n_out >= 16384) {
        const unsigned gx = unsigned((n_out + 127) / 128);
        if constexpr (NT >= 2 && NT <= 5) {
            if ((K_blk % 40) == 0) {
                p100_q80_soa_km_kernel<NT, 1, 40><<<dim3(gx, 1), 128, 0, s>>>(w, x, y, n_in, n_out);
                return;
            }
        }
        p100_q80_soa_km_kernel<NT, 1, 16><<<dim3(gx, 1), 128, 0, s>>>(w, x, y, n_in, n_out);
        return;
    }
    if (n_out <= 2560 && (n_out & 15) == 0) {
        const unsigned gx16 = unsigned(n_out >> 4);
        if ((K_blk & 31) == 0) {
            p100_q80_soa_bsk16_kernel<NT, 8, 32><<<gx16, 256, 0, s>>>(w, x, y, n_in, n_out);
            return;
        }
        if ((K_blk & 15) == 0) {
            p100_q80_soa_bsk16_kernel<NT, 8, 16><<<gx16, 256, 0, s>>>(w, x, y, n_in, n_out);
            return;
        }
        p100_q80_soa_bsk16_kernel<NT, 4, 16><<<gx16, 128, 0, s>>>(w, x, y, n_in, n_out);
        return;
    }
    const unsigned gx = unsigned((n_out + 31) / 32);
    const bool tile40 = (NT <= 7) && ((K_blk % 40) == 0);
    if constexpr (NT <= 7) {
        if (tile40) { p100_q80_soa_bsk_kernel<NT, 8, 40><<<gx, 256, 0, s>>>(w, x, y, n_in, n_out); return; }
    }
    p100_q80_soa_bsk_kernel<NT, 8, 32><<<gx, 256, 0, s>>>(w, x, y, n_in, n_out);
}

template<int NT>
void launch_q80_soa_dual_nt(const uint8_t* wA, const uint8_t* wB, const Q81Block* x,
                            float* yA, float* yB, int n_in, int n_out_A, int n_out_B, cudaStream_t s) {
    const int K_blk = n_in >> 5;
    if (n_out_A <= 2560 && n_out_B <= 2560 && (n_out_A & 15) == 0 && (n_out_B & 15) == 0) {
        const int gxA = n_out_A >> 4, gxB = n_out_B >> 4;
        if ((K_blk & 31) == 0) {
            p100_q80_soa_bsk16_dual_kernel<NT, 8, 32><<<unsigned(gxA + gxB), 256, 0, s>>>(
                wA, wB, x, yA, yB, n_in, n_out_A, n_out_B, gxA);
            return;
        }
        if ((K_blk & 15) == 0) {
            p100_q80_soa_bsk16_dual_kernel<NT, 8, 16><<<unsigned(gxA + gxB), 256, 0, s>>>(
                wA, wB, x, yA, yB, n_in, n_out_A, n_out_B, gxA);
            return;
        }
        p100_q80_soa_bsk16_dual_kernel<NT, 4, 16><<<unsigned(gxA + gxB), 128, 0, s>>>(
            wA, wB, x, yA, yB, n_in, n_out_A, n_out_B, gxA);
    } else {
        const int gxA = (n_out_A + 31) / 32, gxB = (n_out_B + 31) / 32;
        const bool tile40 = (NT <= 7) && ((K_blk % 40) == 0);
        if constexpr (NT <= 7) {
            if (tile40) {
                p100_q80_soa_bsk_dual_kernel<NT, 8, 40><<<unsigned(gxA + gxB), 256, 0, s>>>(
                    wA, wB, x, yA, yB, n_in, n_out_A, n_out_B, gxA);
                return;
            }
        }
        p100_q80_soa_bsk_dual_kernel<NT, 8, 32><<<unsigned(gxA + gxB), 256, 0, s>>>(
            wA, wB, x, yA, yB, n_in, n_out_A, n_out_B, gxA);
    }
}

template<int NT>
void launch_q6k_soa_km_nt(const uint8_t* w, const Q81Block* x, float* y, int n_in, int n_out, cudaStream_t s) {
    const int K_blk = n_in >> 8;
    if (n_out >= 16384) {
        if constexpr (NT == 1) {
            if (K_blk >= 5) {
                const unsigned gx32 = unsigned((n_out + 31) / 32);
                p100_q6k_soa_bsk_kernel<1, 5><<<gx32, 160, 0, s>>>(w, x, y, n_in, n_out);
                return;
            }
        }
        const unsigned gx = unsigned((n_out + 63) / 64);
        p100_q6k_soa_km_kernel<NT, 1><<<dim3(gx, 1), 64, 0, s>>>(w, x, y, n_in, n_out);
        return;
    }
    const unsigned gx = unsigned((n_out + 31) / 32);
    if (K_blk >= 5) {
        p100_q6k_soa_bsk_kernel<NT, 5><<<gx, 160, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        p100_q6k_soa_bsk_kernel<NT, 4><<<gx, 128, 0, s>>>(w, x, y, n_in, n_out);
    }
}

void native_q8_0_soa_mmvq(const void* weights, const void* x_q8_1, float* y,
                          int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights); validate_pointer(x_q8_1); validate_pointer(y); validate_stream(stream);
    const auto* w = static_cast<const uint8_t*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const int K_blk = n_in >> 5;
    for (int c0 = 0; c0 < ncols; ) {
        const int rem = ncols - c0;
        const int nt = rem >= 8 ? 8 : rem;
        const Q81Block* xc = x + (std::size_t) c0 * K_blk;
        float* yc = y + (std::size_t) c0 * n_out;
        switch (nt) {
            case 1: launch_q80_soa_km_nt<1>(w, xc, yc, n_in, n_out, s); break;
            case 2: launch_q80_soa_km_nt<2>(w, xc, yc, n_in, n_out, s); break;
            case 3: launch_q80_soa_km_nt<3>(w, xc, yc, n_in, n_out, s); break;
            case 4: launch_q80_soa_km_nt<4>(w, xc, yc, n_in, n_out, s); break;
            case 5: launch_q80_soa_km_nt<5>(w, xc, yc, n_in, n_out, s); break;
            case 6: launch_q80_soa_km_nt<6>(w, xc, yc, n_in, n_out, s); break;
            case 7: launch_q80_soa_km_nt<7>(w, xc, yc, n_in, n_out, s); break;
            default: launch_q80_soa_km_nt<8>(w, xc, yc, n_in, n_out, s); break;
        }
        c0 += nt;
    }
    launch_check();
}

void native_q6_k_soa_mmvq(const void* weights, const void* x_q8_1, float* y,
                          int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights); validate_pointer(x_q8_1); validate_pointer(y); validate_stream(stream);
    const auto* w = static_cast<const uint8_t*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const int K_blk = n_in >> 5;
    for (int c0 = 0; c0 < ncols; ) {
        const int rem = ncols - c0;
        const int nt = rem >= 8 ? 8 : rem;
        const Q81Block* xc = x + (std::size_t) c0 * K_blk;
        float* yc = y + (std::size_t) c0 * n_out;
        switch (nt) {
            case 1: launch_q6k_soa_km_nt<1>(w, xc, yc, n_in, n_out, s); break;
            case 2: launch_q6k_soa_km_nt<2>(w, xc, yc, n_in, n_out, s); break;
            case 3: launch_q6k_soa_km_nt<3>(w, xc, yc, n_in, n_out, s); break;
            case 4: launch_q6k_soa_km_nt<4>(w, xc, yc, n_in, n_out, s); break;
            case 5: launch_q6k_soa_km_nt<5>(w, xc, yc, n_in, n_out, s); break;
            case 6: launch_q6k_soa_km_nt<6>(w, xc, yc, n_in, n_out, s); break;
            case 7: launch_q6k_soa_km_nt<7>(w, xc, yc, n_in, n_out, s); break;
            default: launch_q6k_soa_km_nt<8>(w, xc, yc, n_in, n_out, s); break;
        }
        c0 += nt;
    }
    launch_check();
}

} // namespace

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    native_quantize_q8_1_kernel<<<blocks, QUANT_THREADS, 0,
                                 static_cast<cudaStream_t>(stream)>>>(x, static_cast<Q81Block*>(x_q8_1), n_total);
    launch_check();
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q5KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q5KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / QK < VDR * WARPS * WARP / QI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q5_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q5_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q5_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q20Traits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q20Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 64 < WARPS * WARP / 2) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q2_0_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q2_0_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q2_0_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q3KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q3KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q3_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q3_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q3_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<IQ4XSTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < 4 * WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_iq4_xs_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_iq4_xs_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_iq4_xs_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q4KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q4KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q4_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q4_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q4_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (p100_mmvq_try_q6k(weights, x_q8_1, y, n_in, n_out, ncols, static_cast<cudaStream_t>(stream))) {
        launch_check();
        return;
    }
    if (ncols > 1) {
        launch_multi<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q6_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q6_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q6_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q40Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q40Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q50Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q50Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (p100_mmvq_try_q80(weights, x_q8_1, y, n_in, n_out, ncols, static_cast<cudaStream_t>(stream))) {
        launch_check();
        return;
    }
    small_mmvq<Q80Block, 8>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q80Block, 8>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<IQ4NLBlock, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<IQ4NLBlock, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool p100_soa_enabled() noexcept {
    static const bool on = [] {
        const char* v = std::getenv("MAYA_P100_SOA");
        if (v != nullptr) return std::atoi(v) != 0;
        return p100_is_sm60();
    }();
    return on;
}

int native_repack_to_soa_host(int ggml_type, const void* src, void* dst,
                              int n_in, int n_out, bool row_major) {
    if (!src || !dst || n_in <= 0 || n_out <= 0 || (n_out % 32 != 0)) return ggml_type;
    if (ggml_type == 8 && (n_in % 32 == 0)) {
        const int K_blk = n_in / 32;
        const auto* s = static_cast<const Q80Block*>(src);
        auto* wq = static_cast<uint8_t*>(dst);
        auto* wd = reinterpret_cast<uint16_t*>(wq + (std::size_t) n_in * (std::size_t) n_out);
        if (row_major) {
            for (int r = 0; r < n_out; ++r) {
                for (int b = 0; b < K_blk; ++b) {
                    const Q80Block& blk = s[(std::size_t) r * K_blk + b];
                    uint8_t* dq = wq + ((std::size_t) r * n_in + (std::size_t) b * 32);
                    for (int i = 0; i < 32; ++i) dq[i] = static_cast<uint8_t>(blk.qs[i]) ^ 0x80u;
                    std::memcpy(wd + (std::size_t) r * K_blk + b, &blk.d, 2);
                }
            }
            return kGgmlQ8_0_SOA_RM;
        }
        for (int r = 0; r < n_out; ++r) {
            for (int b = 0; b < K_blk; ++b) {
                const Q80Block& blk = s[(std::size_t) r * K_blk + b];
                uint8_t* dq0 = wq + (((std::size_t) (b * 2 + 0) * n_out + r) * 16);
                uint8_t* dq1 = wq + (((std::size_t) (b * 2 + 1) * n_out + r) * 16);
                for (int i = 0; i < 16; ++i) dq0[i] = static_cast<uint8_t>(blk.qs[i]) ^ 0x80u;
                for (int i = 0; i < 16; ++i) dq1[i] = static_cast<uint8_t>(blk.qs[16 + i]) ^ 0x80u;
                std::memcpy(wd + (std::size_t) b * n_out + r, &blk.d, 2);
            }
        }
        return kGgmlQ8_0_SOA;
    }
    if (ggml_type == 14 && (n_in % 256 == 0) && !row_major) {
        const int K_blk = n_in / 256;
        const auto* s = static_cast<const Q6KBlock*>(src);
        auto* base = static_cast<uint8_t*>(dst);
        uint8_t* wql = base;
        uint8_t* wqh = wql + (std::size_t) K_blk * 128 * n_out;
        uint8_t* wsc = wqh + (std::size_t) K_blk * 64 * n_out;
        auto* wd = reinterpret_cast<uint16_t*>(wsc + (std::size_t) K_blk * 16 * n_out);
        for (int r = 0; r < n_out; ++r) {
            for (int b = 0; b < K_blk; ++b) {
                const Q6KBlock& blk = s[(std::size_t) r * K_blk + b];
                for (int q = 0; q < 8; ++q)
                    std::memcpy(wql + (((std::size_t) (b * 8 + q) * n_out + r) * 16), blk.ql + q * 16, 16);
                for (int q = 0; q < 4; ++q)
                    std::memcpy(wqh + (((std::size_t) (b * 4 + q) * n_out + r) * 16), blk.qh + q * 16, 16);
                std::memcpy(wsc + (((std::size_t) b * n_out + r) * 16), blk.scales, 16);
                std::memcpy(wd + (std::size_t) b * n_out + r, &blk.d, 2);
            }
        }
        return kGgmlQ6_K_SOA;
    }
    return ggml_type;
}

namespace {
__global__ void gather_q6k_soa_rows_kernel(const uint8_t* __restrict__ src, int K_blk, int src_rows,
                                           const int32_t* __restrict__ ids, int dst_rows, uint8_t* __restrict__ dst) {
    const int r = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= dst_rows) return;
    const int sr = ids[r];
    const uint4* s_ql = reinterpret_cast<const uint4*>(src);
    const uint4* s_qh = s_ql + (std::size_t) K_blk * 8 * src_rows;
    const uint4* s_sc = s_qh + (std::size_t) K_blk * 4 * src_rows;
    const __half* s_d = reinterpret_cast<const __half*>(s_sc + (std::size_t) K_blk * src_rows);

    uint4* d_ql = reinterpret_cast<uint4*>(dst);
    uint4* d_qh = d_ql + (std::size_t) K_blk * 8 * dst_rows;
    uint4* d_sc = d_qh + (std::size_t) K_blk * 4 * dst_rows;
    __half* d_d = reinterpret_cast<__half*>(d_sc + (std::size_t) K_blk * dst_rows);

    for (int p = 0; p < K_blk * 8; ++p)
        d_ql[(std::size_t) p * dst_rows + r] = s_ql[(std::size_t) p * src_rows + sr];
    for (int p = 0; p < K_blk * 4; ++p)
        d_qh[(std::size_t) p * dst_rows + r] = s_qh[(std::size_t) p * src_rows + sr];
    for (int p = 0; p < K_blk; ++p) {
        d_sc[(std::size_t) p * dst_rows + r] = s_sc[(std::size_t) p * src_rows + sr];
        d_d[(std::size_t) p * dst_rows + r]  = s_d[(std::size_t) p * src_rows + sr];
    }
}

__global__ void gather_q80_soa_rows_kernel(const uint8_t* __restrict__ src, int K_blk, int src_rows,
                                           const int32_t* __restrict__ ids, int dst_rows, uint8_t* __restrict__ dst) {
    const int r = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= dst_rows) return;
    const int sr = ids[r];
    const uint4* s_q = reinterpret_cast<const uint4*>(src);
    const __half* s_d = reinterpret_cast<const __half*>(s_q + (std::size_t) K_blk * 2 * src_rows);
    uint4* d_q = reinterpret_cast<uint4*>(dst);
    __half* d_d = reinterpret_cast<__half*>(d_q + (std::size_t) K_blk * 2 * dst_rows);

    for (int p = 0; p < K_blk * 2; ++p)
        d_q[(std::size_t) p * dst_rows + r] = s_q[(std::size_t) p * src_rows + sr];
    for (int p = 0; p < K_blk; ++p)
        d_d[(std::size_t) p * dst_rows + r] = s_d[(std::size_t) p * src_rows + sr];
}
}  // namespace

void native_q6_k_soa_gather_rows(const void* src_soa, int n_in, int src_rows,
                                 const int32_t* ids, int dst_rows, void* dst_soa, void* stream) {
    if (dst_rows <= 0) return;
    validate_shape(n_in, 1, 256);
    validate_pointer(src_soa); validate_pointer(ids); validate_pointer(dst_soa);
    const int K_blk = n_in >> 8;
    const unsigned blocks = unsigned((dst_rows + 255) / 256);
    gather_q6k_soa_rows_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t*>(src_soa), K_blk, src_rows, ids, dst_rows, static_cast<uint8_t*>(dst_soa));
    launch_check();
}

void native_q8_0_soa_gather_rows(const void* src_soa, int n_in, int src_rows,
                                 const int32_t* ids, int dst_rows, void* dst_soa, void* stream) {
    if (dst_rows <= 0) return;
    validate_shape(n_in, 1, 32);
    validate_pointer(src_soa); validate_pointer(ids); validate_pointer(dst_soa);
    const int K_blk = n_in >> 5;
    const unsigned blocks = unsigned((dst_rows + 255) / 256);
    gather_q80_soa_rows_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t*>(src_soa), K_blk, src_rows, ids, dst_rows, static_cast<uint8_t*>(dst_soa));
    launch_check();
}

void native_q8_0_soa_mmvq_dual(const void* wA, const void* wB, const void* x_q8_1,
                               float* yA, float* yB, int n_in, int n_out_A, int n_out_B,
                               int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out_A <= 0 || n_out_B <= 0) throw std::invalid_argument("native dual MMVQ requires positive n_out");
    validate_pointer(wA); validate_pointer(wB); validate_pointer(x_q8_1);
    validate_pointer(yA); validate_pointer(yB); validate_stream(stream);
    const auto* a = static_cast<const uint8_t*>(wA);
    const auto* b = static_cast<const uint8_t*>(wB);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const int K_blk = n_in >> 5;
    for (int c0 = 0; c0 < ncols; ) {
        const int rem = ncols - c0;
        const int nt = rem >= 8 ? 8 : rem;
        const Q81Block* xc = x + (std::size_t) c0 * K_blk;
        float* yAc = yA + (std::size_t) c0 * n_out_A;
        float* yBc = yB + (std::size_t) c0 * n_out_B;
        switch (nt) {
            case 1: launch_q80_soa_dual_nt<1>(a, b, xc, yAc, yBc, n_in, n_out_A, n_out_B, s); break;
            case 2: launch_q80_soa_dual_nt<2>(a, b, xc, yAc, yBc, n_in, n_out_A, n_out_B, s); break;
            case 3: launch_q80_soa_dual_nt<3>(a, b, xc, yAc, yBc, n_in, n_out_A, n_out_B, s); break;
            case 4: launch_q80_soa_dual_nt<4>(a, b, xc, yAc, yBc, n_in, n_out_A, n_out_B, s); break;
            case 5: launch_q80_soa_dual_nt<5>(a, b, xc, yAc, yBc, n_in, n_out_A, n_out_B, s); break;
            case 6: launch_q80_soa_dual_nt<6>(a, b, xc, yAc, yBc, n_in, n_out_A, n_out_B, s); break;
            case 7: launch_q80_soa_dual_nt<7>(a, b, xc, yAc, yBc, n_in, n_out_A, n_out_B, s); break;
            default: launch_q80_soa_dual_nt<8>(a, b, xc, yAc, yBc, n_in, n_out_A, n_out_B, s); break;
        }
        c0 += nt;
    }
    launch_check();
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29 ||
           ggml_type == kGgmlQ8_0_SOA || ggml_type == kGgmlQ8_0_SOA_RM || ggml_type == kGgmlQ6_K_SOA;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 8: case kGgmlQ8_0_SOA: case kGgmlQ8_0_SOA_RM: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: case kGgmlQ6_K_SOA: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 19: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case kGgmlQ8_0_SOA: native_q8_0_soa_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case kGgmlQ6_K_SOA: native_q6_k_soa_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 19: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

} // namespace strata::kernels
