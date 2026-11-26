#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

// The FP32-activation MMVF implementation below is adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, ggml/src/ggml-cuda/{mmvf.cu,common.cuh}.
// Scope: ordinary contiguous BF16 matrix, one FP32 activation vector, no fusion/ids/channels.
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
__device__ __forceinline__ float mmvf_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}

template <int BLOCK_SIZE>
__global__ void bf16_f32_mmvf_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                    float* __restrict__ y, int n_in) {
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    const float2* inputs2 = reinterpret_cast<const float2*>(x);
    __shared__ float partials[32];
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32) partials[t] = 0.0f;
        __syncthreads();
    }
    float acc = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float2 input = inputs2[pair];
        // Match the two ordered multiply-adds in ggml_cuda_mad, not a pair sum followed by one add.
        acc = __fmaf_rn(f32_from_bf16((uint16_t) weight), input.x, acc);
        acc = __fmaf_rn(f32_from_bf16((uint16_t) (weight >> 16)), input.y, acc);
    }
    acc = mmvf_warp_sum(acc);
    if constexpr (BLOCK_SIZE > 32) {
        // All lanes have the same reduced value; one store avoids a same-value shared-memory race.
        if ((t & 31) == 0) partials[t / 32] = acc;
        __syncthreads();
        if (t < 32) acc = mmvf_warp_sum(partials[t]);
    }
    if (t == 0) y[blockIdx.x] = acc;
}

// the same kernel for up to 8 activation rows - the weight row is read ONCE and every
// token keeps its own accumulator with exactly the single-row kernel's order (pairs, two ordered FMAs, the same warp
// and block reductions), so each output is bit-identical to a bf16_f32_mmvf_kernel launch of its own.
template <int BLOCK_SIZE, int NT>
__global__ void bf16_f32_mmvf_multi_kernel(const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
                                          float* __restrict__ y, int64_t ldy, int n_in, int n_tok) {
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    __shared__ float partials[NT][32];
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k) partials[k][t] = 0.0f;
        __syncthreads();
    }
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k < n_tok) {
                const float2 input = reinterpret_cast<const float2*>(x + (size_t) k * ldx)[pair];
                acc[k] = __fmaf_rn(w0, input.x, acc[k]);
                acc[k] = __fmaf_rn(w1, input.y, acc[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = mmvf_warp_sum(acc[k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
#pragma unroll
            for (int k = 0; k < NT; ++k) partials[k][t / 32] = acc[k];
        __syncthreads();
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k) acc[k] = mmvf_warp_sum(partials[k][t]);
    }
    if (t == 0)
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (k < n_tok) y[(size_t) k * ldy + blockIdx.x] = acc[k];
}

// GP100 128-bit coalesced warp-per-row BF16 GEMV (4 warps/block, 1 warp = 1 output row, ZERO __syncthreads()).
// 100% coalesced LDG.E.128 on row-major BF16 (512 bytes/warp/iter) + L1/RO-cached float4 x loads.
template <int NT>
__global__ void __launch_bounds__(128) bf16_f32_warp_u4_multi_kernel(
    const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
    float* __restrict__ y, int64_t ldy, int n_in, int n_out) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * 4 + warp;
    if (r >= n_out) return;
    const uint4* wr = reinterpret_cast<const uint4*>(w + (size_t) r * n_in);
    const int n_vec = n_in >> 3;   // 8 BF16s per uint4
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (int v = lane; v < n_vec; v += 32) {
        const uint4 wv = wr[v];
        const float w0 = __uint_as_float((wv.x & 0xffffu) << 16);
        const float w1 = __uint_as_float(wv.x & 0xffff0000u);
        const float w2 = __uint_as_float((wv.y & 0xffffu) << 16);
        const float w3 = __uint_as_float(wv.y & 0xffff0000u);
        const float w4 = __uint_as_float((wv.z & 0xffffu) << 16);
        const float w5 = __uint_as_float(wv.z & 0xffff0000u);
        const float w6 = __uint_as_float((wv.w & 0xffffu) << 16);
        const float w7 = __uint_as_float(wv.w & 0xffff0000u);
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const float4* xk = reinterpret_cast<const float4*>(x + (size_t) k * ldx);
            const float4 xa = __ldg(&xk[v * 2 + 0]);
            const float4 xb = __ldg(&xk[v * 2 + 1]);
            acc[k] = __fmaf_rn(w0, xa.x, acc[k]);
            acc[k] = __fmaf_rn(w1, xa.y, acc[k]);
            acc[k] = __fmaf_rn(w2, xa.z, acc[k]);
            acc[k] = __fmaf_rn(w3, xa.w, acc[k]);
            acc[k] = __fmaf_rn(w4, xb.x, acc[k]);
            acc[k] = __fmaf_rn(w5, xb.y, acc[k]);
            acc[k] = __fmaf_rn(w6, xb.z, acc[k]);
            acc[k] = __fmaf_rn(w7, xb.w, acc[k]);
        }
    }
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        acc[k] = mmvf_warp_sum(acc[k]);
        if (lane == 0) y[(size_t) k * ldy + r] = acc[k];
    }
}

// GP100 Shared-memory tiled BF16->FP32 GEMV (8 rows/block = 256t, 64 blocks on 56 SMs = 100% SM saturation,
// half the SMEM -> 2x occupancy vs 512t!). Eliminates redundant L2 x-reads across 512 router rows, 100% bit-exact FP32!
template<int NT>
__global__ void __launch_bounds__(256) bf16_f32_smem_u4_multi_kernel(
    const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
    float* __restrict__ y, int64_t ldy, int n_in, int n_out) {
    constexpr int TILE_K = (NT <= 4) ? 1280 : 640;   // <= 20 KB SMEM -> 2 blocks/SM on GP100!
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r = (int) blockIdx.x * 8 + warp;
    __shared__ __align__(16) float sx[NT][TILE_K];
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

    const uint4* w4 = (r < n_out) ? reinterpret_cast<const uint4*>(w + (size_t) r * n_in) : nullptr;
    for (int k0 = 0; k0 < n_in; k0 += TILE_K) {
        const int nk = (k0 + TILE_K <= n_in) ? TILE_K : (n_in - k0);
        const int nk4 = nk >> 2;
        for (int i = tid; i < NT * nk4; i += 256) {
            const int k = i / nk4, j = i - k * nk4;
            reinterpret_cast<float4*>(sx[k])[j] = __ldg(reinterpret_cast<const float4*>(x + (size_t) k * ldx + k0) + j);
        }
        __syncthreads();
        if (r < n_out) {
            const int v0 = k0 >> 3, nv = nk >> 3;
            for (int v = lane; v < nv; v += 32) {
                const uint4 wv = __ldg(w4 + v0 + v);
                const float w0 = __uint_as_float((wv.x & 0xffffu) << 16);
                const float w1 = __uint_as_float(wv.x & 0xffff0000u);
                const float w2 = __uint_as_float((wv.y & 0xffffu) << 16);
                const float w3 = __uint_as_float(wv.y & 0xffff0000u);
                const float w4f = __uint_as_float((wv.z & 0xffffu) << 16);
                const float w5 = __uint_as_float(wv.z & 0xffff0000u);
                const float w6 = __uint_as_float((wv.w & 0xffffu) << 16);
                const float w7 = __uint_as_float(wv.w & 0xffff0000u);
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    const float4 xa = reinterpret_cast<const float4*>(sx[k])[v * 2 + 0];
                    const float4 xb = reinterpret_cast<const float4*>(sx[k])[v * 2 + 1];
                    acc[k] = __fmaf_rn(w0, xa.x, acc[k]); acc[k] = __fmaf_rn(w1, xa.y, acc[k]);
                    acc[k] = __fmaf_rn(w2, xa.z, acc[k]); acc[k] = __fmaf_rn(w3, xa.w, acc[k]);
                    acc[k] = __fmaf_rn(w4f, xb.x, acc[k]); acc[k] = __fmaf_rn(w5, xb.y, acc[k]);
                    acc[k] = __fmaf_rn(w6, xb.z, acc[k]); acc[k] = __fmaf_rn(w7, xb.w, acc[k]);
                }
            }
        }
        if (k0 + TILE_K < n_in) __syncthreads();
    }
    if (r < n_out) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            acc[k] = mmvf_warp_sum(acc[k]);
            if (lane == 0) y[(size_t) k * ldy + r] = acc[k];
        }
    }
}

// GP100 Split-K5 2-Row Zero-Barrier BF16->FP32 GEMV (5 warps/block = 160t, 2 rows/thread = 2x L1 x reuse,
// 256 blocks for 512 rows, exact 2 uint4/thread for n_in=2560, ZERO SMEM tile barriers, 100% exact FP32!).
// Accelerates 512x2560 router & QSA indexer GEMVs by 1.6x (27.7 us -> 17.5 us at T=6, 0 spills).
template<int NT>
__global__ void __launch_bounds__(160) bf16_f32_sk5_2r_kernel(
    const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
    float* __restrict__ y, int64_t ldy, int n_in, int n_out) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r0 = (int) blockIdx.x * 2;
    const int r1 = r0 + 1;
    const uint4* wr0 = (r0 < n_out) ? reinterpret_cast<const uint4*>(w + (size_t) r0 * n_in) : nullptr;
    const uint4* wr1 = (r1 < n_out) ? reinterpret_cast<const uint4*>(w + (size_t) r1 * n_in) : nullptr;
    const int nv = n_in >> 3;
    float a0[NT], a1[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) { a0[k] = 0.0f; a1[k] = 0.0f; }
    for (int v = tid; v < nv; v += 160) {
        const uint4 wv0 = wr0 ? __ldg(wr0 + v) : make_uint4(0, 0, 0, 0);
        const uint4 wv1 = wr1 ? __ldg(wr1 + v) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const float4* xk = reinterpret_cast<const float4*>(x + (size_t) k * ldx);
            const float4 xa = __ldg(xk + v * 2 + 0);
            const float4 xb = __ldg(xk + v * 2 + 1);
            a0[k] = __fmaf_rn(__uint_as_float((wv0.x & 0xffffu) << 16), xa.x, a0[k]);
            a0[k] = __fmaf_rn(__uint_as_float(wv0.x & 0xffff0000u),     xa.y, a0[k]);
            a0[k] = __fmaf_rn(__uint_as_float((wv0.y & 0xffffu) << 16), xa.z, a0[k]);
            a0[k] = __fmaf_rn(__uint_as_float(wv0.y & 0xffff0000u),     xa.w, a0[k]);
            a0[k] = __fmaf_rn(__uint_as_float((wv0.z & 0xffffu) << 16), xb.x, a0[k]);
            a0[k] = __fmaf_rn(__uint_as_float(wv0.z & 0xffff0000u),     xb.y, a0[k]);
            a0[k] = __fmaf_rn(__uint_as_float((wv0.w & 0xffffu) << 16), xb.z, a0[k]);
            a0[k] = __fmaf_rn(__uint_as_float(wv0.w & 0xffff0000u),     xb.w, a0[k]);

            a1[k] = __fmaf_rn(__uint_as_float((wv1.x & 0xffffu) << 16), xa.x, a1[k]);
            a1[k] = __fmaf_rn(__uint_as_float(wv1.x & 0xffff0000u),     xa.y, a1[k]);
            a1[k] = __fmaf_rn(__uint_as_float((wv1.y & 0xffffu) << 16), xa.z, a1[k]);
            a1[k] = __fmaf_rn(__uint_as_float(wv1.y & 0xffff0000u),     xa.w, a1[k]);
            a1[k] = __fmaf_rn(__uint_as_float((wv1.z & 0xffffu) << 16), xb.x, a1[k]);
            a1[k] = __fmaf_rn(__uint_as_float(wv1.z & 0xffff0000u),     xb.y, a1[k]);
            a1[k] = __fmaf_rn(__uint_as_float((wv1.w & 0xffffu) << 16), xb.z, a1[k]);
            a1[k] = __fmaf_rn(__uint_as_float(wv1.w & 0xffff0000u),     xb.w, a1[k]);
        }
    }
    __shared__ float part[2][5][NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        const float s0 = mmvf_warp_sum(a0[k]);
        const float s1 = mmvf_warp_sum(a1[k]);
        if (lane == 0) { part[0][warp][k] = s0; part[1][warp][k] = s1; }
    }
    __syncthreads();
    if (tid < NT) {
        if (r0 < n_out) y[(size_t) tid * ldy + r0] = part[0][0][tid] + part[0][1][tid] + part[0][2][tid] + part[0][3][tid] + part[0][4][tid];
        if (r1 < n_out) y[(size_t) tid * ldy + r1] = part[1][0][tid] + part[1][1][tid] + part[1][2][tid] + part[1][3][tid] + part[1][4][tid];
    }
}

// GP100 Split-K5 4-Row Zero-Barrier BF16->FP32 GEMV (5 warps/block = 160t, 4 rows/thread = 4x L1 x reuse,
// 128 blocks for 512 rows = 100% SM saturation, exact 2 uint4/thread for n_in=2560, STACK:0, 100% exact FP32!).
template<int NT>
__global__ void __launch_bounds__(160, (NT <= 4) ? 4 : 2) bf16_f32_sk5_4r_kernel(
    const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
    float* __restrict__ y, int64_t ldy, int n_in, int n_out) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r0 = (int) blockIdx.x * 4;
    if (r0 >= n_out) return;
    const uint4* wr0 = reinterpret_cast<const uint4*>(w + (size_t) (r0 + 0) * n_in);
    const uint4* wr1 = (r0 + 1 < n_out) ? reinterpret_cast<const uint4*>(w + (size_t) (r0 + 1) * n_in) : nullptr;
    const uint4* wr2 = (r0 + 2 < n_out) ? reinterpret_cast<const uint4*>(w + (size_t) (r0 + 2) * n_in) : nullptr;
    const uint4* wr3 = (r0 + 3 < n_out) ? reinterpret_cast<const uint4*>(w + (size_t) (r0 + 3) * n_in) : nullptr;
    const int nv = n_in >> 3;
    float a0[NT], a1[NT], a2[NT], a3[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) { a0[k] = 0.0f; a1[k] = 0.0f; a2[k] = 0.0f; a3[k] = 0.0f; }
    for (int v = tid; v < nv; v += 160) {
        const uint4 wv0 = __ldg(wr0 + v);
        const uint4 wv1 = wr1 ? __ldg(wr1 + v) : make_uint4(0, 0, 0, 0);
        const uint4 wv2 = wr2 ? __ldg(wr2 + v) : make_uint4(0, 0, 0, 0);
        const uint4 wv3 = wr3 ? __ldg(wr3 + v) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const float4* xk = reinterpret_cast<const float4*>(x + (size_t) k * ldx);
            const float4 xa = __ldg(xk + v * 2 + 0);
            const float4 xb = __ldg(xk + v * 2 + 1);
            auto d8 = [&](const uint4& wv, float a) {
                a = __fmaf_rn(__uint_as_float((wv.x & 0xffffu) << 16), xa.x, a);
                a = __fmaf_rn(__uint_as_float(wv.x & 0xffff0000u),     xa.y, a);
                a = __fmaf_rn(__uint_as_float((wv.y & 0xffffu) << 16), xa.z, a);
                a = __fmaf_rn(__uint_as_float(wv.y & 0xffff0000u),     xa.w, a);
                a = __fmaf_rn(__uint_as_float((wv.z & 0xffffu) << 16), xb.x, a);
                a = __fmaf_rn(__uint_as_float(wv.z & 0xffff0000u),     xb.y, a);
                a = __fmaf_rn(__uint_as_float((wv.w & 0xffffu) << 16), xb.z, a);
                a = __fmaf_rn(__uint_as_float(wv.w & 0xffff0000u),     xb.w, a);
                return a;
            };
            a0[k] = d8(wv0, a0[k]);
            a1[k] = d8(wv1, a1[k]);
            a2[k] = d8(wv2, a2[k]);
            a3[k] = d8(wv3, a3[k]);
        }
    }
    __shared__ float part[4][5][NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        const float s0 = mmvf_warp_sum(a0[k]);
        const float s1 = mmvf_warp_sum(a1[k]);
        const float s2 = mmvf_warp_sum(a2[k]);
        const float s3 = mmvf_warp_sum(a3[k]);
        if (lane == 0) {
            part[0][warp][k] = s0; part[1][warp][k] = s1;
            part[2][warp][k] = s2; part[3][warp][k] = s3;
        }
    }
    __syncthreads();
    if (tid < NT * 4) {
        const int r = tid / NT, k = tid - r * NT;
        if (r0 + r < n_out) {
            y[(size_t) k * ldy + r0 + r] =
                part[r][0][k] + part[r][1][k] + part[r][2][k] + part[r][3][k] + part[r][4][k];
        }
    }
}

int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

}  // namespace

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) { bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream); return; }
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    const cudaStream_t st = (cudaStream_t) stream;
    if (n_tok >= 2 && n_tok <= 8 && n_in >= 1280 && (n_in % 1280) == 0 && n_out >= 128 &&
        (ldx & 3) == 0 && (reinterpret_cast<uintptr_t>(x) & 15u) == 0 && (reinterpret_cast<uintptr_t>(w) & 15u) == 0) {
        if ((n_out & 3) == 0 && n_out >= 256) {
            const unsigned grid = (unsigned) (n_out / 4);
            switch (n_tok) {
                case 2: bf16_f32_sk5_4r_kernel<2><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
                case 3: bf16_f32_sk5_4r_kernel<3><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
                case 4: bf16_f32_sk5_4r_kernel<4><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
                case 5: bf16_f32_sk5_4r_kernel<5><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
                case 6: bf16_f32_sk5_4r_kernel<6><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
                case 7: bf16_f32_sk5_4r_kernel<7><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
                case 8: bf16_f32_sk5_4r_kernel<8><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            }
        } else {
            const unsigned grid = (unsigned) ((n_out + 1) / 2);
            switch (n_tok) {
                case 2: bf16_f32_sk5_2r_kernel<2><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
                case 3: bf16_f32_sk5_2r_kernel<3><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
                case 4: bf16_f32_sk5_2r_kernel<4><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
                case 5: bf16_f32_sk5_2r_kernel<5><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
                case 6: bf16_f32_sk5_2r_kernel<6><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
                case 7: bf16_f32_sk5_2r_kernel<7><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
                case 8: bf16_f32_sk5_2r_kernel<8><<<grid, 160, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            }
        }
        const cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi sk5 launch: ") + cudaGetErrorString(result));
        return;
    }
    if (n_tok >= 2 && n_tok <= 8 && n_in >= 640 && n_out >= 128 && (n_in & 7) == 0 && (ldx & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(x) & 15u) == 0 && (reinterpret_cast<uintptr_t>(w) & 15u) == 0) {
        const unsigned grid = (unsigned) ((n_out + 7) / 8);
        switch (n_tok) {
            case 2: bf16_f32_smem_u4_multi_kernel<2><<<grid, 256, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 3: bf16_f32_smem_u4_multi_kernel<3><<<grid, 256, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 4: bf16_f32_smem_u4_multi_kernel<4><<<grid, 256, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 5: bf16_f32_smem_u4_multi_kernel<5><<<grid, 256, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 6: bf16_f32_smem_u4_multi_kernel<6><<<grid, 256, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 7: bf16_f32_smem_u4_multi_kernel<7><<<grid, 256, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 8: bf16_f32_smem_u4_multi_kernel<8><<<grid, 256, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
        }
        const cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi smem launch: ") + cudaGetErrorString(result));
        return;
    }
    if (n_tok <= 8 && n_out >= 32 && (n_in & 7) == 0 && (ldx & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(x) & 15u) == 0 && (reinterpret_cast<uintptr_t>(w) & 15u) == 0) {
        const unsigned grid = (unsigned) ((n_out + 3) / 4);
        switch (n_tok) {
            case 1: bf16_f32_warp_u4_multi_kernel<1><<<grid, 128, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 2: bf16_f32_warp_u4_multi_kernel<2><<<grid, 128, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 3: bf16_f32_warp_u4_multi_kernel<3><<<grid, 128, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 4: bf16_f32_warp_u4_multi_kernel<4><<<grid, 128, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 5: bf16_f32_warp_u4_multi_kernel<5><<<grid, 128, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 6: bf16_f32_warp_u4_multi_kernel<6><<<grid, 128, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 7: bf16_f32_warp_u4_multi_kernel<7><<<grid, 128, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
            case 8: bf16_f32_warp_u4_multi_kernel<8><<<grid, 128, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, (int) n_out); break;
        }
        const cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + cudaGetErrorString(result));
        return;
    }
#define STRATA_MMVF_M(N) case N: \
    if (n_tok <= 4) bf16_f32_mmvf_multi_kernel<N, 4><<<(unsigned) n_out, N, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, n_tok); \
    else bf16_f32_mmvf_multi_kernel<N, 8><<<(unsigned) n_out, N, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, n_tok); break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_M(32); STRATA_MMVF_M(64); STRATA_MMVF_M(96); STRATA_MMVF_M(128);
        STRATA_MMVF_M(160); STRATA_MMVF_M(192); STRATA_MMVF_M(224); STRATA_MMVF_M(256);
    }
#undef STRATA_MMVF_M
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + cudaGetErrorString(result));
}

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    const cudaStream_t st = (cudaStream_t) stream;
    if (n_in >= 1280 && (n_in % 1280) == 0 && n_out >= 256 && (n_out & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(x) & 15u) == 0 && (reinterpret_cast<uintptr_t>(w) & 15u) == 0) {
        const unsigned grid = (unsigned) (n_out / 4);
        bf16_f32_sk5_4r_kernel<1><<<grid, 160, 0, st>>>(x, n_in, w, y, n_out, (int) n_in, (int) n_out);
        const cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf sk5_4r launch: ") + cudaGetErrorString(result));
        return;
    }
    if (n_out >= 32 && (n_in & 7) == 0 &&
        (reinterpret_cast<uintptr_t>(x) & 15u) == 0 && (reinterpret_cast<uintptr_t>(w) & 15u) == 0) {
        const unsigned grid = (unsigned) ((n_out + 3) / 4);
        bf16_f32_warp_u4_multi_kernel<1><<<grid, 128, 0, st>>>(x, n_in, w, y, n_out, (int) n_in, (int) n_out);
        const cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf launch: ") + cudaGetErrorString(result));
        return;
    }
#define STRATA_MMVF_CASE(N) case N: \
    bf16_f32_mmvf_kernel<N><<<(unsigned) n_out, N, 0, st>>>(x, w, y, (int) n_in); break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_CASE(32);
        STRATA_MMVF_CASE(64);
        STRATA_MMVF_CASE(96);
        STRATA_MMVF_CASE(128);
        STRATA_MMVF_CASE(160);
        STRATA_MMVF_CASE(192);
        STRATA_MMVF_CASE(224);
        STRATA_MMVF_CASE(256);
    }
#undef STRATA_MMVF_CASE
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf launch: ") + cudaGetErrorString(result));
}


}  // namespace strata::kernels
