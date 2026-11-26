// src/kernels/cuda/iq_kernels.cu - see include/strata/kernels/iq_kernels.hpp.
//
// The dot products (vec_dot_*_q8_1), the dequantizers and the q8_1 quantizer are transcribed from llama.cpp
// (ggml/src/ggml-cuda/vecdotq.cuh, dequantize.cuh, quantize.cu at the commit in third_party/ggml/VERSION.txt;
// MIT license, third_party/ggml/LICENSE).  The block structs and codebook grids come from its ggml-common.h,
// included unchanged.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "strata/kernels/dp4a.hpp"   // sm_60: __dp4a -> the vmad fallback

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }
__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
// Pascal sm_60: avoid ~16-instruction emulated __vcmpne4 / __vsub4 (bit-exact for nonzero grid bytes).
__device__ __forceinline__ int iq_sflip4(uint32_t g, uint32_t x4) {
    const uint32_t m1 = ((x4 & 0xFu) * 0x00204081u) & 0x01010101u;
    return (int) ((g ^ (m1 * 255u)) + m1);
}
__device__ __forceinline__ uint32_t iq_ksign8(uint32_t v) {
    v &= 0xFFu;
    return v ^ ((__popc(v) & 1u) << 7);
}
#endif
__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}
#define ggml_cuda_dp4a(a, b, c) __dp4a((a), (b), (c))

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
__device__ __forceinline__ float vec_dot_q2_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xxs_grid)[aux8[k0 / 2]];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
        const uint32_t sb = iq_ksign8(aux32 >> (7 * k0 / 2));
        const int grid0 = iq_sflip4(grid_pos.x, sb);
        const int grid1 = iq_sflip4(grid_pos.y, sb >> 4);
#else
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
#endif
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
        const uint32_t s8 = signs_packed_8[l0 / 2];
        const int grid_l = iq_sflip4((uint32_t) grid_pos[0], s8);
        const int grid_h = iq_sflip4((uint32_t) grid_pos[1], s8 >> 4);
#else
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
#endif
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                        iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq1_m_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

// IQ1_S: one call covers one 32-elem q8_1 chunk (iqs 0..7 within a 256 block); the delta term rides
// the q8_1 block's fp16 sum (ds.y) exactly as vecdotq.cuh does it.
__device__ __forceinline__ float vec_dot_iq1_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq1_s* bq1 = (const block_iq1_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x * sumi + ds.y * delta);
}

__device__ __forceinline__ float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

// Q5_0 (vecdotq.cuh's vec_dot_q5_0_q8_1 + _impl<VDR 2>, unchanged): unsloth's Qwen3.8 UD-IQ2_XS keeps one layer's
// routed down experts (blk.4) in this format.  One call covers half a 32-value block (iqs 0 or 2).
__device__ __forceinline__ float vec_dot_q5_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_0* bq5_0 = (const block_q5_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = get_int_b2(bq5_0->qs, iqs + i);
        const int vh = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i);
        const int u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
    }
    const float d5 = __half2float(bq5_0->d);
    const float2 ds8f = __half22float2(bq8_1->ds);
    return d5 * (sumi * ds8f.x - (16 * 2 / QI5_0) * ds8f.y);
}

// Q5_1: 32 values per 24-byte block (half2 dm, uint8 qh[4], uint8 qs[16]); w = d * q5 + m.
__device__ __forceinline__ float vec_dot_q5_1_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_1* bq5_1 = (const block_q5_1*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = get_int_b4(bq5_1->qs, iqs + i);
        const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i);
        const int u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
    }
    const float2 dm5 = __half22float2(bq5_1->dm);
    const float2 ds8f = __half22float2(bq8_1->ds);
    return dm5.x * (sumi * ds8f.x) + dm5.y * (ds8f.y * (2.0f / QI5_1));
}

// Q8_0: 32 values per 34-byte block (half d, int8 qs[32]); 4 calls of VDR=2 per block.
__device__ __forceinline__ float vec_dot_q8_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q8_0* bq8_0 = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = get_int_b2(bq8_0->qs, iqs + i);
        const int u = get_int_b4(bq8_1->qs, iqs + i);
        sumi = ggml_cuda_dp4a(v, u, sumi);
    }
    return __half2float(bq8_0->d) * __low2float(bq8_1->ds) * (float) sumi;
}

// IQ4_XS: 256 values as 8 sub-blocks of 32 (6-bit scale each); one call covers one sub-block (iqs = 4 * sub-block),
// and `bq8_1` is the super-block's first q8_1 block, so the call's activation is bq8_1[iqs / 4].  Strata's Qwen IQ3_S
// file keeps one layer's routed gate/up experts in this format.
__device__ __forceinline__ float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * sumi;
}

// Q5_K: 256 values, 8 sub-blocks of 32 with 6-bit scales and mins; one call covers 2 x 4 values of two sub-blocks
// (VDR 2).  CYBER-FROST-3.8's routed gate/up experts are Q5_K.  vecdotq.cuh's vec_dot_q5_K_q8_1 and its _impl_vmmq at
// the pinned commit (the branch-free scale unpacking).
__device__ __forceinline__ float vec_dot_q5_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_K* bq5_K = (const block_q5_K*) vbq + kbx;
    int vl[2], vh[2], u[2 * QR5_K];
    float d8[QR5_K];
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int* ql = (const int*) (bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*) (bq5_K->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    const uint16_t* scales = (const uint16_t*) bq5_K->scales;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(bq5_K->dm);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
template<int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<6> { static constexpr int qk = 32, ipb = 2, step = 2;    // Q5_0: QI5_0 4 / VDR 2
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<7> { static constexpr int qk = 32, ipb = 2, step = 2;    // Q5_1: QI5_1 4 / VDR 2
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<8> { static constexpr int qk = 32, ipb = 4, step = 2;    // Q8_0: QI8_0 8 / VDR 2
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q8_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<19> { static constexpr int qk = 256, ipb = 8, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<13> { static constexpr int qk = 256, ipb = 16, step = 2;   // QI5_K 32 / VDR 2
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };
#include "iq_kquant.cuh"   // glm-dsa B3.5: Fmt<10> .. Fmt<14>, the K-quants (Q2_K .. Q6_K) for native_expert_grouped

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One row against one q8_1 activation, the whole warp: call k = (block, part) is lane-strided.
template<int TY>
__device__ __forceinline__ float row_dot(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}

/// One row against NC q8_1 activations at once.
///
/// row_dot() is called once per activation column and re-reads and re-dequantises the SAME weight bytes
/// each time, so a k-column batch paid the weight traffic k times.  Here the block index is the OUTER loop:
/// one block is loaded and decoded once and applied to every column.  The per-lane accumulation order is
/// unchanged (same k, same stride, same warp_sum), so each column is bit-identical to the single-column
/// call - this is a scheduling change, not an arithmetic one (tests/iq_mmvq_parity_test.cu checks it over
/// every supported format and width).
template<int TY, int NC>
__device__ __forceinline__ void row_dot_multi(const uint8_t* row, const block_q8_1* x, int xb, int nb, int lane,
                                              float* s) {
    using F = Fmt<TY>;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const block_q8_1* base = x + (size_t) kbx * (F::qk / 32);
#pragma unroll
        for (int c = 0; c < NC; ++c) s[c] += F::dot(row, base + (size_t) c * (size_t) xb, kbx, iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = warp_sum(s[c]);
}

template<int TY, int NC>
__global__ void __launch_bounds__(128) mmvq_multi_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                         const block_q8_1* __restrict__ x, float* __restrict__ y,
                                                         int n_in, int n_out, int ncols) {
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk;
    const int xb = n_in / 32;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    int c0 = 0;
    for (; c0 + NC <= ncols; c0 += NC) {
        float s[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) s[c] = 0.0f;
        row_dot_multi<TY, NC>(wr, x + (size_t) c0 * (size_t) xb, xb, nb, lane, s);
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < NC; ++c) y[(size_t) (c0 + c) * (size_t) n_out + (size_t) row] = s[c];
        }
    }
    for (; c0 < ncols; ++c0) {                 // the tail: the single-column form
        const float s = row_dot<TY>(wr, x + (size_t) c0 * (size_t) xb, nb, lane);
        if (lane == 0) y[(size_t) c0 * (size_t) n_out + (size_t) row] = s;
    }
}

template<int TY>
__global__ void __launch_bounds__(128) mmvq_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                   const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in,
                                                   int n_out, int ncols);

/// The widest column group a batch supports; a single column keeps the plain kernel.
template<int TY>
inline void launch_mmvq(const uint8_t* W, size_t rb, const block_q8_1* X, float* y, int n_in, int n_out, int ncols,
                        const dim3& grid, const dim3& block, cudaStream_t s) {
    if (ncols >= 8)
        mmvq_multi_kernel<TY, 8><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols);
    else if (ncols >= 4)
        mmvq_multi_kernel<TY, 4><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols);
    else if (ncols >= 2)
        mmvq_multi_kernel<TY, 2><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols);
    else
        mmvq_kernel<TY><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols);
}

template<int TY>
__global__ void __launch_bounds__(128) mmvq_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                   const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in,
                                                   int n_out, int ncols) {
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s = row_dot<TY>(wr, x + (size_t) c * (n_in / 32), nb, lane);
        if (lane == 0) y[(size_t) c * n_out + row] = s;
    }
}

// ---------------------------------------------------------------- grouped native experts
constexpr int GU_ROWS = 8;     // rows per block (one warp each)

template<int TG>
__global__ void __launch_bounds__(256) native_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups,
                                                        const int32_t* __restrict__ ent_tok,
                                                        const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                        float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU_ROWS + warp;             // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s = row_dot<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, lane);
        if (lane == 0) (is_up ? up : gate)[(size_t) e * L.n_ff + r] = s;
    }
}

// GP100 (sm_60) HFMA2 8-lane IQ2_S grouped gate+up kernel: 0% idle lanes, 1-inst PRMT(0x6400) + FP16 sign XOR +
// HFMA2, reads weights ONCE when 2 tokens in a window pick the same expert. 1.36x faster on cold HBM2 (40/48 layers).
__global__ void __launch_bounds__(256) p100_iq2s_gu_hfma2_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                 const int32_t* __restrict__ grp_start,
                                                                 const int32_t* __restrict__ n_groups,
                                                                 const int32_t* __restrict__ ent_tok,
                                                                 const block_q8_1* __restrict__ xq,
                                                                 NativeExpertLayout L,
                                                                 float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    __shared__ __half2 sxh[2][1280];
    __shared__ float sxd[2][80];
    const int tid = threadIdx.x;
    const int row = blockIdx.x * 32 + (tid >> 3);
    const int lane8 = tid & 7;
    const bool valid_row = row < 2 * (int) L.n_ff;
    const bool is_up = row >= (int) L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const block_iq2_s* wr = valid_row ? (const block_iq2_s*) (blob + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row) : nullptr;
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);

    auto stage_tok = [&](int slot, int tok_id) {
        const block_q8_1* xt = xq + (size_t) tok_id * 80;
        for (int i = tid; i < 1280; i += 256) {
            const int blk = i >> 4, sub = (i & 15) << 1;
            const uint16_t pair = *reinterpret_cast<const uint16_t*>(xt[blk].qs + sub);
            const uint32_t vx = ((uint32_t) pair) ^ 0x8080u;
            const uint32_t h = __byte_perm(vx, 0x64006400u, 0x5150);
            sxh[slot][i] = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&h), c1152), inv128);
        }
        if (tid < 80) sxd[slot][tid] = __low2float(xt[tid].ds) * 128.0f;
    };

    for (int eb = e0; eb < e1; eb += 2) {
        const int ne = (eb + 1 < e1) ? 2 : 1;
        stage_tok(0, ent_tok[eb]);
        if (ne == 2) stage_tok(1, ent_tok[eb + 1]);
        __syncthreads();
        if (valid_row) {
            float acc0 = 0.0f, acc1 = 0.0f;
            for (int kbx = 0; kbx < 10; ++kbx) {
                const block_iq2_s* bq2 = wr + kbx;
                const int sb = lane8;
                const int qs_packed = get_int_b2(bq2->qs, sb);
                const uint8_t* qs = (const uint8_t*) &qs_packed;
                const int qh = bq2->qh[sb];
                const int signs_packed_32 = get_int_b2(bq2->qs, 8 + sb);
                const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
                const float ls0 = 0.25f * (0.5f + (float) (bq2->scales[sb] & 0x0F));
                const float ls1 = 0.25f * (0.5f + (float) (bq2->scales[sb] >> 4));
                const __half2* xp0 = sxh[0] + (kbx * 8 + sb) * 16;
                const __half2* xp1 = sxh[1] + (kbx * 8 + sb) * 16;
                __half2 s0a = __floats2half2_rn(0.0f, 0.0f), s1a = __floats2half2_rn(0.0f, 0.0f);
                __half2 s0b = __floats2half2_rn(0.0f, 0.0f), s1b = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                for (int l0 = 0; l0 < 4; ++l0) {
                    const uint2 gp = ((const uint2*) iq2s_grid)[qs[l0] | ((qh << (8 - 2 * l0)) & 0x300)];
                    const uint32_t s8 = signs_packed_8[l0];
                    uint32_t ha = __byte_perm(gp.x, 0x64006400u, 0x5150);
                    uint32_t hb = __byte_perm(gp.x, 0x64006400u, 0x5352);
                    uint32_t hc = __byte_perm(gp.y, 0x64006400u, 0x5150);
                    uint32_t hd = __byte_perm(gp.y, 0x64006400u, 0x5352);
                    __half2 va = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                    __half2 vb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
                    __half2 vc = __hsub2(*reinterpret_cast<const __half2*>(&hc), c1024);
                    __half2 vd = __hsub2(*reinterpret_cast<const __half2*>(&hd), c1024);
                    const uint32_t u_va = (*reinterpret_cast<uint32_t*>(&va)) ^ (((s8 & 1u) << 15) | ((s8 & 2u) << 30));
                    const uint32_t u_vb = (*reinterpret_cast<uint32_t*>(&vb)) ^ (((s8 & 4u) << 13) | ((s8 & 8u) << 28));
                    const uint32_t u_vc = (*reinterpret_cast<uint32_t*>(&vc)) ^ (((s8 & 16u) << 11) | ((s8 & 32u) << 26));
                    const uint32_t u_vd = (*reinterpret_cast<uint32_t*>(&vd)) ^ (((s8 & 64u) << 9) | ((s8 & 128u) << 24));
                    const __half2 wa = *reinterpret_cast<const __half2*>(&u_va);
                    const __half2 wb = *reinterpret_cast<const __half2*>(&u_vb);
                    const __half2 wc = *reinterpret_cast<const __half2*>(&u_vc);
                    const __half2 wd = *reinterpret_cast<const __half2*>(&u_vd);
                    __half2& da = (l0 < 2) ? s0a : s1a;
                    da = __hfma2(wa, xp0[l0 * 4 + 0], da);
                    da = __hfma2(wb, xp0[l0 * 4 + 1], da);
                    da = __hfma2(wc, xp0[l0 * 4 + 2], da);
                    da = __hfma2(wd, xp0[l0 * 4 + 3], da);
                    if (ne == 2) {
                        __half2& db = (l0 < 2) ? s0b : s1b;
                        db = __hfma2(wa, xp1[l0 * 4 + 0], db);
                        db = __hfma2(wb, xp1[l0 * 4 + 1], db);
                        db = __hfma2(wc, xp1[l0 * 4 + 2], db);
                        db = __hfma2(wd, xp1[l0 * 4 + 3], db);
                    }
                }
                const float dw = __half2float(bq2->d);
                const float2 f0a = __half22float2(s0a), f1a = __half22float2(s1a);
                acc0 = fmaf(dw * sxd[0][kbx * 8 + sb], (f0a.x + f0a.y) * ls0 + (f1a.x + f1a.y) * ls1, acc0);
                if (ne == 2) {
                    const float2 f0b = __half22float2(s0b), f1b = __half22float2(s1b);
                    acc1 = fmaf(dw * sxd[1][kbx * 8 + sb], (f0b.x + f0b.y) * ls0 + (f1b.x + f1b.y) * ls1, acc1);
                }
            }
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1);
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2);
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 4);
            if (lane8 == 0) (is_up ? up : gate)[(size_t) eb * L.n_ff + r] = acc0;
            if (ne == 2) {
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1);
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2);
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 4);
                if (lane8 == 0) (is_up ? up : gate)[(size_t) (eb + 1) * L.n_ff + r] = acc1;
            }
        }
        if (eb + 2 < e1) __syncthreads();
    }
}

// GP100 (sm_60) HFMA2 8-lane IQ2_XXS grouped gate+up kernel (8/48 layers, 7 on Stage 0!).
__global__ void __launch_bounds__(256) p100_iq2xxs_gu_hfma2_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                   const int32_t* __restrict__ grp_start,
                                                                   const int32_t* __restrict__ n_groups,
                                                                   const int32_t* __restrict__ ent_tok,
                                                                   const block_q8_1* __restrict__ xq,
                                                                   NativeExpertLayout L,
                                                                   float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    __shared__ __half2 sxh[2][1280];
    __shared__ float sxd[2][80];
    const int tid = threadIdx.x;
    const int row = blockIdx.x * 32 + (tid >> 3);
    const int lane8 = tid & 7;
    const bool valid_row = row < 2 * (int) L.n_ff;
    const bool is_up = row >= (int) L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const block_iq2_xxs* wr = valid_row ? (const block_iq2_xxs*) (blob + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row) : nullptr;
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);

    auto stage_tok = [&](int slot, int tok_id) {
        const block_q8_1* xt = xq + (size_t) tok_id * 80;
        for (int i = tid; i < 1280; i += 256) {
            const int blk = i >> 4, sub = (i & 15) << 1;
            const uint16_t pair = *reinterpret_cast<const uint16_t*>(xt[blk].qs + sub);
            const uint32_t vx = ((uint32_t) pair) ^ 0x8080u;
            const uint32_t h = __byte_perm(vx, 0x64006400u, 0x5150);
            sxh[slot][i] = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&h), c1152), inv128);
        }
        if (tid < 80) sxd[slot][tid] = __low2float(xt[tid].ds) * 128.0f;
    };

    for (int eb = e0; eb < e1; eb += 2) {
        const int ne = (eb + 1 < e1) ? 2 : 1;
        stage_tok(0, ent_tok[eb]);
        if (ne == 2) stage_tok(1, ent_tok[eb + 1]);
        __syncthreads();
        if (valid_row) {
            float acc0 = 0.0f, acc1 = 0.0f;
            for (int kbx = 0; kbx < 10; ++kbx) {
                const block_iq2_xxs* bq2 = wr + kbx;
                const int sb = lane8;
                const int q2 = get_int_b2(bq2->qs, 2 * sb);
                const uint8_t* aux8 = (const uint8_t*) &q2;
                const uint32_t aux32 = get_int_b2(bq2->qs, 2 * sb + 1);
                const float ls = (float) ((aux32 >> 27) | 1u) * 0.125f;
                const __half2* xp0 = sxh[0] + (kbx * 8 + sb) * 16;
                const __half2* xp1 = sxh[1] + (kbx * 8 + sb) * 16;
                __half2 sa = __floats2half2_rn(0.0f, 0.0f);
                __half2 sb2 = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                for (int l0 = 0; l0 < 4; ++l0) {
                    const uint2 gp = ((const uint2*) iq2xxs_grid)[aux8[l0]];
                    uint32_t s7 = (aux32 >> (7 * l0)) & 0x7Fu;
                    const uint32_t s8 = s7 ^ ((__popc(s7) & 1u) << 7);
                    uint32_t ha = __byte_perm(gp.x, 0x64006400u, 0x5150);
                    uint32_t hb = __byte_perm(gp.x, 0x64006400u, 0x5352);
                    uint32_t hc = __byte_perm(gp.y, 0x64006400u, 0x5150);
                    uint32_t hd = __byte_perm(gp.y, 0x64006400u, 0x5352);
                    __half2 va = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                    __half2 vb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
                    __half2 vc = __hsub2(*reinterpret_cast<const __half2*>(&hc), c1024);
                    __half2 vd = __hsub2(*reinterpret_cast<const __half2*>(&hd), c1024);
                    const uint32_t u_va = (*reinterpret_cast<uint32_t*>(&va)) ^ (((s8 & 1u) << 15) | ((s8 & 2u) << 30));
                    const uint32_t u_vb = (*reinterpret_cast<uint32_t*>(&vb)) ^ (((s8 & 4u) << 13) | ((s8 & 8u) << 28));
                    const uint32_t u_vc = (*reinterpret_cast<uint32_t*>(&vc)) ^ (((s8 & 16u) << 11) | ((s8 & 32u) << 26));
                    const uint32_t u_vd = (*reinterpret_cast<uint32_t*>(&vd)) ^ (((s8 & 64u) << 9) | ((s8 & 128u) << 24));
                    const __half2 wa = *reinterpret_cast<const __half2*>(&u_va);
                    const __half2 wb = *reinterpret_cast<const __half2*>(&u_vb);
                    const __half2 wc = *reinterpret_cast<const __half2*>(&u_vc);
                    const __half2 wd = *reinterpret_cast<const __half2*>(&u_vd);
                    sa = __hfma2(wa, xp0[l0 * 4 + 0], sa);
                    sa = __hfma2(wb, xp0[l0 * 4 + 1], sa);
                    sa = __hfma2(wc, xp0[l0 * 4 + 2], sa);
                    sa = __hfma2(wd, xp0[l0 * 4 + 3], sa);
                    if (ne == 2) {
                        sb2 = __hfma2(wa, xp1[l0 * 4 + 0], sb2);
                        sb2 = __hfma2(wb, xp1[l0 * 4 + 1], sb2);
                        sb2 = __hfma2(wc, xp1[l0 * 4 + 2], sb2);
                        sb2 = __hfma2(wd, xp1[l0 * 4 + 3], sb2);
                    }
                }
                const float dls = __half2float(bq2->d) * ls;
                const float2 fa = __half22float2(sa);
                acc0 = fmaf(dls * sxd[0][kbx * 8 + sb], fa.x + fa.y, acc0);
                if (ne == 2) {
                    const float2 fb = __half22float2(sb2);
                    acc1 = fmaf(dls * sxd[1][kbx * 8 + sb], fb.x + fb.y, acc1);
                }
            }
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1);
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2);
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 4);
            if (lane8 == 0) (is_up ? up : gate)[(size_t) eb * L.n_ff + r] = acc0;
            if (ne == 2) {
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1);
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2);
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 4);
                if (lane8 == 0) (is_up ? up : gate)[(size_t) (eb + 1) * L.n_ff + r] = acc1;
            }
        }
        if (eb + 2 < e1) __syncthreads();
    }
}

// Fast 4-lane bit-exact Q2_0 grouped down kernel for n_ff == 640 (23/48 layers -> 1.93x faster on P100!).
// 0% idle lanes (20 sub-blocks / 4 lanes = 5 exact), stages hq (720 B) in smem, reads weights ONCE for pairs!
__global__ void __launch_bounds__(256) p100_q20_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                            const int32_t* __restrict__ grp_start,
                                                            const int32_t* __restrict__ n_groups,
                                                            const int32_t* __restrict__ ent_dst,
                                                            const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                            float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    __shared__ block_q8_1 sxg[2][20];
    const int tid = threadIdx.x;
    const int r = blockIdx.x * 64 + (tid >> 2);
    const int sub = tid & 3;
    const bool valid_r = r < (int) L.n_embd;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const block_q2_0* wr = valid_r ? (const block_q2_0*) (blob + L.down_off + (size_t) r * L.d_row) : nullptr;

    for (int eb = e0; eb < e1; eb += 2) {
        const int ne = (eb + 1 < e1) ? 2 : 1;
        for (int i = tid; i < 20 * 9; i += 256) {
            ((int*) sxg[0])[i] = ((const int*) (hq + (size_t) eb * 20))[i];
            if (ne == 2) ((int*) sxg[1])[i] = ((const int*) (hq + (size_t) (eb + 1) * 20))[i];
        }
        __syncthreads();
        if (valid_r) {
            float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
            for (int k = 0; k < 5; ++k) {
                const int idx = sub + k * 4;
                const int kbx = idx >> 1, iqs = idx & 1;
                const block_q2_0* bq = wr + kbx;
                const int16_t* qs = (const int16_t*) bq->qs + iqs * 4;
                const float d2 = __half2float(bq->d);
                const block_q8_1& bx0 = sxg[0][idx];
                const block_q8_1& bx1 = sxg[1][idx];
                int sumi0 = 0, sumi1 = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int q = qs[j];
                    const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
                    const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
                    const int qx = __byte_perm(qe, qo, 0x5140);
                    const int qy = __byte_perm(qe, qo, 0x7362);
                    sumi0 = ggml_cuda_dp4a(((const int*) bx0.qs)[j * 2 + 0], qx, sumi0);
                    sumi0 = ggml_cuda_dp4a(((const int*) bx0.qs)[j * 2 + 1], qy, sumi0);
                    if (ne == 2) {
                        sumi1 = ggml_cuda_dp4a(((const int*) bx1.qs)[j * 2 + 0], qx, sumi1);
                        sumi1 = ggml_cuda_dp4a(((const int*) bx1.qs)[j * 2 + 1], qy, sumi1);
                    }
                }
                acc0 += d2 * __low2float(bx0.ds) * sumi0;
                if (ne == 2) acc1 += d2 * __low2float(bx1.ds) * sumi1;
            }
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1);
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2);
            if (sub == 0) out[(size_t) ent_dst[eb] * L.n_embd + r] = acc0;
            if (ne == 2) {
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1);
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2);
                if (sub == 0) out[(size_t) ent_dst[eb + 1] * L.n_embd + r] = acc1;
            }
        }
        if (eb + 2 < e1) __syncthreads();
    }
}

// Fast 4-lane IQ4_NL grouped down kernel for n_ff == 640 (24/48 layers = HALF the model!).
// 0% idle lanes (20 blocks / 4 lanes = 5 exact), stages hq for token pairs in smem as __half2(q[j]/128, q[j+16]/128),
// pairs low/high nibbles of qs[j] directly into 16 HFMA2 instructions per 32-element block!
__global__ void __launch_bounds__(256) p100_iq4nl_down_hfma2_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                    const int32_t* __restrict__ grp_start,
                                                                    const int32_t* __restrict__ n_groups,
                                                                    const int32_t* __restrict__ ent_dst,
                                                                    const block_q8_1* __restrict__ hq,
                                                                    NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    __shared__ __half2 sxh[2][20][16];
    __shared__ float sxd[2][20];
    __shared__ __half slut[16];
    const int tid = threadIdx.x;
    if (tid < 16) slut[tid] = __float2half((float) kvalues_iq4nl[tid]);
    const int r = blockIdx.x * 64 + (tid >> 2);
    const int sub = tid & 3;
    const bool valid_r = r < (int) L.n_embd;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const block_iq4_nl* wr = valid_r ? (const block_iq4_nl*) (blob + L.down_off + (size_t) r * L.d_row) : nullptr;

    for (int eb = e0; eb < e1; eb += 2) {
        const int ne = (eb + 1 < e1) ? 2 : 1;
        // 20 blocks x 16 pairs = 320 threads worth of work across 256 threads
        for (int i = tid; i < 20 * 16; i += 256) {
            const int b = i >> 4, j = i & 15;
            const block_q8_1& b0 = hq[(size_t) eb * 20 + b];
            sxh[0][b][j] = __floats2half2_rn((float) b0.qs[j] * (1.0f / 128.0f), (float) b0.qs[j + 16] * (1.0f / 128.0f));
            if (j == 0) sxd[0][b] = __low2float(b0.ds) * 128.0f;
            if (ne == 2) {
                const block_q8_1& b1 = hq[(size_t) (eb + 1) * 20 + b];
                sxh[1][b][j] = __floats2half2_rn((float) b1.qs[j] * (1.0f / 128.0f), (float) b1.qs[j + 16] * (1.0f / 128.0f));
                if (j == 0) sxd[1][b] = __low2float(b1.ds) * 128.0f;
            }
        }
        __syncthreads();
        if (valid_r) {
            float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
            for (int k = 0; k < 5; ++k) {
                const int b = sub + k * 4;
                const block_iq4_nl* bq = wr + b;
                const float d4 = __half2float(__ldg(&bq->d));
                const uint16_t* q16 = reinterpret_cast<const uint16_t*>(bq->qs);
                __half2 h0 = __float2half2_rn(0.0f);
                __half2 h1 = __float2half2_rn(0.0f);
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const uint16_t pair = __ldg(q16 + u);
                    const uint32_t b_lo = pair & 0xffu;
                    const uint32_t b_hi = pair >> 8;
                    const __half2 w0 = __halves2half2(slut[b_lo & 0xfu], slut[b_lo >> 4]);
                    const __half2 w1 = __halves2half2(slut[b_hi & 0xfu], slut[b_hi >> 4]);
                    h0 = __hfma2(w0, sxh[0][b][u * 2 + 0], h0);
                    h0 = __hfma2(w1, sxh[0][b][u * 2 + 1], h0);
                    if (ne == 2) {
                        h1 = __hfma2(w0, sxh[1][b][u * 2 + 0], h1);
                        h1 = __hfma2(w1, sxh[1][b][u * 2 + 1], h1);
                    }
                }
                const float2 f0 = __half22float2(h0);
                acc0 = fmaf(d4 * sxd[0][b], f0.x + f0.y, acc0);
                if (ne == 2) {
                    const float2 f1 = __half22float2(h1);
                    acc1 = fmaf(d4 * sxd[1][b], f1.x + f1.y, acc1);
                }
            }
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1);
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2);
            if (sub == 0) out[(size_t) ent_dst[eb] * L.n_embd + r] = acc0;
            if (ne == 2) {
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1);
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2);
                if (sub == 0) out[(size_t) ent_dst[eb + 1] * L.n_embd + r] = acc1;
            }
        }
        if (eb + 2 < e1) __syncthreads();
    }
}

// Fast 4-lane Q5_0 grouped down kernel for n_ff == 640 (blk.4 on Stage 0): 0% idle lanes (20 blocks / 4 lanes = 5 exact),
// stages hq for token pairs in smem as __half2(q[j]/128, q[j+16]/128), exact PRMT + HSUB2(1040) -> HFMA2 (6x faster than VMAD!).
__global__ void __launch_bounds__(256) p100_q50_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                            const int32_t* __restrict__ grp_start,
                                                            const int32_t* __restrict__ n_groups,
                                                            const int32_t* __restrict__ ent_dst,
                                                            const block_q8_1* __restrict__ hq,
                                                            NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    __shared__ __half2 sxh[2][20][16];
    __shared__ float sxd[2][20];
    const int tid = threadIdx.x;
    const int r = blockIdx.x * 64 + (tid >> 2);
    const int sub = tid & 3;
    const bool valid_r = r < (int) L.n_embd;
    const int K_blk = (int) (L.n_ff / 32);
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const block_q5_0* wr = valid_r ? (const block_q5_0*) (blob + L.down_off + (size_t) r * L.d_row) : nullptr;
    const __half2 c1040 = __floats2half2_rn(1040.0f, 1040.0f);

    for (int eb = e0; eb < e1; eb += 2) {
        const int ne = (eb + 1 < e1) ? 2 : 1;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = (b0 + 20 <= K_blk) ? 20 : (K_blk - b0);
            for (int i = tid; i < nb * 16; i += 256) {
                const int db = i >> 4, j = i & 15;
                const block_q8_1& x0 = hq[(size_t) eb * K_blk + b0 + db];
                sxh[0][db][j] = __floats2half2_rn((float) x0.qs[j] * (1.0f / 128.0f), (float) x0.qs[j + 16] * (1.0f / 128.0f));
                if (j == 0) sxd[0][db] = __low2float(x0.ds) * 128.0f;
                if (ne == 2) {
                    const block_q8_1& x1 = hq[(size_t) (eb + 1) * K_blk + b0 + db];
                    sxh[1][db][j] = __floats2half2_rn((float) x1.qs[j] * (1.0f / 128.0f), (float) x1.qs[j + 16] * (1.0f / 128.0f));
                    if (j == 0) sxd[1][db] = __low2float(x1.ds) * 128.0f;
                }
            }
            __syncthreads();
            if (valid_r) {
                for (int db = sub; db < nb; db += 4) {
                    const block_q5_0* bq = wr + b0 + db;
                    const float d5 = __half2float(__ldg(&bq->d));
                    const uint32_t qh_u = (uint32_t) get_int_b2(bq->qh, 0);
                    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(bq->qs);
                    __half2 h0 = __float2half2_rn(0.0f), h1 = __float2half2_rn(0.0f);
#pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        const uint16_t pair = __ldg(q16 + u);
                        const uint32_t b_lo = pair & 0xffu, b_hi = pair >> 8;
                        const int j0 = u * 2, j1 = u * 2 + 1;
                        const uint32_t p0 = ((b_lo & 0x0fu) | (((qh_u >> j0) & 1u) << 4)) |
                                            (((b_lo >> 4) | (((qh_u >> (j0 + 16)) & 1u) << 4)) << 8);
                        const uint32_t p1 = ((b_hi & 0x0fu) | (((qh_u >> j1) & 1u) << 4)) |
                                            (((b_hi >> 4) | (((qh_u >> (j1 + 16)) & 1u) << 4)) << 8);
                        const uint32_t u0 = __byte_perm(p0, 0x64006400u, 0x5150);
                        const uint32_t u1 = __byte_perm(p1, 0x64006400u, 0x5150);
                        const __half2 w0 = __hsub2(*reinterpret_cast<const __half2*>(&u0), c1040);
                        const __half2 w1 = __hsub2(*reinterpret_cast<const __half2*>(&u1), c1040);
                        h0 = __hfma2(w0, sxh[0][db][j0], h0);
                        h0 = __hfma2(w1, sxh[0][db][j1], h0);
                        if (ne == 2) {
                            h1 = __hfma2(w0, sxh[1][db][j0], h1);
                            h1 = __hfma2(w1, sxh[1][db][j1], h1);
                        }
                    }
                    const float2 f0 = __half22float2(h0);
                    acc0 = fmaf(d5 * sxd[0][db], f0.x + f0.y, acc0);
                    if (ne == 2) {
                        const float2 f1 = __half22float2(h1);
                        acc1 = fmaf(d5 * sxd[1][db], f1.x + f1.y, acc1);
                    }
                }
            }
            if (b0 + 20 < K_blk) __syncthreads();
        }
        if (valid_r) {
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1);
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2);
            if (sub == 0) out[(size_t) ent_dst[eb] * L.n_embd + r] = acc0;
            if (ne == 2) {
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1);
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2);
                if (sub == 0) out[(size_t) ent_dst[eb + 1] * L.n_embd + r] = acc1;
            }
        }
        if (eb + 2 < e1) __syncthreads();
    }
}

// Strict bit-exact 4-lane IQ4_NL grouped down kernel for n_ff == 640 (when MAYA_P100_HFMA2=0).
__global__ void __launch_bounds__(256) p100_iq4nl_down_vmad_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                   const int32_t* __restrict__ grp_start,
                                                                   const int32_t* __restrict__ n_groups,
                                                                   const int32_t* __restrict__ ent_dst,
                                                                   const block_q8_1* __restrict__ hq,
                                                                   NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    __shared__ block_q8_1 sxg[2][20];
    const int tid = threadIdx.x;
    const int r = blockIdx.x * 64 + (tid >> 2);
    const int sub = tid & 3;
    const bool valid_r = r < (int) L.n_embd;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const block_iq4_nl* wr = valid_r ? (const block_iq4_nl*) (blob + L.down_off + (size_t) r * L.d_row) : nullptr;

    for (int eb = e0; eb < e1; eb += 2) {
        const int ne = (eb + 1 < e1) ? 2 : 1;
        for (int i = tid; i < 20 * 9; i += 256) {
            ((int*) sxg[0])[i] = ((const int*) (hq + (size_t) eb * 20))[i];
            if (ne == 2) ((int*) sxg[1])[i] = ((const int*) (hq + (size_t) (eb + 1) * 20))[i];
        }
        __syncthreads();
        if (valid_r) {
            float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
            for (int k = 0; k < 5; ++k) {
                const int b = sub + k * 4;
                const block_iq4_nl* bq = wr + b;
                const float d4 = __half2float(__ldg(&bq->d));
                const block_q8_1& bx0 = sxg[0][b];
                const block_q8_1& bx1 = sxg[1][b];
                int sumi0 = 0, sumi1 = 0;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int aux_q4 = get_int_b2(bq->qs, l);
                    const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
                    sumi0 = ggml_cuda_dp4a(v.x, ((const int*) bx0.qs)[l + 0], sumi0);
                    sumi0 = ggml_cuda_dp4a(v.y, ((const int*) bx0.qs)[l + 4], sumi0);
                    if (ne == 2) {
                        sumi1 = ggml_cuda_dp4a(v.x, ((const int*) bx1.qs)[l + 0], sumi1);
                        sumi1 = ggml_cuda_dp4a(v.y, ((const int*) bx1.qs)[l + 4], sumi1);
                    }
                }
                acc0 += d4 * __low2float(bx0.ds) * sumi0;
                if (ne == 2) acc1 += d4 * __low2float(bx1.ds) * sumi1;
            }
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1);
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2);
            if (sub == 0) out[(size_t) ent_dst[eb] * L.n_embd + r] = acc0;
            if (ne == 2) {
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1);
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2);
                if (sub == 0) out[(size_t) ent_dst[eb + 1] * L.n_embd + r] = acc1;
            }
        }
        if (eb + 2 < e1) __syncthreads();
    }
}

// ---- GP100 K-Major SoA MoE Expert Kernels (Increment 5B) ----
// 100% coalesced 128-bit LDG.E.128 weight loads across warp lanes + Block-Internal Split-K (zero atomicAdd)
// + register-only LUT / bit-extract HFMA2 decode.

__device__ uint4 g_iq2s_h2[1024];

__global__ void init_iq2s_h2_kernel() {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1024) return;
    const uint2 gp = ((const uint2*) iq2s_grid)[i];
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    const uint32_t ha = __byte_perm(gp.x, 0x64006400u, 0x5150);
    const uint32_t hb = __byte_perm(gp.x, 0x64006400u, 0x5352);
    const uint32_t hc = __byte_perm(gp.y, 0x64006400u, 0x5150);
    const uint32_t hd = __byte_perm(gp.y, 0x64006400u, 0x5352);
    const __half2 va = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
    const __half2 vb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
    const __half2 vc = __hsub2(*reinterpret_cast<const __half2*>(&hc), c1024);
    const __half2 vd = __hsub2(*reinterpret_cast<const __half2*>(&hd), c1024);
    uint4 out;
    out.x = *reinterpret_cast<const uint32_t*>(&va);
    out.y = *reinterpret_cast<const uint32_t*>(&vb);
    out.z = *reinterpret_cast<const uint32_t*>(&vc);
    out.w = *reinterpret_cast<const uint32_t*>(&vd);
    g_iq2s_h2[i] = out;
}

__global__ void repack_iq2xxs_soa_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                         int N, int K_blk) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K_blk) return;
    const int r = idx / K_blk;
    const int b = idx % K_blk;
    const uint8_t* sb = src + (size_t) idx * 66;
    uint16_t d_bits;
    memcpy(&d_bits, sb, 2);
    reinterpret_cast<uint16_t*>(dst + (size_t) 64 * K_blk * N)[(size_t) b * N + r] = d_bits;
    uint4* Wq = reinterpret_cast<uint4*>(dst);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        uint4 v;
        memcpy(&v, sb + 2 + q * 16, 16);
        Wq[((size_t) b * 4 + q) * N + r] = v;
    }
}

__global__ void repack_iq2s_soa_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                       int N, int K_blk) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K_blk) return;
    const int r = idx / K_blk;
    const int b = idx % K_blk;
    const uint8_t* sb = src + (size_t) idx * 82;
    uint16_t d_bits;
    memcpy(&d_bits, sb, 2);
    reinterpret_cast<uint16_t*>(dst + (size_t) 80 * K_blk * N)[(size_t) b * N + r] = d_bits;
    uint4* Wqs = reinterpret_cast<uint4*>(dst);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        uint4 v;
        memcpy(&v, sb + 2 + q * 16, 16);
        Wqs[((size_t) b * 4 + q) * N + r] = v;
    }
    uint4 qhsc;
    memcpy(&qhsc, sb + 66, 16);
    reinterpret_cast<uint4*>(dst + (size_t) 64 * K_blk * N)[(size_t) b * N + r] = qhsc;
}

__global__ void repack_18b_soa_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                      int N, int K_blk) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K_blk) return;
    const int r = idx / K_blk;
    const int b = idx % K_blk;
    const uint8_t* sb = src + (size_t) idx * 18;
    uint16_t d_bits;
    memcpy(&d_bits, sb, 2);
    reinterpret_cast<uint16_t*>(dst + (size_t) 16 * K_blk * N)[(size_t) b * N + r] = d_bits;
    uint4 v;
    memcpy(&v, sb + 2, 16);
    reinterpret_cast<uint4*>(dst)[(size_t) b * N + r] = v;
}

__global__ void repack_q4k_soa_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                      int N, int K_blk) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K_blk) return;
    const int r = idx / K_blk, b = idx % K_blk;
    const block_q4_K* s = reinterpret_cast<const block_q4_K*>(src) + idx;
    uint4* Wqs = reinterpret_cast<uint4*>(dst);
    // Fused 16-byte uint4 Wscdm = {scales[0..3], scales[4..7], scales[8..11], dm} -> 1x LDG.E.128!
    uint4* Wscdm = reinterpret_cast<uint4*>(dst + (size_t) 128 * K_blk * N);
    const uint4* sq = reinterpret_cast<const uint4*>(s->qs);
#pragma unroll
    for (int u = 0; u < 8; ++u) Wqs[((size_t) b * 8 + u) * N + r] = sq[u];
    const uint32_t* sc = reinterpret_cast<const uint32_t*>(s->scales);
    Wscdm[(size_t) b * N + r] = make_uint4(sc[0], sc[1], sc[2], *reinterpret_cast<const uint32_t*>(&s->dm));
}

__global__ void repack_q51_soa_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                      int N, int K_blk) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K_blk) return;
    const int r = idx / K_blk, b = idx % K_blk;
    const block_q5_1* s = reinterpret_cast<const block_q5_1*>(src) + idx;
    uint4* Wqs = reinterpret_cast<uint4*>(dst);
    uint2* Wdmqh = reinterpret_cast<uint2*>(dst + (size_t) 16 * K_blk * N);
    uint4 q4;
    memcpy(&q4, s->qs, 16);
    Wqs[(size_t) b * N + r] = q4;
    uint2 h;
    memcpy(&h.x, &s->dm, 4);
    // Wave 6: Bit-transpose qh offline so GP100 unpacks all 4 high bits with 0 XMAD (just 1 shift+AND)!
    // Byte k (0..3) of qh_t holds low-half bits u=0..3 in bits [0..3] and high-half bits u=0..3 in bits [4..7].
    uint32_t qh_orig;
    memcpy(&qh_orig, s->qh, 4);
    uint32_t qh_t = 0u;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int j = u * 4 + k;
            qh_t |= (((qh_orig >> j) & 1u) << (k * 8 + u));
            qh_t |= (((qh_orig >> (16 + j)) & 1u) << (k * 8 + 4 + u));
        }
    }
    h.y = qh_t;
    Wdmqh[(size_t) b * N + r] = h;
}

__global__ void repack_q80_soa_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                      int N, int K_blk) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K_blk) return;
    const int r = idx / K_blk, b = idx % K_blk;
    const block_q8_0* s = reinterpret_cast<const block_q8_0*>(src) + idx;
    uint4* Wqs = reinterpret_cast<uint4*>(dst);
    __half* Wd = reinterpret_cast<__half*>(dst + (size_t) 32 * K_blk * N);
    uint4 q0, q1;
    memcpy(&q0, s->qs, 16);
    memcpy(&q1, s->qs + 16, 16);
    // Wave 6: Pre-XOR by 0x80 offline so all SoA Q8_0 decode/prefill kernels drop runtime ^ 0x80808080u!
    q0.x ^= 0x80808080u; q0.y ^= 0x80808080u; q0.z ^= 0x80808080u; q0.w ^= 0x80808080u;
    q1.x ^= 0x80808080u; q1.y ^= 0x80808080u; q1.z ^= 0x80808080u; q1.w ^= 0x80808080u;
    Wqs[((size_t) b * 2 + 0) * N + r] = q0;
    Wqs[((size_t) b * 2 + 1) * N + r] = q1;
    Wd[(size_t) b * N + r] = s->d;
}

__device__ __forceinline__ uint4 make_sign_mask4(uint32_t sg) {
    uint4 m;
    m.x = ((sg & 1u) << 15) | ((sg & 2u) << 30);
    m.y = ((sg & 4u) << 13) | ((sg & 8u) << 28);
    m.z = ((sg & 16u) << 11) | ((sg & 32u) << 26);
    m.w = ((sg & 64u) << 9) | ((sg & 128u) << 24);
    return m;
}

__device__ __forceinline__ __half2 iq_q81_pair_h2(const int8_t* qs2, __half2 c1152, __half2 inv128) {
    const uint16_t pair = *reinterpret_cast<const uint16_t*>(qs2);
    const uint32_t vx = ((uint32_t) pair) ^ 0x8080u;
    const uint32_t h = __byte_perm(vx, 0x64006400u, 0x5150);
    return __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&h), c1152), inv128);
}

__device__ __forceinline__ __half2 iq_decode_h2(uint32_t gword, uint32_t prmt_sel, uint32_t smask) {
    const uint32_t h = __byte_perm(gword, 0x64006400u, prmt_sel);
    const uint32_t c1024_u = 0x64006400u;
    const __half2 v = __hsub2(*reinterpret_cast<const __half2*>(&h), *reinterpret_cast<const __half2*>(&c1024_u));
    const uint32_t w = (*reinterpret_cast<const uint32_t*>(&v)) ^ smask;
    return *reinterpret_cast<const __half2*>(&w);
}

// IQ2_XXS K-Major SoA Block-Internal Split-K: 64 rows/block x 5 warps = 320 threads, GP100 32-bit bank-split SoA codebook!
// Splits uint2 iq2xxs_grid[256] into sg_lo[256] + sg_hi[256] (2 KB, 2x LDS.32, ZERO LDS.128 bank conflicts, ZERO spills at 64 regs -> 3 blk/SM!).
__global__ void __launch_bounds__(320, 3) p100_iq2xxs_gu_soa_bsk_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok,
    const block_q8_1* __restrict__ xq, NativeExpertLayout L,
    float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    constexpr int NW = 5;
    const int tid = threadIdx.x;
    const int w_all = tid >> 5;
    const int lane = tid & 31;
    const int half = (w_all >= NW) ? 1 : 0;
    const int warp = half ? (w_all - NW) : w_all;
    const int col = half * 32 + lane;
    const int N = 2 * (int) L.n_ff;
    const int K_blk = (int) (L.n_embd / 256);   // 10 for n_embd=2560
    const int row = blockIdx.x * 64 + col;
    const bool valid_row = row < N;
    const bool is_up = row >= (int) L.n_ff;
    const int r_out = is_up ? row - (int) L.n_ff : row;

    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint4* Wq = reinterpret_cast<const uint4*>(blob);
    const __half* Wd = reinterpret_cast<const __half*>(blob + (size_t) 64 * K_blk * N);

    __shared__ uint32_t sg_lo[256], sg_hi[256]; //  2048 B (32-bit bank-split LDS.32!)
    __shared__ __align__(16) __half2 sxh[2][1280]; // 10240 B (16B-aligned for LDS.128!)
    __shared__ float sxd[2][80];                //   640 B
    __shared__ float spart[2][NW][64];          //  2560 B -> total 15,488 B (3 blk/SM = 30 warps/SM, 64 regs, ZERO spills!)

    if (tid < 256) {
        const uint2 gp = __ldg(reinterpret_cast<const uint2*>(iq2xxs_grid) + tid);
        sg_lo[tid] = gp.x;
        sg_hi[tid] = gp.y;
    }

    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);
    auto stage_tok = [&](int slot, int tok, int b0, int nb) {
        const block_q8_1* xt = xq + (size_t) tok * (K_blk * 8) + b0 * 8;
        const int nh2 = nb * 128;
        for (int i = tid; i < nh2; i += 320) {
            const int blk = i >> 4, sub = (i & 15) << 1;
            sxh[slot][i] = iq_q81_pair_h2(xt[blk].qs + sub, c1152, inv128);
        }
        if (tid < nb * 8) sxd[slot][tid] = __low2float(xt[tid].ds) * 128.0f;
    };

    for (int eb = e0; eb < e1; eb += 2) {
        const int ne = (eb + 1 < e1) ? 2 : 1;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 10) {
            const int nb = (b0 + 10 <= K_blk) ? 10 : (K_blk - b0);
            stage_tok(0, ent_tok[eb], b0, nb);
            if (ne == 2) stage_tok(1, ent_tok[eb + 1], b0, nb);
            __syncthreads();

            if (valid_row) {
                if (ne == 1) {
                    for (int db_k = warp; db_k < nb; db_k += NW) {
                        const int b = b0 + db_k;
                        const uint4* Wqb = Wq + (size_t) b * 4 * N + row;
                        float bsum0 = 0.0f;
                        auto do_sb1 = [&](int sb, uint32_t idx32, uint32_t aux32) {
                            const int qblk = db_k * 8 + sb;
                            const float ls = (float) ((aux32 >> 27) | 1u);
                            const __half2* x0 = &sxh[0][qblk * 16];
                            __half2 d0 = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                            for (int k0 = 0; k0 < 4; ++k0) {
                                const uint32_t gidx = (idx32 >> (k0 * 8)) & 0xffu;
                                const uint32_t gx = sg_lo[gidx], gy = sg_hi[gidx];
                                const uint32_t s7 = (aux32 >> (7 * k0)) & 0x7fu;
                                const uint32_t sg = s7 | ((__popc(s7) & 1u) << 7);
                                d0 = __hfma2(iq_decode_h2(gx, 0x5150, ((sg & 1u) << 15) | ((sg & 2u) << 30)), x0[k0 * 4 + 0], d0);
                                d0 = __hfma2(iq_decode_h2(gx, 0x5352, ((sg & 4u) << 13) | ((sg & 8u) << 28)), x0[k0 * 4 + 1], d0);
                                d0 = __hfma2(iq_decode_h2(gy, 0x5150, ((sg & 16u) << 11) | ((sg & 32u) << 26)), x0[k0 * 4 + 2], d0);
                                d0 = __hfma2(iq_decode_h2(gy, 0x5352, ((sg & 64u) << 9) | ((sg & 128u) << 24)), x0[k0 * 4 + 3], d0);
                            }
                            bsum0 = fmaf((__low2float(d0) + __high2float(d0)) * ls, sxd[0][qblk], bsum0);
                        };
#pragma unroll 1
                        for (int q = 0; q < 4; ++q) {
                            const uint4 wv = __ldg(Wqb + (size_t) q * N);
                            do_sb1(q * 2 + 0, wv.x, wv.y);
                            do_sb1(q * 2 + 1, wv.z, wv.w);
                        }
                        const float db = __half2float(__ldg(Wd + (size_t) b * N + row)) * 0.125f;
                        acc0 = fmaf(db, bsum0, acc0);
                    }
                } else {
                    for (int db_k = warp; db_k < nb; db_k += NW) {
                        const int b = b0 + db_k;
                        const uint4* Wqb = Wq + (size_t) b * 4 * N + row;
                        float bsum0 = 0.0f, bsum1 = 0.0f;
                        auto do_sb2 = [&](int sb, uint32_t idx32, uint32_t aux32) {
                            const int qblk = db_k * 8 + sb;
                            const float ls = (float) ((aux32 >> 27) | 1u);
                            const __half2* x0 = &sxh[0][qblk * 16];
                            const __half2* x1 = &sxh[1][qblk * 16];
                            __half2 d0 = __floats2half2_rn(0.0f, 0.0f), d1 = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                            for (int k0 = 0; k0 < 4; ++k0) {
                                const uint32_t gidx = (idx32 >> (k0 * 8)) & 0xffu;
                                const uint32_t gx = sg_lo[gidx], gy = sg_hi[gidx];
                                const uint32_t s7 = (aux32 >> (7 * k0)) & 0x7fu;
                                const uint32_t sg = s7 | ((__popc(s7) & 1u) << 7);
                                __half2 w = iq_decode_h2(gx, 0x5150, ((sg & 1u) << 15) | ((sg & 2u) << 30));
                                d0 = __hfma2(w, x0[k0 * 4 + 0], d0); d1 = __hfma2(w, x1[k0 * 4 + 0], d1);
                                w = iq_decode_h2(gx, 0x5352, ((sg & 4u) << 13) | ((sg & 8u) << 28));
                                d0 = __hfma2(w, x0[k0 * 4 + 1], d0); d1 = __hfma2(w, x1[k0 * 4 + 1], d1);
                                w = iq_decode_h2(gy, 0x5150, ((sg & 16u) << 11) | ((sg & 32u) << 26));
                                d0 = __hfma2(w, x0[k0 * 4 + 2], d0); d1 = __hfma2(w, x1[k0 * 4 + 2], d1);
                                w = iq_decode_h2(gy, 0x5352, ((sg & 64u) << 9) | ((sg & 128u) << 24));
                                d0 = __hfma2(w, x0[k0 * 4 + 3], d0); d1 = __hfma2(w, x1[k0 * 4 + 3], d1);
                            }
                            bsum0 = fmaf((__low2float(d0) + __high2float(d0)) * ls, sxd[0][qblk], bsum0);
                            bsum1 = fmaf((__low2float(d1) + __high2float(d1)) * ls, sxd[1][qblk], bsum1);
                        };
#pragma unroll 1
                        for (int q = 0; q < 4; ++q) {
                            const uint4 wv = __ldg(Wqb + (size_t) q * N);
                            do_sb2(q * 2 + 0, wv.x, wv.y);
                            do_sb2(q * 2 + 1, wv.z, wv.w);
                        }
                        const float db = __half2float(__ldg(Wd + (size_t) b * N + row)) * 0.125f;
                        acc0 = fmaf(db, bsum0, acc0);
                        acc1 = fmaf(db, bsum1, acc1);
                    }
                }
            }
            if (b0 + 10 < K_blk) __syncthreads();
        }
        spart[0][warp][col] = acc0;
        if (ne == 2) spart[1][warp][col] = acc1;
        __syncthreads();
        if (warp == 0 && valid_row) {
            float s0 = spart[0][0][col] + spart[0][1][col] + spart[0][2][col] + spart[0][3][col] + spart[0][4][col];
            (is_up ? up : gate)[(size_t) eb * L.n_ff + r_out] = s0;
            if (ne == 2) {
                float s1 = spart[1][0][col] + spart[1][1][col] + spart[1][2][col] + spart[1][3][col] + spart[1][4][col];
                (is_up ? up : gate)[(size_t) (eb + 1) * L.n_ff + r_out] = s1;
            }
        }
        if (eb + 2 < e1) __syncthreads();
    }
}

// IQ2_S K-Major SoA Block-Internal Split-K: 64 rows/block x 5 warps = 320 threads, GP100 32-bit bank-split SoA codebook!
// Splits uint2 iq2s_grid[1024] into sg_lo[1024] + sg_hi[1024] (2x LDS.32 = 8 KB, ZERO LDS.64 bank conflicts, ZERO L1 TEX thrash,
// ZERO stack frame, ZERO spills, 64 regs -> 3 blocks/SM = 30 warps/SM, measured 109-114 us vs 171 us = +56% faster on GP100!).
__global__ void __launch_bounds__(320, 3) p100_iq2s_gu_soa_bsk_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok,
    const block_q8_1* __restrict__ xq, NativeExpertLayout L,
    float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    constexpr int NW = 5;
    const int tid = threadIdx.x;
    const int w_all = tid >> 5;
    const int lane = tid & 31;
    const int half = (w_all >= NW) ? 1 : 0;
    const int warp = half ? (w_all - NW) : w_all;
    const int col = half * 32 + lane;
    const int N = 2 * (int) L.n_ff;
    const int K_blk = (int) (L.n_embd / 256);   // 10 for n_embd=2560
    const int row = blockIdx.x * 64 + col;
    const bool valid_row = row < N;
    const bool is_up = row >= (int) L.n_ff;
    const int r_out = is_up ? row - (int) L.n_ff : row;

    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint4* Wqs = reinterpret_cast<const uint4*>(blob);
    const uint4* Wqhsc = reinterpret_cast<const uint4*>(blob + (size_t) 64 * K_blk * N);
    const __half* Wd = reinterpret_cast<const __half*>(blob + (size_t) 80 * K_blk * N);

    __shared__ uint32_t sg_lo[1024], sg_hi[1024]; //  8192 B (32-bit bank-split LDS.32!)
    __shared__ __align__(16) __half2 sxh[2][1280]; // 10240 B (16B-aligned for LDS.128!)
    __shared__ float sxd[2][80];                  //   640 B
    __shared__ float spart[2][NW][64];            //  2560 B -> total 21,632 B (3 blk/SM = 30 warps/SM!)

    for (int i = tid; i < 1024; i += 320) {
        const uint2 gp = __ldg(reinterpret_cast<const uint2*>(iq2s_grid) + i);
        sg_lo[i] = gp.x;
        sg_hi[i] = gp.y;
    }

    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);
    auto stage_tok = [&](int slot, int tok, int b0, int nb) {
        const block_q8_1* xt = xq + (size_t) tok * (K_blk * 8) + b0 * 8;
        const int nh2 = nb * 128;
        for (int i = tid; i < nh2; i += 320) {
            const int blk = i >> 4, sub = (i & 15) << 1;
            sxh[slot][i] = iq_q81_pair_h2(xt[blk].qs + sub, c1152, inv128);
        }
        if (tid < nb * 8) sxd[slot][tid] = __low2float(xt[tid].ds) * 128.0f;
    };

    for (int eb = e0; eb < e1; eb += 2) {
        const int ne = (eb + 1 < e1) ? 2 : 1;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 10) {
            const int nb = (b0 + 10 <= K_blk) ? 10 : (K_blk - b0);
            stage_tok(0, ent_tok[eb], b0, nb);
            if (ne == 2) stage_tok(1, ent_tok[eb + 1], b0, nb);
            __syncthreads();

            if (valid_row) {
                if (ne == 1) {
                    for (int db_k = warp; db_k < nb; db_k += NW) {
                        const int b = b0 + db_k;
                        const uint4 qhsc = __ldg(Wqhsc + (size_t) b * N + row);
                        const uint4* Wb = Wqs + (size_t) b * 4 * N + row;
                        float bsum0 = 0.0f;
                        auto do_sb1 = [&](int sb, uint32_t g32, uint32_t s32, uint32_t qh, uint32_t sc) {
                            const int qblk = db_k * 8 + sb;
                            const __half2* x0 = &sxh[0][qblk * 16];
                            float p0 = 0.0f;
#pragma unroll
                            for (int half_sb = 0; half_sb < 2; ++half_sb) {
                                __half2 d0 = __floats2half2_rn(0.0f, 0.0f);
                                const int kbase = half_sb * 2;
#pragma unroll
                                for (int kk = 0; kk < 2; ++kk) {
                                    const int k0 = kbase + kk;
                                    const uint32_t gidx = ((g32 >> (k0 * 8)) & 0xffu) | ((qh << (8 - 2 * k0)) & 0x300u);
                                    const uint32_t gx = sg_lo[gidx], gy = sg_hi[gidx];
                                    const uint32_t sg = (s32 >> (k0 * 8)) & 0xffu;
                                    d0 = __hfma2(iq_decode_h2(gx, 0x5150, ((sg & 1u) << 15) | ((sg & 2u) << 30)), x0[k0 * 4 + 0], d0);
                                    d0 = __hfma2(iq_decode_h2(gx, 0x5352, ((sg & 4u) << 13) | ((sg & 8u) << 28)), x0[k0 * 4 + 1], d0);
                                    d0 = __hfma2(iq_decode_h2(gy, 0x5150, ((sg & 16u) << 11) | ((sg & 32u) << 26)), x0[k0 * 4 + 2], d0);
                                    d0 = __hfma2(iq_decode_h2(gy, 0x5352, ((sg & 64u) << 9) | ((sg & 128u) << 24)), x0[k0 * 4 + 3], d0);
                                }
                                const float ls = (float) ((half_sb ? (sc >> 4) : (sc & 0x0fu)) * 2u + 1u);
                                p0 = fmaf(__low2float(d0) + __high2float(d0), ls, p0);
                            }
                            bsum0 = fmaf(p0, sxd[0][qblk], bsum0);
                        };
#pragma unroll 1
                        for (int h = 0; h < 2; ++h) {
                            const uint4 gw = __ldg(Wb + (size_t) h * N);
                            const uint4 sw = __ldg(Wb + (size_t) (h + 2) * N);
                            const uint32_t qh4 = h ? qhsc.y : qhsc.x;
                            const uint32_t sc4 = h ? qhsc.w : qhsc.z;
                            do_sb1(h * 4 + 0, gw.x, sw.x, qh4 & 0xffu, sc4 & 0xffu);
                            do_sb1(h * 4 + 1, gw.y, sw.y, (qh4 >> 8) & 0xffu, (sc4 >> 8) & 0xffu);
                            do_sb1(h * 4 + 2, gw.z, sw.z, (qh4 >> 16) & 0xffu, (sc4 >> 16) & 0xffu);
                            do_sb1(h * 4 + 3, gw.w, sw.w, (qh4 >> 24) & 0xffu, (sc4 >> 24) & 0xffu);
                        }
                        const float db = __half2float(__ldg(Wd + (size_t) b * N + row)) * 0.125f;
                        acc0 = fmaf(db, bsum0, acc0);
                    }
                } else {
                    for (int db_k = warp; db_k < nb; db_k += NW) {
                        const int b = b0 + db_k;
                        const uint4 qhsc = __ldg(Wqhsc + (size_t) b * N + row);
                        const uint4* Wb = Wqs + (size_t) b * 4 * N + row;
                        float bsum0 = 0.0f, bsum1 = 0.0f;
                        auto do_sb2 = [&](int sb, uint32_t g32, uint32_t s32, uint32_t qh, uint32_t sc) {
                            const int qblk = db_k * 8 + sb;
                            const __half2* x0 = &sxh[0][qblk * 16];
                            const __half2* x1 = &sxh[1][qblk * 16];
                            float p0 = 0.0f, p1 = 0.0f;
#pragma unroll
                            for (int half_sb = 0; half_sb < 2; ++half_sb) {
                                __half2 d0 = __floats2half2_rn(0.0f, 0.0f), d1 = __floats2half2_rn(0.0f, 0.0f);
                                const int kbase = half_sb * 2;
#pragma unroll
                                for (int kk = 0; kk < 2; ++kk) {
                                    const int k0 = kbase + kk;
                                    const uint32_t gidx = ((g32 >> (k0 * 8)) & 0xffu) | ((qh << (8 - 2 * k0)) & 0x300u);
                                    const uint32_t gx = sg_lo[gidx], gy = sg_hi[gidx];
                                    const uint32_t sg = (s32 >> (k0 * 8)) & 0xffu;
                                    __half2 w = iq_decode_h2(gx, 0x5150, ((sg & 1u) << 15) | ((sg & 2u) << 30));
                                    d0 = __hfma2(w, x0[k0 * 4 + 0], d0); d1 = __hfma2(w, x1[k0 * 4 + 0], d1);
                                    w = iq_decode_h2(gx, 0x5352, ((sg & 4u) << 13) | ((sg & 8u) << 28));
                                    d0 = __hfma2(w, x0[k0 * 4 + 1], d0); d1 = __hfma2(w, x1[k0 * 4 + 1], d1);
                                    w = iq_decode_h2(gy, 0x5150, ((sg & 16u) << 11) | ((sg & 32u) << 26));
                                    d0 = __hfma2(w, x0[k0 * 4 + 2], d0); d1 = __hfma2(w, x1[k0 * 4 + 2], d1);
                                    w = iq_decode_h2(gy, 0x5352, ((sg & 64u) << 9) | ((sg & 128u) << 24));
                                    d0 = __hfma2(w, x0[k0 * 4 + 3], d0); d1 = __hfma2(w, x1[k0 * 4 + 3], d1);
                                }
                                const float ls = (float) ((half_sb ? (sc >> 4) : (sc & 0x0fu)) * 2u + 1u);
                                p0 = fmaf(__low2float(d0) + __high2float(d0), ls, p0);
                                p1 = fmaf(__low2float(d1) + __high2float(d1), ls, p1);
                            }
                            bsum0 = fmaf(p0, sxd[0][qblk], bsum0);
                            bsum1 = fmaf(p1, sxd[1][qblk], bsum1);
                        };
#pragma unroll 1
                        for (int h = 0; h < 2; ++h) {
                            const uint4 gw = __ldg(Wb + (size_t) h * N);
                            const uint4 sw = __ldg(Wb + (size_t) (h + 2) * N);
                            const uint32_t qh4 = h ? qhsc.y : qhsc.x;
                            const uint32_t sc4 = h ? qhsc.w : qhsc.z;
                            do_sb2(h * 4 + 0, gw.x, sw.x, qh4 & 0xffu, sc4 & 0xffu);
                            do_sb2(h * 4 + 1, gw.y, sw.y, (qh4 >> 8) & 0xffu, (sc4 >> 8) & 0xffu);
                            do_sb2(h * 4 + 2, gw.z, sw.z, (qh4 >> 16) & 0xffu, (sc4 >> 16) & 0xffu);
                            do_sb2(h * 4 + 3, gw.w, sw.w, (qh4 >> 24) & 0xffu, (sc4 >> 24) & 0xffu);
                        }
                        const float db = __half2float(__ldg(Wd + (size_t) b * N + row)) * 0.125f;
                        acc0 = fmaf(db, bsum0, acc0);
                        acc1 = fmaf(db, bsum1, acc1);
                    }
                }
            }
            if (b0 + 10 < K_blk) __syncthreads();
        }
        spart[0][warp][col] = acc0;
        if (ne == 2) spart[1][warp][col] = acc1;
        __syncthreads();
        if (warp == 0 && valid_row) {
            float s0 = spart[0][0][col] + spart[0][1][col] + spart[0][2][col] + spart[0][3][col] + spart[0][4][col];
            (is_up ? up : gate)[(size_t) eb * L.n_ff + r_out] = s0;
            if (ne == 2) {
                float s1 = spart[1][0][col] + spart[1][1][col] + spart[1][2][col] + spart[1][3][col] + spart[1][4][col];
                (is_up ? up : gate)[(size_t) (eb + 1) * L.n_ff + r_out] = s1;
            }
        }
        if (eb + 2 < e1) __syncthreads();
    }
}

// Q2_0 K-Major SoA Down (64 threads/block, 1 thread = 1 row, zero LUT, phase-reordered bit-extract HFMA2).
__global__ void __launch_bounds__(64, 16) p100_q20_down_soa_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst,
    const block_q8_1* __restrict__ hq, NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    const int tid = threadIdx.x;
    const int N = (int) L.n_embd;
    const int K_blk = (int) (L.n_ff / 64);      // 10 for n_ff=640
    const int hb = (int) (L.n_ff / 32);         // 20 for n_ff=640
    const int r = blockIdx.x * 64 + tid;

    const uint8_t* blob = (const uint8_t*) grp_ptr[g] + L.down_off;
    const uint4* Wq = reinterpret_cast<const uint4*>(blob);
    const __half* Wd = reinterpret_cast<const __half*>(blob + (size_t) 16 * K_blk * N);

    // Phase-reordered sxh[slot][b][u][ph][2]: pairs (4*byte+ph) for byte=0..3 within each 4-byte word u!
    __shared__ __align__(8) __half2 sxh[2][10][4][4][2];
    __shared__ float sxd[2][20];
    const __half2 c1025 = __floats2half2_rn(1025.0f, 1025.0f);
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);

    auto stage_ent = [&](int slot, int ent_id, int b0, int nb) {
        const block_q8_1* xt = hq + (size_t) ent_id * hb + b0 * 2;
        for (int idx = tid; idx < nb * 32; idx += 64) {
            const int db = idx >> 5;
            const int rem = idx & 31;
            const int u = rem >> 3;            // 0..3 (which 4-byte word of qs[16])
            const int ph = (rem >> 1) & 3;     // 0..3 (which 2-bit phase)
            const int p2 = rem & 1;            // 0 -> bytes (0,1), 1 -> bytes (2,3)
            const int qblk = db * 2 + (u >> 1);
            const int base = (u & 1) * 16;
            const int i0 = base + (p2 * 2 + 0) * 4 + ph;
            const int i1 = base + (p2 * 2 + 1) * 4 + ph;
            const uint32_t b0v = (uint8_t) xt[qblk].qs[i0];
            const uint32_t b1v = (uint8_t) xt[qblk].qs[i1];
            const uint32_t vx = (b0v | (b1v << 8)) ^ 0x8080u;
            const uint32_t h = __byte_perm(vx, 0x64006400u, 0x5150);
            sxh[slot][db][u][ph][p2] = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&h), c1152), inv128);
        }
        if (tid < nb * 2) sxd[slot][tid] = __low2float(xt[tid].ds) * 128.0f;
    };

    for (int eb = e0; eb < e1; eb += 2) {
        const int ne = (eb + 1 < e1) ? 2 : 1;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 10) {
            const int nb = (b0 + 10 <= K_blk) ? 10 : (K_blk - b0);
            stage_ent(0, eb, b0, nb);
            if (ne == 2) stage_ent(1, eb + 1, b0, nb);
            __syncthreads();

            if (r < N) {
                if (ne == 1) {
                    for (int db_k = 0; db_k < nb; ++db_k) {
                        const int b = b0 + db_k;
                        const uint4 q4 = __ldg(Wq + (size_t) b * N + r);
                        const float db = __half2float(__ldg(Wd + (size_t) b * N + r));
                        const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                        __half2 s0_lo = __floats2half2_rn(0.0f, 0.0f), s0_hi = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const uint32_t w = wv[u];
                            __half2& dst0 = (u < 2) ? s0_lo : s0_hi;
#pragma unroll
                            for (int ph = 0; ph < 4; ++ph) {
                                const uint32_t e = (w >> (ph * 2)) & 0x03030303u;
                                const uint32_t h0 = __byte_perm(0x64006400u, e, 0x3514);
                                const uint32_t h1 = __byte_perm(0x64006400u, e, 0x3716);
                                const __half2 v0 = __hsub2(*reinterpret_cast<const __half2*>(&h0), c1025);
                                const __half2 v1 = __hsub2(*reinterpret_cast<const __half2*>(&h1), c1025);
                                dst0 = __hfma2(v0, sxh[0][db_k][u][ph][0], dst0);
                                dst0 = __hfma2(v1, sxh[0][db_k][u][ph][1], dst0);
                            }
                        }
                        acc0 += db * ((__low2float(s0_lo) + __high2float(s0_lo)) * sxd[0][db_k * 2] +
                                      (__low2float(s0_hi) + __high2float(s0_hi)) * sxd[0][db_k * 2 + 1]);
                    }
                } else {
                    for (int db_k = 0; db_k < nb; ++db_k) {
                        const int b = b0 + db_k;
                        const uint4 q4 = __ldg(Wq + (size_t) b * N + r);
                        const float db = __half2float(__ldg(Wd + (size_t) b * N + r));
                        const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                        __half2 s0_lo = __floats2half2_rn(0.0f, 0.0f), s0_hi = __floats2half2_rn(0.0f, 0.0f);
                        __half2 s1_lo = __floats2half2_rn(0.0f, 0.0f), s1_hi = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const uint32_t w = wv[u];
                            __half2& dst0 = (u < 2) ? s0_lo : s0_hi;
                            __half2& dst1 = (u < 2) ? s1_lo : s1_hi;
#pragma unroll
                            for (int ph = 0; ph < 4; ++ph) {
                                const uint32_t e = (w >> (ph * 2)) & 0x03030303u;
                                const uint32_t h0 = __byte_perm(0x64006400u, e, 0x3514);
                                const uint32_t h1 = __byte_perm(0x64006400u, e, 0x3716);
                                const __half2 v0 = __hsub2(*reinterpret_cast<const __half2*>(&h0), c1025);
                                const __half2 v1 = __hsub2(*reinterpret_cast<const __half2*>(&h1), c1025);
                                dst0 = __hfma2(v0, sxh[0][db_k][u][ph][0], dst0);
                                dst0 = __hfma2(v1, sxh[0][db_k][u][ph][1], dst0);
                                dst1 = __hfma2(v0, sxh[1][db_k][u][ph][0], dst1);
                                dst1 = __hfma2(v1, sxh[1][db_k][u][ph][1], dst1);
                            }
                        }
                        acc0 += db * ((__low2float(s0_lo) + __high2float(s0_lo)) * sxd[0][db_k * 2] +
                                      (__low2float(s0_hi) + __high2float(s0_hi)) * sxd[0][db_k * 2 + 1]);
                        acc1 += db * ((__low2float(s1_lo) + __high2float(s1_lo)) * sxd[1][db_k * 2] +
                                      (__low2float(s1_hi) + __high2float(s1_hi)) * sxd[1][db_k * 2 + 1]);
                    }
                }
            }
            if (b0 + 10 < K_blk) __syncthreads();
        }
        if (r < N) {
            out[(size_t) ent_dst[eb] * N + r] = acc0;
            if (ne == 2) out[(size_t) ent_dst[eb + 1] * N + r] = acc1;
        }
        if (eb + 2 < e1) __syncthreads();
    }
}

// Register-only 16-entry IQ4_NL LUT decode for 4 packed bytes (8 nibbles -> 4 lo + 4 hi biased uint8s).
__device__ __forceinline__ uint2 iq4nl_lut_reg(uint32_t q32, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    const uint32_t lo = q32 & 0x0f0f0f0fu;
    const uint32_t hi = (q32 >> 4) & 0x0f0f0f0fu;
    auto map4 = [&](uint32_t n) -> uint32_t {
        uint32_t sel = (n & 0x03u) | ((n >> 4) & 0x30u) | ((n >> 8) & 0x0300u) | ((n >> 12) & 0x3000u);
        uint32_t r01 = __byte_perm(t0, t1, sel | ((n & 0x04u)) | ((n >> 4) & 0x40u) | ((n >> 8) & 0x0400u) | ((n >> 12) & 0x4000u));
        uint32_t r23 = __byte_perm(t2, t3, sel | ((n & 0x04u)) | ((n >> 4) & 0x40u) | ((n >> 8) & 0x0400u) | ((n >> 12) & 0x4000u));
        uint32_t m = ((n & 0x08080808u) >> 3) * 0xffu;
        return (r01 & ~m) | (r23 & m);
    };
    return make_uint2(map4(lo), map4(hi));
}

// IQ4_NL K-Major SoA Down (64 threads/block, 1 thread = 1 row, zero shared-memory LUT bank conflicts).
__global__ void __launch_bounds__(64, 16) p100_iq4nl_down_soa_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst,
    const block_q8_1* __restrict__ hq, NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    const int tid = threadIdx.x;
    const int N = (int) L.n_embd;
    const int K_blk = (int) (L.n_ff / 32);      // 20 for n_ff=640
    const int r = blockIdx.x * 64 + tid;

    const uint8_t* blob = (const uint8_t*) grp_ptr[g] + L.down_off;
    const uint4* Wq = reinterpret_cast<const uint4*>(blob);
    const __half* Wd = reinterpret_cast<const __half*>(blob + (size_t) 16 * K_blk * N);

    __shared__ __align__(8) __half2 sxh[2][20][16];
    __shared__ float sxd[2][20];
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);
    constexpr uint32_t T0 = 0x3f2d1801u, T1 = 0x766a5d4fu, T2 = 0xa6998d81u, T3 = 0xf1d9c5b5u;

    auto stage_ent = [&](int slot, int ent_id, int b0, int nb) {
        const block_q8_1* xt = hq + (size_t) ent_id * K_blk + b0;
        for (int i = tid; i < nb * 16; i += 64) {
            const int db = i >> 4, sub = (i & 15) << 1;
            const uint16_t pair = *reinterpret_cast<const uint16_t*>(xt[db].qs + sub);
            const uint32_t vx = ((uint32_t) pair) ^ 0x8080u;
            const uint32_t h = __byte_perm(vx, 0x64006400u, 0x5150);
            sxh[slot][db][i & 15] = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&h), c1152), inv128);
        }
        if (tid < nb) sxd[slot][tid] = __low2float(xt[tid].ds) * 128.0f;
    };

    for (int eb = e0; eb < e1; eb += 2) {
        const int ne = (eb + 1 < e1) ? 2 : 1;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = (b0 + 20 <= K_blk) ? 20 : (K_blk - b0);
            stage_ent(0, eb, b0, nb);
            if (ne == 2) stage_ent(1, eb + 1, b0, nb);
            __syncthreads();

            if (r < N) {
                if (ne == 1) {
                    for (int db_k = 0; db_k < nb; ++db_k) {
                        const int b = b0 + db_k;
                        const uint4 q4 = __ldg(Wq + (size_t) b * N + r);
                        const float db = __half2float(__ldg(Wd + (size_t) b * N + r));
                        const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                        __half2 s0 = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const uint2 v = iq4nl_lut_reg(wv[u], T0, T1, T2, T3);
                            const uint32_t a0 = __byte_perm(0x64006400u, v.x, 0x3514);
                            const uint32_t a1 = __byte_perm(0x64006400u, v.x, 0x3716);
                            const uint32_t b0v = __byte_perm(0x64006400u, v.y, 0x3514);
                            const uint32_t b1v = __byte_perm(0x64006400u, v.y, 0x3716);
                            const uint2 x0a = *reinterpret_cast<const uint2*>(&sxh[0][db_k][u * 2]);
                            const uint2 x0b = *reinterpret_cast<const uint2*>(&sxh[0][db_k][8 + u * 2]);
                            s0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&a0), c1152), *reinterpret_cast<const __half2*>(&x0a.x), s0);
                            s0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&a1), c1152), *reinterpret_cast<const __half2*>(&x0a.y), s0);
                            s0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&b0v), c1152), *reinterpret_cast<const __half2*>(&x0b.x), s0);
                            s0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&b1v), c1152), *reinterpret_cast<const __half2*>(&x0b.y), s0);
                        }
                        acc0 += db * sxd[0][db_k] * (__low2float(s0) + __high2float(s0));
                    }
                } else {
                    for (int db_k = 0; db_k < nb; ++db_k) {
                        const int b = b0 + db_k;
                        const uint4 q4 = __ldg(Wq + (size_t) b * N + r);
                        const float db = __half2float(__ldg(Wd + (size_t) b * N + r));
                        const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                        __half2 s0 = __floats2half2_rn(0.0f, 0.0f);
                        __half2 s1 = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const uint2 v = iq4nl_lut_reg(wv[u], T0, T1, T2, T3);
                            const uint32_t a0 = __byte_perm(0x64006400u, v.x, 0x3514);
                            const uint32_t a1 = __byte_perm(0x64006400u, v.x, 0x3716);
                            const uint32_t b0v = __byte_perm(0x64006400u, v.y, 0x3514);
                            const uint32_t b1v = __byte_perm(0x64006400u, v.y, 0x3716);
                            const __half2 va0 = __hsub2(*reinterpret_cast<const __half2*>(&a0), c1152);
                            const __half2 va1 = __hsub2(*reinterpret_cast<const __half2*>(&a1), c1152);
                            const __half2 vb0 = __hsub2(*reinterpret_cast<const __half2*>(&b0v), c1152);
                            const __half2 vb1 = __hsub2(*reinterpret_cast<const __half2*>(&b1v), c1152);
                            const uint2 x0a = *reinterpret_cast<const uint2*>(&sxh[0][db_k][u * 2]);
                            const uint2 x0b = *reinterpret_cast<const uint2*>(&sxh[0][db_k][8 + u * 2]);
                            const uint2 x1a = *reinterpret_cast<const uint2*>(&sxh[1][db_k][u * 2]);
                            const uint2 x1b = *reinterpret_cast<const uint2*>(&sxh[1][db_k][8 + u * 2]);
                            s0 = __hfma2(va0, *reinterpret_cast<const __half2*>(&x0a.x), s0);
                            s0 = __hfma2(va1, *reinterpret_cast<const __half2*>(&x0a.y), s0);
                            s0 = __hfma2(vb0, *reinterpret_cast<const __half2*>(&x0b.x), s0);
                            s0 = __hfma2(vb1, *reinterpret_cast<const __half2*>(&x0b.y), s0);
                            s1 = __hfma2(va0, *reinterpret_cast<const __half2*>(&x1a.x), s1);
                            s1 = __hfma2(va1, *reinterpret_cast<const __half2*>(&x1a.y), s1);
                            s1 = __hfma2(vb0, *reinterpret_cast<const __half2*>(&x1b.x), s1);
                            s1 = __hfma2(vb1, *reinterpret_cast<const __half2*>(&x1b.y), s1);
                        }
                        acc0 += db * sxd[0][db_k] * (__low2float(s0) + __high2float(s0));
                        acc1 += db * sxd[1][db_k] * (__low2float(s1) + __high2float(s1));
                    }
                }
            }
            if (b0 + 20 < K_blk) __syncthreads();
        }
        if (r < N) {
            out[(size_t) ent_dst[eb] * N + r] = acc0;
            if (ne == 2) out[(size_t) ent_dst[eb + 1] * N + r] = acc1;
        }
        if (eb + 2 < e1) __syncthreads();
    }
}

// IQ4_NL K-Major SoA Down Split-K5 (320t = 5 K-workers x 64 rows, 4 blocks/worker, REG:48 STACK:0! 1.23x faster!).
__global__ void __launch_bounds__(320, 4) p100_iq4nl_down_soa_sk5_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst,
    const block_q8_1* __restrict__ hq, NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    const int tid = threadIdx.x, kw = tid >> 6, rl = tid & 63;
    const int N = (int) L.n_embd;
    const int K_blk = (int) (L.n_ff / 32);
    const int r = blockIdx.x * 64 + rl;

    const uint8_t* blob = (const uint8_t*) grp_ptr[g] + L.down_off;
    const uint4* Wq = reinterpret_cast<const uint4*>(blob);
    const __half* Wd = reinterpret_cast<const __half*>(blob + (size_t) 16 * K_blk * N);

    union Iq4nlDnSk5Smem {
        struct { __align__(8) __half2 sxh[2][20][16]; float sxd[2][20]; } x;
        float part[5][2][64];
    };
    __shared__ Iq4nlDnSk5Smem smem;
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);
    constexpr uint32_t T0 = 0x3f2d1801u, T1 = 0x766a5d4fu, T2 = 0xa6998d81u, T3 = 0xf1d9c5b5u;

    for (int eb = e0; eb < e1; eb += 2) {
        const int ne = (eb + 1 < e1) ? 2 : 1;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = (b0 + 20 <= K_blk) ? 20 : (K_blk - b0);
            if (tid < nb * 16) {
                const int db = tid >> 4, sub = (tid & 15) << 1;
                const block_q8_1* xt0 = hq + (size_t) eb * K_blk + b0;
                const uint16_t p0 = *reinterpret_cast<const uint16_t*>(xt0[db].qs + sub);
                const uint32_t h0 = __byte_perm(((uint32_t) p0) ^ 0x8080u, 0x64006400u, 0x5150);
                smem.x.sxh[0][db][tid & 15] = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&h0), c1152), inv128);
                if ((tid & 15) == 0) smem.x.sxd[0][db] = __low2float(xt0[db].ds) * 128.0f;
                if (ne == 2) {
                    const block_q8_1* xt1 = hq + (size_t) (eb + 1) * K_blk + b0;
                    const uint16_t p1 = *reinterpret_cast<const uint16_t*>(xt1[db].qs + sub);
                    const uint32_t h1 = __byte_perm(((uint32_t) p1) ^ 0x8080u, 0x64006400u, 0x5150);
                    smem.x.sxh[1][db][tid & 15] = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&h1), c1152), inv128);
                    if ((tid & 15) == 0) smem.x.sxd[1][db] = __low2float(xt1[db].ds) * 128.0f;
                }
            }
            __syncthreads();

            if (r < N) {
                for (int db_k = kw; db_k < nb; db_k += 5) {
                    const int b = b0 + db_k;
                    const uint4 q4 = __ldg(Wq + (size_t) b * N + r);
                    const float db = __half2float(__ldg(Wd + (size_t) b * N + r));
                    const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                    __half2 s0 = __floats2half2_rn(0.0f, 0.0f), s1 = s0;
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const uint2 v = iq4nl_lut_reg(wv[u], T0, T1, T2, T3);
                        const uint32_t a0 = __byte_perm(0x64006400u, v.x, 0x3514);
                        const uint32_t a1 = __byte_perm(0x64006400u, v.x, 0x3716);
                        const uint32_t b0v = __byte_perm(0x64006400u, v.y, 0x3514);
                        const uint32_t b1v = __byte_perm(0x64006400u, v.y, 0x3716);
                        const __half2 va0 = __hsub2(*reinterpret_cast<const __half2*>(&a0), c1152);
                        const __half2 va1 = __hsub2(*reinterpret_cast<const __half2*>(&a1), c1152);
                        const __half2 vb0 = __hsub2(*reinterpret_cast<const __half2*>(&b0v), c1152);
                        const __half2 vb1 = __hsub2(*reinterpret_cast<const __half2*>(&b1v), c1152);
                        const uint2 x0a = *reinterpret_cast<const uint2*>(&smem.x.sxh[0][db_k][u * 2]);
                        const uint2 x0b = *reinterpret_cast<const uint2*>(&smem.x.sxh[0][db_k][8 + u * 2]);
                        s0 = __hfma2(va0, *reinterpret_cast<const __half2*>(&x0a.x), s0);
                        s0 = __hfma2(va1, *reinterpret_cast<const __half2*>(&x0a.y), s0);
                        s0 = __hfma2(vb0, *reinterpret_cast<const __half2*>(&x0b.x), s0);
                        s0 = __hfma2(vb1, *reinterpret_cast<const __half2*>(&x0b.y), s0);
                        if (ne == 2) {
                            const uint2 x1a = *reinterpret_cast<const uint2*>(&smem.x.sxh[1][db_k][u * 2]);
                            const uint2 x1b = *reinterpret_cast<const uint2*>(&smem.x.sxh[1][db_k][8 + u * 2]);
                            s1 = __hfma2(va0, *reinterpret_cast<const __half2*>(&x1a.x), s1);
                            s1 = __hfma2(va1, *reinterpret_cast<const __half2*>(&x1a.y), s1);
                            s1 = __hfma2(vb0, *reinterpret_cast<const __half2*>(&x1b.x), s1);
                            s1 = __hfma2(vb1, *reinterpret_cast<const __half2*>(&x1b.y), s1);
                        }
                    }
                    acc0 += db * smem.x.sxd[0][db_k] * (__low2float(s0) + __high2float(s0));
                    if (ne == 2) acc1 += db * smem.x.sxd[1][db_k] * (__low2float(s1) + __high2float(s1));
                }
            }
            __syncthreads();
        }
        smem.part[kw][0][rl] = acc0;
        if (ne == 2) smem.part[kw][1][rl] = acc1;
        __syncthreads();
        if (kw == 0 && r < N) {
            out[(size_t) ent_dst[eb] * N + r] =
                smem.part[0][0][rl] + smem.part[1][0][rl] + smem.part[2][0][rl] + smem.part[3][0][rl] + smem.part[4][0][rl];
            if (ne == 2)
                out[(size_t) ent_dst[eb + 1] * N + r] =
                    smem.part[0][1][rl] + smem.part[1][1][rl] + smem.part[2][1][rl] + smem.part[3][1][rl] + smem.part[4][1][rl];
        }
        if (eb + 2 < e1) __syncthreads();
    }
}

// GP100 (sm_60) Q4_K K-Major SoA Block-Internal Split-K Gate+Up: 64 rows/block x 5 K-parts = 320 threads,
// 16-byte coalesced uint4 loads + fused Wscdm LDG.E.128 + 4-token register reuse (eb += 4), STACK:0!
__global__ void __launch_bounds__(320, 2) p100_q4k_gu_soa_bsk_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok,
    const block_q8_1* __restrict__ xq, NativeExpertLayout L,
    float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    const int tid = threadIdx.x;
    const int r_loc = tid & 63;
    const int kb = tid >> 6;         // 0..4 (5 K-partitions)
    const int N = 2 * (int) L.n_ff;
    const int row = blockIdx.x * 64 + r_loc;
    const bool valid_row = row < N;
    const bool is_up = row >= (int) L.n_ff;
    const int r_out = is_up ? (row - (int) L.n_ff) : row;
    const int K_blk = (int) (L.n_embd / 256);
    const int K_q8 = (int) (L.n_embd / 32);

    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint4* Wqs = reinterpret_cast<const uint4*>(blob);
    const uint4* Wscdm = reinterpret_cast<const uint4*>(blob + (size_t) 128 * K_blk * N);

    union Q4kSmem {
        struct {
            __align__(16) __half2 sxh[4][1280];
            float sxd[4][80];
            float sxs[4][80];
        } x;
        float part[4][5][64];
    };
    __shared__ Q4kSmem smem;
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);

    for (int eb = e0; eb < e1; eb += 4) {
        const int ne = (eb + 4 <= e1) ? 4 : (e1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 10) {
            const int nb = (b0 + 10 <= K_blk) ? 10 : (K_blk - b0);
            const int nq8 = nb * 8, nh2 = nb * 128;
            for (int t = 0; t < ne; ++t) {
                const block_q8_1* xb = xq + (size_t) ent_tok[eb + t] * K_q8 + b0 * 8;
                for (int i = tid; i < nh2; i += 320) {
                    const int qb = i >> 4, sub = (i & 15) << 1;
                    smem.x.sxh[t][i] = iq_q81_pair_h2(xb[qb].qs + sub, c1152, inv128);
                }
                if (tid < nq8) {
                    const float2 ds = __half22float2(xb[tid].ds);
                    smem.x.sxd[t][tid] = ds.x * 128.0f;
                    smem.x.sxs[t][tid] = ds.y;
                }
            }
            if (ne == 3) {
                for (int i = tid; i < nh2; i += 320) smem.x.sxh[3][i] = __float2half2_rn(0.0f);
                if (tid < nq8) { smem.x.sxd[3][tid] = 0.0f; smem.x.sxs[3][tid] = 0.0f; }
            }
            __syncthreads();

            if (valid_row) {
                if (ne == 1) {
                    for (int db_k = kb; db_k < nb; db_k += 5) {
                        const int b = b0 + db_k;
                        const uint4 scdm = __ldg(Wscdm + (size_t) b * N + row);
                        const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&scdm.w));
                        const uint32_t sc_w[2] = {scdm.x & 0x3f3f3f3fu, (scdm.z & 0x0f0f0f0fu) | ((scdm.x >> 2) & 0x30303030u)};
                        const uint32_t m_w[2]  = {scdm.y & 0x3f3f3f3fu, ((scdm.z >> 4) & 0x0f0f0f0fu) | ((scdm.y >> 2) & 0x30303030u)};
                        float sumd0 = 0.0f, summ0 = 0.0f;
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const uint4 qa = __ldg(Wqs + ((size_t) b * 8 + c * 2 + 0) * N + row);
                            const uint4 qb = __ldg(Wqs + ((size_t) b * 8 + c * 2 + 1) * N + row);
                            const uint32_t wv[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
                            const int sb0 = c * 2, sb1 = c * 2 + 1;
                            const float sc0 = (float) ((sc_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                            const float sc1 = (float) ((sc_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                            const float mn0 = (float) ((m_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                            const float mn1 = (float) ((m_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                            __half2 p0a = __float2half2_rn(0.0f), p0b = __float2half2_rn(0.0f);
                            const int x0_base = (db_k * 8 + sb0) * 16, x1_base = (db_k * 8 + sb1) * 16;
#pragma unroll
                            for (int u = 0; u < 8; ++u) {
                                const uint32_t w = wv[u];
                                const uint32_t lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu;
                                const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                                const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                                const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                                const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                                p0a = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&la), c1024), smem.x.sxh[0][x0_base + u * 2 + 0], p0a);
                                p0a = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&lb), c1024), smem.x.sxh[0][x0_base + u * 2 + 1], p0a);
                                p0b = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&ha), c1024), smem.x.sxh[0][x1_base + u * 2 + 0], p0b);
                                p0b = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&hb), c1024), smem.x.sxh[0][x1_base + u * 2 + 1], p0b);
                            }
                            const float2 f0a = __half22float2(p0a), f0b = __half22float2(p0b);
                            sumd0 = fmaf(sc0 * smem.x.sxd[0][db_k * 8 + sb0], f0a.x + f0a.y, sumd0);
                            sumd0 = fmaf(sc1 * smem.x.sxd[0][db_k * 8 + sb1], f0b.x + f0b.y, sumd0);
                            summ0 = fmaf(mn0, smem.x.sxs[0][db_k * 8 + sb0], summ0);
                            summ0 = fmaf(mn1, smem.x.sxs[0][db_k * 8 + sb1], summ0);
                        }
                        acc0 = fmaf(dm.x, sumd0, fmaf(-dm.y, summ0, acc0));
                    }
                } else if (ne == 2) {
                    for (int db_k = kb; db_k < nb; db_k += 5) {
                        const int b = b0 + db_k;
                        const uint4 scdm = __ldg(Wscdm + (size_t) b * N + row);
                        const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&scdm.w));
                        const uint32_t sc_w[2] = {scdm.x & 0x3f3f3f3fu, (scdm.z & 0x0f0f0f0fu) | ((scdm.x >> 2) & 0x30303030u)};
                        const uint32_t m_w[2]  = {scdm.y & 0x3f3f3f3fu, ((scdm.z >> 4) & 0x0f0f0f0fu) | ((scdm.y >> 2) & 0x30303030u)};
                        float sumd0 = 0.0f, summ0 = 0.0f, sumd1 = 0.0f, summ1 = 0.0f;
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const uint4 qa = __ldg(Wqs + ((size_t) b * 8 + c * 2 + 0) * N + row);
                            const uint4 qb = __ldg(Wqs + ((size_t) b * 8 + c * 2 + 1) * N + row);
                            const int sb0 = c * 2, sb1 = c * 2 + 1;
                            const float sc0 = (float) ((sc_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                            const float sc1 = (float) ((sc_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                            const float mn0 = (float) ((m_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                            const float mn1 = (float) ((m_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                            __half2 p0a = __float2half2_rn(0.0f), p0b = __float2half2_rn(0.0f);
                            __half2 p1a = __float2half2_rn(0.0f), p1b = __float2half2_rn(0.0f);
                            const int x0_base = (db_k * 8 + sb0) * 16, x1_base = (db_k * 8 + sb1) * 16;
                            auto step_u = [&](uint32_t w, int u) {
                                const uint32_t lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu;
                                const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                                const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                                const __half2 wla = __hsub2(*reinterpret_cast<const __half2*>(&la), c1024);
                                const __half2 wlb = __hsub2(*reinterpret_cast<const __half2*>(&lb), c1024);
                                p0a = __hfma2(wla, smem.x.sxh[0][x0_base + u * 2 + 0], p0a);
                                p0a = __hfma2(wlb, smem.x.sxh[0][x0_base + u * 2 + 1], p0a);
                                p1a = __hfma2(wla, smem.x.sxh[1][x0_base + u * 2 + 0], p1a);
                                p1a = __hfma2(wlb, smem.x.sxh[1][x0_base + u * 2 + 1], p1a);
                                const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                                const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                                const __half2 wha = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                                const __half2 whb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
                                p0b = __hfma2(wha, smem.x.sxh[0][x1_base + u * 2 + 0], p0b);
                                p0b = __hfma2(whb, smem.x.sxh[0][x1_base + u * 2 + 1], p0b);
                                p1b = __hfma2(wha, smem.x.sxh[1][x1_base + u * 2 + 0], p1b);
                                p1b = __hfma2(whb, smem.x.sxh[1][x1_base + u * 2 + 1], p1b);
                            };
                            step_u(qa.x, 0); step_u(qa.y, 1); step_u(qa.z, 2); step_u(qa.w, 3);
                            step_u(qb.x, 4); step_u(qb.y, 5); step_u(qb.z, 6); step_u(qb.w, 7);
                            const float2 f0a = __half22float2(p0a), f0b = __half22float2(p0b);
                            const float2 f1a = __half22float2(p1a), f1b = __half22float2(p1b);
                            sumd0 = fmaf(sc0 * smem.x.sxd[0][db_k * 8 + sb0], f0a.x + f0a.y, sumd0);
                            sumd0 = fmaf(sc1 * smem.x.sxd[0][db_k * 8 + sb1], f0b.x + f0b.y, sumd0);
                            summ0 = fmaf(mn0, smem.x.sxs[0][db_k * 8 + sb0], summ0); summ0 = fmaf(mn1, smem.x.sxs[0][db_k * 8 + sb1], summ0);
                            sumd1 = fmaf(sc0 * smem.x.sxd[1][db_k * 8 + sb0], f1a.x + f1a.y, sumd1);
                            sumd1 = fmaf(sc1 * smem.x.sxd[1][db_k * 8 + sb1], f1b.x + f1b.y, sumd1);
                            summ1 = fmaf(mn0, smem.x.sxs[1][db_k * 8 + sb0], summ1); summ1 = fmaf(mn1, smem.x.sxs[1][db_k * 8 + sb1], summ1);
                        }
                        acc0 = fmaf(dm.x, sumd0, fmaf(-dm.y, summ0, acc0));
                        acc1 = fmaf(dm.x, sumd1, fmaf(-dm.y, summ1, acc1));
                    }
                } else {
                    for (int db_k = kb; db_k < nb; db_k += 5) {
                        const int b = b0 + db_k;
                        const uint4 scdm = __ldg(Wscdm + (size_t) b * N + row);
                        const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&scdm.w));
                        const uint32_t sc_w[2] = {scdm.x & 0x3f3f3f3fu, (scdm.z & 0x0f0f0f0fu) | ((scdm.x >> 2) & 0x30303030u)};
                        const uint32_t m_w[2]  = {scdm.y & 0x3f3f3f3fu, ((scdm.z >> 4) & 0x0f0f0f0fu) | ((scdm.y >> 2) & 0x30303030u)};
                        float sumd[4] = {0.0f, 0.0f, 0.0f, 0.0f}, summ[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const uint4 qa = __ldg(Wqs + ((size_t) b * 8 + c * 2 + 0) * N + row);
                            const uint4 qb = __ldg(Wqs + ((size_t) b * 8 + c * 2 + 1) * N + row);
                            const uint32_t wv[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
                            const int sb0 = c * 2, sb1 = c * 2 + 1;
                            const float sc0 = (float) ((sc_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                            const float sc1 = (float) ((sc_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                            const float mn0 = (float) ((m_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                            const float mn1 = (float) ((m_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                            __half2 pa[4], pb[4];
#pragma unroll
                            for (int t = 0; t < 4; ++t) { pa[t] = __float2half2_rn(0.0f); pb[t] = __float2half2_rn(0.0f); }
                            const int x0_base = (db_k * 8 + sb0) * 16, x1_base = (db_k * 8 + sb1) * 16;
#pragma unroll
                            for (int u = 0; u < 8; ++u) {
                                const uint32_t w = wv[u];
                                const uint32_t lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu;
                                const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                                const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                                const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                                const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                                const __half2 wla = __hsub2(*reinterpret_cast<const __half2*>(&la), c1024);
                                const __half2 wlb = __hsub2(*reinterpret_cast<const __half2*>(&lb), c1024);
                                const __half2 wha = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                                const __half2 whb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
#pragma unroll
                                for (int t = 0; t < 4; ++t) {
                                    pa[t] = __hfma2(wla, smem.x.sxh[t][x0_base + u * 2 + 0], pa[t]);
                                    pa[t] = __hfma2(wlb, smem.x.sxh[t][x0_base + u * 2 + 1], pa[t]);
                                    pb[t] = __hfma2(wha, smem.x.sxh[t][x1_base + u * 2 + 0], pb[t]);
                                    pb[t] = __hfma2(whb, smem.x.sxh[t][x1_base + u * 2 + 1], pb[t]);
                                }
                            }
#pragma unroll
                            for (int t = 0; t < 4; ++t) {
                                const float2 fa = __half22float2(pa[t]), fb = __half22float2(pb[t]);
                                sumd[t] = fmaf(sc0 * smem.x.sxd[t][db_k * 8 + sb0], fa.x + fa.y, sumd[t]);
                                sumd[t] = fmaf(sc1 * smem.x.sxd[t][db_k * 8 + sb1], fb.x + fb.y, sumd[t]);
                                summ[t] = fmaf(mn0, smem.x.sxs[t][db_k * 8 + sb0], summ[t]);
                                summ[t] = fmaf(mn1, smem.x.sxs[t][db_k * 8 + sb1], summ[t]);
                            }
                        }
                        acc0 = fmaf(dm.x, sumd[0], fmaf(-dm.y, summ[0], acc0));
                        acc1 = fmaf(dm.x, sumd[1], fmaf(-dm.y, summ[1], acc1));
                        acc2 = fmaf(dm.x, sumd[2], fmaf(-dm.y, summ[2], acc2));
                        acc3 = fmaf(dm.x, sumd[3], fmaf(-dm.y, summ[3], acc3));
                    }
                }
            }
            if (b0 + 10 < K_blk) __syncthreads();
        }
        __syncthreads();
        smem.part[0][kb][r_loc] = acc0;
        if (ne > 1) smem.part[1][kb][r_loc] = acc1;
        if (ne > 2) smem.part[2][kb][r_loc] = acc2;
        if (ne > 3) smem.part[3][kb][r_loc] = acc3;
        __syncthreads();
        if (kb == 0 && valid_row) {
            float* dst_p = is_up ? up : gate;
            dst_p[(size_t) (eb + 0) * L.n_ff + r_out] =
                smem.part[0][0][r_loc] + smem.part[0][1][r_loc] + smem.part[0][2][r_loc] + smem.part[0][3][r_loc] + smem.part[0][4][r_loc];
            if (ne > 1)
                dst_p[(size_t) (eb + 1) * L.n_ff + r_out] =
                    smem.part[1][0][r_loc] + smem.part[1][1][r_loc] + smem.part[1][2][r_loc] + smem.part[1][3][r_loc] + smem.part[1][4][r_loc];
            if (ne > 2)
                dst_p[(size_t) (eb + 2) * L.n_ff + r_out] =
                    smem.part[2][0][r_loc] + smem.part[2][1][r_loc] + smem.part[2][2][r_loc] + smem.part[2][3][r_loc] + smem.part[2][4][r_loc];
            if (ne > 3)
                dst_p[(size_t) (eb + 3) * L.n_ff + r_out] =
                    smem.part[3][0][r_loc] + smem.part[3][1][r_loc] + smem.part[3][2][r_loc] + smem.part[3][3][r_loc] + smem.part[3][4][r_loc];
        }
        if (eb + 4 < e1) __syncthreads();
    }
}

// GP100 (sm_60) High-Occupancy Decode Q4_K K-Major SoA Block-Internal Split-K Gate+Up (eb += 2):
// Halves SMEM (23 KB -> 11.5 KB) and drops reg pressure so 3-4 CTAs/SM run concurrently under EP=4 decode!
template <bool FIXED_2560_640 = false>
__global__ void __launch_bounds__(320, 2) p100_q4k_gu_soa_bsk2_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok,
    const block_q8_1* __restrict__ xq, NativeExpertLayout L,
    float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    const int tid = threadIdx.x;
    const int r_loc = tid & 63;
    const int kb = tid >> 6;         // 0..4 (5 K-partitions)
    const int N_FF = FIXED_2560_640 ? 640 : (int) L.n_ff;
    const int N = FIXED_2560_640 ? 1280 : (2 * (int) L.n_ff);
    const int row = blockIdx.x * 64 + r_loc;
    const bool valid_row = FIXED_2560_640 ? true : (row < N);
    const bool is_up = row >= N_FF;
    const int r_out = is_up ? (row - N_FF) : row;
    const int K_blk = FIXED_2560_640 ? 10 : (int) (L.n_embd / 256);
    const int K_q8 = FIXED_2560_640 ? 80 : (int) (L.n_embd / 32);

    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint4* Wqs = reinterpret_cast<const uint4*>(blob);
    const uint4* Wscdm = reinterpret_cast<const uint4*>(blob + (size_t) 128 * K_blk * N);

    union Q4kSmem2 {
        struct {
            __align__(16) __half2 sxh[2][1280];
            float sxd[2][80];
            float sxs[2][80];
        } x;
        float part[2][5][64];
    };
    __shared__ Q4kSmem2 smem;
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);

    for (int eb = e0; eb < e1; eb += 2) {
        const int ne = (eb + 2 <= e1) ? 2 : 1;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 10) {
            const int nb = FIXED_2560_640 ? 10 : ((b0 + 10 <= K_blk) ? 10 : (K_blk - b0));
            const int nq8 = nb * 8, nh2 = nb * 128;
            for (int t = 0; t < ne; ++t) {
                const block_q8_1* xb = xq + (size_t) ent_tok[eb + t] * K_q8 + b0 * 8;
                if constexpr (FIXED_2560_640) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int i = tid + k * 320;
                        const int qb = i >> 4, sub = (i & 15) << 1;
                        smem.x.sxh[t][i] = iq_q81_pair_h2(xb[qb].qs + sub, c1152, inv128);
                    }
                } else {
                    for (int i = tid; i < nh2; i += 320) {
                        const int qb = i >> 4, sub = (i & 15) << 1;
                        smem.x.sxh[t][i] = iq_q81_pair_h2(xb[qb].qs + sub, c1152, inv128);
                    }
                }
                if (tid < nq8) {
                    const float2 ds = __half22float2(xb[tid].ds);
                    smem.x.sxd[t][tid] = ds.x * 128.0f;
                    smem.x.sxs[t][tid] = ds.y;
                }
            }
            __syncthreads();

            if (valid_row) {
                const uint4* wsc_p = Wscdm + (size_t) (b0 + kb) * N + row;
                const uint4* wqs_p = Wqs + ((size_t) (b0 + kb) * 8) * N + row;
                const uint4* x_p = reinterpret_cast<const uint4*>(&smem.x.sxh[0][kb * 128]);
                const float* sxd_p = &smem.x.sxd[0][kb * 8];
                const float* sxs_p = &smem.x.sxs[0][kb * 8];
                if (ne == 1) {
#pragma unroll 1
                    for (int db_k = kb; db_k < nb; db_k += 5, wsc_p += 5 * N, wqs_p += 40 * N, x_p += 160, sxd_p += 40, sxs_p += 40) {
                        const uint4 scdm = __ldg(wsc_p);
                        const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&scdm.w));
                        const uint32_t sc_w[2] = {scdm.x & 0x3f3f3f3fu, (scdm.z & 0x0f0f0f0fu) | ((scdm.x >> 2) & 0x30303030u)};
                        const uint32_t m_w[2]  = {scdm.y & 0x3f3f3f3fu, ((scdm.z >> 4) & 0x0f0f0f0fu) | ((scdm.y >> 2) & 0x30303030u)};
                        float sumd0 = 0.0f, summ0 = 0.0f;
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const uint4 qa = __ldg(wqs_p + (c * 2 + 0) * N);
                            const uint4 qb = __ldg(wqs_p + (c * 2 + 1) * N);
                            const uint32_t wv[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
                            const int sb0 = c * 2, sb1 = c * 2 + 1;
                            const float sc0 = (float) ((sc_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                            const float sc1 = (float) ((sc_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                            const float mn0 = (float) ((m_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                            const float mn1 = (float) ((m_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                            __half2 p0a = __float2half2_rn(0.0f), p0b = __float2half2_rn(0.0f);
#pragma unroll
                            for (int p = 0; p < 4; ++p) {
                                const uint4 x0v = x_p[c * 8 + p];
                                const uint4 x1v = x_p[c * 8 + 4 + p];
                                const uint32_t wa = wv[p * 2 + 0], wb = wv[p * 2 + 1];
                                const uint32_t loa = wa & 0x0f0f0f0fu, hia = (wa >> 4) & 0x0f0f0f0fu;
                                const uint32_t lob = wb & 0x0f0f0f0fu, hib = (wb >> 4) & 0x0f0f0f0fu;
                                const uint32_t la0 = __byte_perm(loa, 0x64006400u, 0x5150);
                                const uint32_t la1 = __byte_perm(loa, 0x64006400u, 0x5352);
                                const uint32_t lb0 = __byte_perm(lob, 0x64006400u, 0x5150);
                                const uint32_t lb1 = __byte_perm(lob, 0x64006400u, 0x5352);
                                const uint32_t ha0 = __byte_perm(hia, 0x64006400u, 0x5150);
                                const uint32_t ha1 = __byte_perm(hia, 0x64006400u, 0x5352);
                                const uint32_t hb0 = __byte_perm(hib, 0x64006400u, 0x5150);
                                const uint32_t hb1 = __byte_perm(hib, 0x64006400u, 0x5352);
                                p0a = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&la0), c1024), *reinterpret_cast<const __half2*>(&x0v.x), p0a);
                                p0a = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&la1), c1024), *reinterpret_cast<const __half2*>(&x0v.y), p0a);
                                p0a = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&lb0), c1024), *reinterpret_cast<const __half2*>(&x0v.z), p0a);
                                p0a = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&lb1), c1024), *reinterpret_cast<const __half2*>(&x0v.w), p0a);
                                p0b = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&ha0), c1024), *reinterpret_cast<const __half2*>(&x1v.x), p0b);
                                p0b = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&ha1), c1024), *reinterpret_cast<const __half2*>(&x1v.y), p0b);
                                p0b = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&hb0), c1024), *reinterpret_cast<const __half2*>(&x1v.z), p0b);
                                p0b = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&hb1), c1024), *reinterpret_cast<const __half2*>(&x1v.w), p0b);
                            }
                            const float2 f0a = __half22float2(p0a), f0b = __half22float2(p0b);
                            sumd0 = fmaf(sc0 * sxd_p[sb0], f0a.x + f0a.y, sumd0);
                            sumd0 = fmaf(sc1 * sxd_p[sb1], f0b.x + f0b.y, sumd0);
                            summ0 = fmaf(mn0, sxs_p[sb0], summ0);
                            summ0 = fmaf(mn1, sxs_p[sb1], summ0);
                        }
                        acc0 = fmaf(dm.x, sumd0, fmaf(-dm.y, summ0, acc0));
                    }
                } else {
#pragma unroll 1
                    for (int db_k = kb; db_k < nb; db_k += 5, wsc_p += 5 * N, wqs_p += 40 * N, x_p += 160, sxd_p += 40, sxs_p += 40) {
                        const uint4 scdm = __ldg(wsc_p);
                        const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&scdm.w));
                        const uint32_t sc_w[2] = {scdm.x & 0x3f3f3f3fu, (scdm.z & 0x0f0f0f0fu) | ((scdm.x >> 2) & 0x30303030u)};
                        const uint32_t m_w[2]  = {scdm.y & 0x3f3f3f3fu, ((scdm.z >> 4) & 0x0f0f0f0fu) | ((scdm.y >> 2) & 0x30303030u)};
                        float sumd0 = 0.0f, summ0 = 0.0f, sumd1 = 0.0f, summ1 = 0.0f;
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const uint4 qa = __ldg(wqs_p + (c * 2 + 0) * N);
                            const uint4 qb = __ldg(wqs_p + (c * 2 + 1) * N);
                            const int sb0 = c * 2, sb1 = c * 2 + 1;
                            const float sc0 = (float) ((sc_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                            const float sc1 = (float) ((sc_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                            const float mn0 = (float) ((m_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                            const float mn1 = (float) ((m_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                            __half2 p0a = __float2half2_rn(0.0f), p0b = __float2half2_rn(0.0f);
                            __half2 p1a = __float2half2_rn(0.0f), p1b = __float2half2_rn(0.0f);
                            auto step_pair = [&](uint32_t wa, uint32_t wb, int p) {
                                const uint4 x0a = x_p[c * 8 + p];
                                const uint4 x1a = x_p[320 + c * 8 + p];
                                const uint32_t loa = wa & 0x0f0f0f0fu, lob = wb & 0x0f0f0f0fu;
                                const uint32_t la0 = __byte_perm(loa, 0x64006400u, 0x5150);
                                const uint32_t la1 = __byte_perm(loa, 0x64006400u, 0x5352);
                                const uint32_t lb0 = __byte_perm(lob, 0x64006400u, 0x5150);
                                const uint32_t lb1 = __byte_perm(lob, 0x64006400u, 0x5352);
                                const __half2 wla0 = __hsub2(*reinterpret_cast<const __half2*>(&la0), c1024);
                                const __half2 wla1 = __hsub2(*reinterpret_cast<const __half2*>(&la1), c1024);
                                const __half2 wlb0 = __hsub2(*reinterpret_cast<const __half2*>(&lb0), c1024);
                                const __half2 wlb1 = __hsub2(*reinterpret_cast<const __half2*>(&lb1), c1024);
                                p0a = __hfma2(wla0, *reinterpret_cast<const __half2*>(&x0a.x), p0a);
                                p0a = __hfma2(wla1, *reinterpret_cast<const __half2*>(&x0a.y), p0a);
                                p0a = __hfma2(wlb0, *reinterpret_cast<const __half2*>(&x0a.z), p0a);
                                p0a = __hfma2(wlb1, *reinterpret_cast<const __half2*>(&x0a.w), p0a);
                                p1a = __hfma2(wla0, *reinterpret_cast<const __half2*>(&x1a.x), p1a);
                                p1a = __hfma2(wla1, *reinterpret_cast<const __half2*>(&x1a.y), p1a);
                                p1a = __hfma2(wlb0, *reinterpret_cast<const __half2*>(&x1a.z), p1a);
                                p1a = __hfma2(wlb1, *reinterpret_cast<const __half2*>(&x1a.w), p1a);
                                const uint4 x0b = x_p[c * 8 + 4 + p];
                                const uint4 x1b = x_p[320 + c * 8 + 4 + p];
                                const uint32_t hia = (wa >> 4) & 0x0f0f0f0fu, hib = (wb >> 4) & 0x0f0f0f0fu;
                                const uint32_t ha0 = __byte_perm(hia, 0x64006400u, 0x5150);
                                const uint32_t ha1 = __byte_perm(hia, 0x64006400u, 0x5352);
                                const uint32_t hb0 = __byte_perm(hib, 0x64006400u, 0x5150);
                                const uint32_t hb1 = __byte_perm(hib, 0x64006400u, 0x5352);
                                const __half2 wha0 = __hsub2(*reinterpret_cast<const __half2*>(&ha0), c1024);
                                const __half2 wha1 = __hsub2(*reinterpret_cast<const __half2*>(&ha1), c1024);
                                const __half2 whb0 = __hsub2(*reinterpret_cast<const __half2*>(&hb0), c1024);
                                const __half2 whb1 = __hsub2(*reinterpret_cast<const __half2*>(&hb1), c1024);
                                p0b = __hfma2(wha0, *reinterpret_cast<const __half2*>(&x0b.x), p0b);
                                p0b = __hfma2(wha1, *reinterpret_cast<const __half2*>(&x0b.y), p0b);
                                p0b = __hfma2(whb0, *reinterpret_cast<const __half2*>(&x0b.z), p0b);
                                p0b = __hfma2(whb1, *reinterpret_cast<const __half2*>(&x0b.w), p0b);
                                p1b = __hfma2(wha0, *reinterpret_cast<const __half2*>(&x1b.x), p1b);
                                p1b = __hfma2(wha1, *reinterpret_cast<const __half2*>(&x1b.y), p1b);
                                p1b = __hfma2(whb0, *reinterpret_cast<const __half2*>(&x1b.z), p1b);
                                p1b = __hfma2(whb1, *reinterpret_cast<const __half2*>(&x1b.w), p1b);
                            };
                            step_pair(qa.x, qa.y, 0); step_pair(qa.z, qa.w, 1);
                            step_pair(qb.x, qb.y, 2); step_pair(qb.z, qb.w, 3);
                            const float2 f0a = __half22float2(p0a), f0b = __half22float2(p0b);
                            const float2 f1a = __half22float2(p1a), f1b = __half22float2(p1b);
                            sumd0 = fmaf(sc0 * sxd_p[sb0], f0a.x + f0a.y, sumd0);
                            sumd0 = fmaf(sc1 * sxd_p[sb1], f0b.x + f0b.y, sumd0);
                            summ0 = fmaf(mn0, sxs_p[sb0], summ0); summ0 = fmaf(mn1, sxs_p[sb1], summ0);
                            sumd1 = fmaf(sc0 * sxd_p[80 + sb0], f1a.x + f1a.y, sumd1);
                            sumd1 = fmaf(sc1 * sxd_p[80 + sb1], f1b.x + f1b.y, sumd1);
                            summ1 = fmaf(mn0, sxs_p[80 + sb0], summ1); summ1 = fmaf(mn1, sxs_p[80 + sb1], summ1);
                        }
                        acc0 = fmaf(dm.x, sumd0, fmaf(-dm.y, summ0, acc0));
                        acc1 = fmaf(dm.x, sumd1, fmaf(-dm.y, summ1, acc1));
                    }
                }
            }
            if (!FIXED_2560_640 && b0 + 10 < K_blk) __syncthreads();
        }
        __syncthreads();
        smem.part[0][kb][r_loc] = acc0;
        if (ne > 1) smem.part[1][kb][r_loc] = acc1;
        __syncthreads();
        if (kb == 0 && valid_row) {
            float* dst_p = is_up ? up : gate;
            dst_p[(size_t) (eb + 0) * N_FF + r_out] =
                smem.part[0][0][r_loc] + smem.part[0][1][r_loc] + smem.part[0][2][r_loc] + smem.part[0][3][r_loc] + smem.part[0][4][r_loc];
            if (ne > 1)
                dst_p[(size_t) (eb + 1) * N_FF + r_out] =
                    smem.part[1][0][r_loc] + smem.part[1][1][r_loc] + smem.part[1][2][r_loc] + smem.part[1][3][r_loc] + smem.part[1][4][r_loc];
        }
        if (eb + 2 < e1) __syncthreads();
    }
}

// GP100 (sm_60) Q5_1 K-Major SoA Down with SWAR 0x00204081u 4-way bit expansion + 4-token register reuse (eb += 4).
__global__ void __launch_bounds__(64, 10) p100_q51_down_soa_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst,
    const block_q8_1* __restrict__ hq, NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    const int tid = threadIdx.x;
    const int N = (int) L.n_embd;
    const int K_blk = (int) (L.n_ff / 32);
    const int r = blockIdx.x * 64 + tid;

    const uint8_t* blob = (const uint8_t*) grp_ptr[g] + L.down_off;
    const uint4* Wqs = reinterpret_cast<const uint4*>(blob);
    const uint2* Wdmqh = reinterpret_cast<const uint2*>(blob + (size_t) 16 * K_blk * N);

    __shared__ __align__(16) __half2 sxh[4][20][16];
    __shared__ float sxd[4][20];
    __shared__ float sxs[4][20];
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);

    for (int eb = e0; eb < e1; eb += 4) {
        const int ne = (eb + 4 <= e1) ? 4 : (e1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = (b0 + 20 <= K_blk) ? 20 : (K_blk - b0);
            for (int t = 0; t < ne; ++t) {
                const block_q8_1* xt = hq + (size_t) (eb + t) * K_blk + b0;
                for (int i = tid; i < nb * 16; i += 64) {
                    const int db = i >> 4, sub = (i & 15) << 1;
                    sxh[t][db][i & 15] = iq_q81_pair_h2(xt[db].qs + sub, c1152, inv128);
                }
                if (tid < nb) {
                    const float2 ds = __half22float2(xt[tid].ds);
                    sxd[t][tid] = ds.x * 128.0f;
                    sxs[t][tid] = ds.y;
                }
            }
            if (ne == 3) {
                for (int i = tid; i < nb * 16; i += 64) sxh[3][i >> 4][i & 15] = __float2half2_rn(0.0f);
                if (tid < nb) { sxd[3][tid] = 0.0f; sxs[3][tid] = 0.0f; }
            }
            __syncthreads();
            if (r < N) {
                const uint4* wqs_p = Wqs + (size_t) b0 * N + r;
                const uint2* wdm_p = Wdmqh + (size_t) b0 * N + r;
                if (ne == 1) {
                    for (int db = 0; db < nb; ++db, wqs_p += N, wdm_p += N) {
                        const uint4 q4 = __ldg(wqs_p);
                        const uint2 dmqh = __ldg(wdm_p);
                        const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&dmqh.x));
                        const uint32_t qh = dmqh.y;
                        const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                        __half2 h0 = __float2half2_rn(0.0f);
                        const uint4* x0 = reinterpret_cast<const uint4*>(&sxh[0][db][0]);
                        const uint4 x0a = x0[0], x0b = x0[1], x0c = x0[2], x0d = x0[3];
                        const uint32_t xlo[8] = {x0a.x, x0a.y, x0a.z, x0a.w, x0b.x, x0b.y, x0b.z, x0b.w};
                        const uint32_t xhi[8] = {x0c.x, x0c.y, x0c.z, x0c.w, x0d.x, x0d.y, x0d.z, x0d.w};
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const uint32_t w = wv[u];
                            const uint32_t lo = (w & 0x0f0f0f0fu) | ((qh << (4 - u)) & 0x10101010u);
                            const uint32_t hi = ((w >> 4) & 0x0f0f0f0fu) | ((qh >> u) & 0x10101010u);
                            const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                            const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                            const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                            const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                            h0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&la), c1024), *reinterpret_cast<const __half2*>(&xlo[u * 2 + 0]), h0);
                            h0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&lb), c1024), *reinterpret_cast<const __half2*>(&xlo[u * 2 + 1]), h0);
                            h0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&ha), c1024), *reinterpret_cast<const __half2*>(&xhi[u * 2 + 0]), h0);
                            h0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&hb), c1024), *reinterpret_cast<const __half2*>(&xhi[u * 2 + 1]), h0);
                        }
                        const float2 f0 = __half22float2(h0);
                        acc0 = fmaf(dm.y, sxs[0][db], fmaf(dm.x * sxd[0][db], f0.x + f0.y, acc0));
                    }
                } else if (ne == 2) {
                    for (int db = 0; db < nb; ++db, wqs_p += N, wdm_p += N) {
                        const uint4 q4 = __ldg(wqs_p);
                        const uint2 dmqh = __ldg(wdm_p);
                        const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&dmqh.x));
                        const uint32_t qh = dmqh.y;
                        const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                        __half2 h0 = __float2half2_rn(0.0f), h1 = __float2half2_rn(0.0f);
#pragma unroll
                        for (int p = 0; p < 2; ++p) {
                            const uint4 x0l = *reinterpret_cast<const uint4*>(&sxh[0][db][p * 4]);
                            const uint4 x1l = *reinterpret_cast<const uint4*>(&sxh[1][db][p * 4]);
                            const uint4 x0h = *reinterpret_cast<const uint4*>(&sxh[0][db][8 + p * 4]);
                            const uint4 x1h = *reinterpret_cast<const uint4*>(&sxh[1][db][8 + p * 4]);
                            const uint32_t x0la[4] = {x0l.x, x0l.y, x0l.z, x0l.w};
                            const uint32_t x1la[4] = {x1l.x, x1l.y, x1l.z, x1l.w};
                            const uint32_t x0ha[4] = {x0h.x, x0h.y, x0h.z, x0h.w};
                            const uint32_t x1ha[4] = {x1h.x, x1h.y, x1h.z, x1h.w};
#pragma unroll
                            for (int q = 0; q < 2; ++q) {
                                const int u = p * 2 + q;
                                const uint32_t w = wv[u];
                                const uint32_t lo = (w & 0x0f0f0f0fu) | ((qh << (4 - u)) & 0x10101010u);
                                const uint32_t hi = ((w >> 4) & 0x0f0f0f0fu) | ((qh >> u) & 0x10101010u);
                                const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                                const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                                const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                                const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                                const __half2 wla = __hsub2(*reinterpret_cast<const __half2*>(&la), c1024);
                                const __half2 wlb = __hsub2(*reinterpret_cast<const __half2*>(&lb), c1024);
                                const __half2 wha = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                                const __half2 whb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
                                h0 = __hfma2(wla, *reinterpret_cast<const __half2*>(&x0la[q * 2 + 0]), h0);
                                h0 = __hfma2(wlb, *reinterpret_cast<const __half2*>(&x0la[q * 2 + 1]), h0);
                                h0 = __hfma2(wha, *reinterpret_cast<const __half2*>(&x0ha[q * 2 + 0]), h0);
                                h0 = __hfma2(whb, *reinterpret_cast<const __half2*>(&x0ha[q * 2 + 1]), h0);
                                h1 = __hfma2(wla, *reinterpret_cast<const __half2*>(&x1la[q * 2 + 0]), h1);
                                h1 = __hfma2(wlb, *reinterpret_cast<const __half2*>(&x1la[q * 2 + 1]), h1);
                                h1 = __hfma2(wha, *reinterpret_cast<const __half2*>(&x1ha[q * 2 + 0]), h1);
                                h1 = __hfma2(whb, *reinterpret_cast<const __half2*>(&x1ha[q * 2 + 1]), h1);
                            }
                        }
                        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
                        acc0 = fmaf(dm.y, sxs[0][db], fmaf(dm.x * sxd[0][db], f0.x + f0.y, acc0));
                        acc1 = fmaf(dm.y, sxs[1][db], fmaf(dm.x * sxd[1][db], f1.x + f1.y, acc1));
                    }
                } else {
                    for (int db = 0; db < nb; ++db, wqs_p += N, wdm_p += N) {
                        const uint4 q4 = __ldg(wqs_p);
                        const uint2 dmqh = __ldg(wdm_p);
                        const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&dmqh.x));
                        const uint32_t qh = dmqh.y;
                        const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                        __half2 h0 = __float2half2_rn(0.0f), h1 = h0, h2 = h0, h3 = h0;
#pragma unroll
                        for (int p = 0; p < 2; ++p) {
                            const uint4 x0l = *reinterpret_cast<const uint4*>(&sxh[0][db][p * 4]);
                            const uint4 x1l = *reinterpret_cast<const uint4*>(&sxh[1][db][p * 4]);
                            const uint4 x2l = *reinterpret_cast<const uint4*>(&sxh[2][db][p * 4]);
                            const uint4 x3l = *reinterpret_cast<const uint4*>(&sxh[3][db][p * 4]);
                            const uint4 x0h = *reinterpret_cast<const uint4*>(&sxh[0][db][8 + p * 4]);
                            const uint4 x1h = *reinterpret_cast<const uint4*>(&sxh[1][db][8 + p * 4]);
                            const uint4 x2h = *reinterpret_cast<const uint4*>(&sxh[2][db][8 + p * 4]);
                            const uint4 x3h = *reinterpret_cast<const uint4*>(&sxh[3][db][8 + p * 4]);
                            const uint32_t x0la[4] = {x0l.x, x0l.y, x0l.z, x0l.w};
                            const uint32_t x1la[4] = {x1l.x, x1l.y, x1l.z, x1l.w};
                            const uint32_t x2la[4] = {x2l.x, x2l.y, x2l.z, x2l.w};
                            const uint32_t x3la[4] = {x3l.x, x3l.y, x3l.z, x3l.w};
                            const uint32_t x0ha[4] = {x0h.x, x0h.y, x0h.z, x0h.w};
                            const uint32_t x1ha[4] = {x1h.x, x1h.y, x1h.z, x1h.w};
                            const uint32_t x2ha[4] = {x2h.x, x2h.y, x2h.z, x2h.w};
                            const uint32_t x3ha[4] = {x3h.x, x3h.y, x3h.z, x3h.w};
#pragma unroll
                            for (int q = 0; q < 2; ++q) {
                                const int u = p * 2 + q;
                                const uint32_t w = wv[u];
                                const uint32_t lo = (w & 0x0f0f0f0fu) | ((qh << (4 - u)) & 0x10101010u);
                                const uint32_t hi = ((w >> 4) & 0x0f0f0f0fu) | ((qh >> u) & 0x10101010u);
                                const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                                const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                                const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                                const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                                const __half2 wla = __hsub2(*reinterpret_cast<const __half2*>(&la), c1024);
                                const __half2 wlb = __hsub2(*reinterpret_cast<const __half2*>(&lb), c1024);
                                const __half2 wha = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                                const __half2 whb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
                                h0 = __hfma2(wla, *reinterpret_cast<const __half2*>(&x0la[q * 2 + 0]), h0);
                                h0 = __hfma2(wlb, *reinterpret_cast<const __half2*>(&x0la[q * 2 + 1]), h0);
                                h0 = __hfma2(wha, *reinterpret_cast<const __half2*>(&x0ha[q * 2 + 0]), h0);
                                h0 = __hfma2(whb, *reinterpret_cast<const __half2*>(&x0ha[q * 2 + 1]), h0);
                                h1 = __hfma2(wla, *reinterpret_cast<const __half2*>(&x1la[q * 2 + 0]), h1);
                                h1 = __hfma2(wlb, *reinterpret_cast<const __half2*>(&x1la[q * 2 + 1]), h1);
                                h1 = __hfma2(wha, *reinterpret_cast<const __half2*>(&x1ha[q * 2 + 0]), h1);
                                h1 = __hfma2(whb, *reinterpret_cast<const __half2*>(&x1ha[q * 2 + 1]), h1);
                                h2 = __hfma2(wla, *reinterpret_cast<const __half2*>(&x2la[q * 2 + 0]), h2);
                                h2 = __hfma2(wlb, *reinterpret_cast<const __half2*>(&x2la[q * 2 + 1]), h2);
                                h2 = __hfma2(wha, *reinterpret_cast<const __half2*>(&x2ha[q * 2 + 0]), h2);
                                h2 = __hfma2(whb, *reinterpret_cast<const __half2*>(&x2ha[q * 2 + 1]), h2);
                                h3 = __hfma2(wla, *reinterpret_cast<const __half2*>(&x3la[q * 2 + 0]), h3);
                                h3 = __hfma2(wlb, *reinterpret_cast<const __half2*>(&x3la[q * 2 + 1]), h3);
                                h3 = __hfma2(wha, *reinterpret_cast<const __half2*>(&x3ha[q * 2 + 0]), h3);
                                h3 = __hfma2(whb, *reinterpret_cast<const __half2*>(&x3ha[q * 2 + 1]), h3);
                            }
                        }
                        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1), f2 = __half22float2(h2), f3 = __half22float2(h3);
                        acc0 = fmaf(dm.y, sxs[0][db], fmaf(dm.x * sxd[0][db], f0.x + f0.y, acc0));
                        acc1 = fmaf(dm.y, sxs[1][db], fmaf(dm.x * sxd[1][db], f1.x + f1.y, acc1));
                        acc2 = fmaf(dm.y, sxs[2][db], fmaf(dm.x * sxd[2][db], f2.x + f2.y, acc2));
                        acc3 = fmaf(dm.y, sxs[3][db], fmaf(dm.x * sxd[3][db], f3.x + f3.y, acc3));
                    }
                }
            }
            if (b0 + 20 < K_blk) __syncthreads();
        }
        if (r < N) {
            out[(size_t) ent_dst[eb + 0] * N + r] = acc0;
            if (ne > 1) out[(size_t) ent_dst[eb + 1] * N + r] = acc1;
            if (ne > 2) out[(size_t) ent_dst[eb + 2] * N + r] = acc2;
            if (ne > 3) out[(size_t) ent_dst[eb + 3] * N + r] = acc3;
        }
        if (eb + 4 < e1) __syncthreads();
    }
}

// GP100 (sm_60) Q8_0 K-Major SoA Down with 4-token register reuse (eb += 4).
__global__ void __launch_bounds__(64, 10) p100_q80_down_soa_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst,
    const block_q8_1* __restrict__ hq, NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    const int tid = threadIdx.x;
    const int N = (int) L.n_embd;
    const int K_blk = (int) (L.n_ff / 32);
    const int r = blockIdx.x * 64 + tid;

    const uint8_t* blob = (const uint8_t*) grp_ptr[g] + L.down_off;
    const uint4* Wqs = reinterpret_cast<const uint4*>(blob);
    const __half* Wd = reinterpret_cast<const __half*>(blob + (size_t) 32 * K_blk * N);

    __shared__ __half2 sxh[4][20][16];
    __shared__ float sxd[4][20];
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);

    for (int eb = e0; eb < e1; eb += 4) {
        const int ne = (eb + 4 <= e1) ? 4 : (e1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = (b0 + 20 <= K_blk) ? 20 : (K_blk - b0);
            for (int t = 0; t < ne; ++t) {
                const block_q8_1* xt = hq + (size_t) (eb + t) * K_blk + b0;
                for (int i = tid; i < nb * 16; i += 64) {
                    const int db = i >> 4, sub = (i & 15) << 1;
                    sxh[t][db][i & 15] = iq_q81_pair_h2(xt[db].qs + sub, c1152, inv128);
                }
                if (tid < nb) sxd[t][tid] = __low2float(xt[tid].ds) * 128.0f;
            }
            if (ne == 3) {
                for (int i = tid; i < nb * 16; i += 64) sxh[3][i >> 4][i & 15] = __float2half2_rn(0.0f);
                if (tid < nb) sxd[3][tid] = 0.0f;
            }
            __syncthreads();
            if (r < N) {
                if (ne == 1) {
                    for (int db = 0; db < nb; ++db) {
                        const int b = b0 + db;
                        const float d = __half2float(__ldg(Wd + (size_t) b * N + r));
                        __half2 h0 = __float2half2_rn(0.0f);
#pragma unroll
                        for (int hf = 0; hf < 2; ++hf) {
                            const uint4 qv = __ldg(Wqs + ((size_t) b * 2 + hf) * N + r);
                            const uint32_t ws[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
                            for (int u = 0; u < 4; ++u) {
                                const uint32_t vx = ws[u];
                                const uint32_t a = __byte_perm(vx, 0x64006400u, 0x5150);
                                const uint32_t bv = __byte_perm(vx, 0x64006400u, 0x5352);
                                const int xo = hf * 8 + u * 2;
                                h0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&a), c1152), sxh[0][db][xo + 0], h0);
                                h0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&bv), c1152), sxh[0][db][xo + 1], h0);
                            }
                        }
                        const float2 f0 = __half22float2(h0);
                        acc0 = fmaf(d * sxd[0][db], f0.x + f0.y, acc0);
                    }
                } else if (ne == 2) {
                    for (int db = 0; db < nb; ++db) {
                        const int b = b0 + db;
                        const float d = __half2float(__ldg(Wd + (size_t) b * N + r));
                        __half2 h0 = __float2half2_rn(0.0f), h1 = h0;
#pragma unroll
                        for (int hf = 0; hf < 2; ++hf) {
                            const uint4 qv = __ldg(Wqs + ((size_t) b * 2 + hf) * N + r);
                            const uint32_t ws[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
                            for (int u = 0; u < 4; ++u) {
                                const uint32_t vx = ws[u];
                                const uint32_t a = __byte_perm(vx, 0x64006400u, 0x5150);
                                const uint32_t bv = __byte_perm(vx, 0x64006400u, 0x5352);
                                const __half2 wa = __hsub2(*reinterpret_cast<const __half2*>(&a), c1152);
                                const __half2 wb = __hsub2(*reinterpret_cast<const __half2*>(&bv), c1152);
                                const int xo = hf * 8 + u * 2;
                                h0 = __hfma2(wa, sxh[0][db][xo + 0], h0); h0 = __hfma2(wb, sxh[0][db][xo + 1], h0);
                                h1 = __hfma2(wa, sxh[1][db][xo + 0], h1); h1 = __hfma2(wb, sxh[1][db][xo + 1], h1);
                            }
                        }
                        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
                        acc0 = fmaf(d * sxd[0][db], f0.x + f0.y, acc0);
                        acc1 = fmaf(d * sxd[1][db], f1.x + f1.y, acc1);
                    }
                } else {
                    for (int db = 0; db < nb; ++db) {
                        const int b = b0 + db;
                        const float d = __half2float(__ldg(Wd + (size_t) b * N + r));
                        __half2 h0 = __float2half2_rn(0.0f), h1 = h0, h2 = h0, h3 = h0;
#pragma unroll
                        for (int hf = 0; hf < 2; ++hf) {
                            const uint4 qv = __ldg(Wqs + ((size_t) b * 2 + hf) * N + r);
                            const uint32_t ws[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
                            for (int u = 0; u < 4; ++u) {
                                const uint32_t vx = ws[u];
                                const uint32_t a = __byte_perm(vx, 0x64006400u, 0x5150);
                                const uint32_t bv = __byte_perm(vx, 0x64006400u, 0x5352);
                                const __half2 wa = __hsub2(*reinterpret_cast<const __half2*>(&a), c1152);
                                const __half2 wb = __hsub2(*reinterpret_cast<const __half2*>(&bv), c1152);
                                const int xo = hf * 8 + u * 2;
                                h0 = __hfma2(wa, sxh[0][db][xo + 0], h0); h0 = __hfma2(wb, sxh[0][db][xo + 1], h0);
                                h1 = __hfma2(wa, sxh[1][db][xo + 0], h1); h1 = __hfma2(wb, sxh[1][db][xo + 1], h1);
                                h2 = __hfma2(wa, sxh[2][db][xo + 0], h2); h2 = __hfma2(wb, sxh[2][db][xo + 1], h2);
                                h3 = __hfma2(wa, sxh[3][db][xo + 0], h3); h3 = __hfma2(wb, sxh[3][db][xo + 1], h3);
                            }
                        }
                        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1), f2 = __half22float2(h2), f3 = __half22float2(h3);
                        acc0 = fmaf(d * sxd[0][db], f0.x + f0.y, acc0);
                        acc1 = fmaf(d * sxd[1][db], f1.x + f1.y, acc1);
                        acc2 = fmaf(d * sxd[2][db], f2.x + f2.y, acc2);
                        acc3 = fmaf(d * sxd[3][db], f3.x + f3.y, acc3);
                    }
                }
            }
            if (b0 + 20 < K_blk) __syncthreads();
        }
        if (r < N) {
            out[(size_t) ent_dst[eb + 0] * N + r] = acc0;
            if (ne > 1) out[(size_t) ent_dst[eb + 1] * N + r] = acc1;
            if (ne > 2) out[(size_t) ent_dst[eb + 2] * N + r] = acc2;
            if (ne > 3) out[(size_t) ent_dst[eb + 3] * N + r] = acc3;
        }
        if (eb + 4 < e1) __syncthreads();
    }
}

// GP100 (sm_60) Q5_1 K-Major SoA Down Split-K5 (320t = 5 K-parts x 64 rows, 4 Q5_1 blocks/thread for n_ff=640!):
// 1.50-1.66x faster under TP=4 EP=4 (ng=2..5 groups/GPU), REG:64, STACK:0, 5.76 KB union SMEM!
template <bool FIXED_2560_640 = false>
__global__ void __launch_bounds__(320, 3) p100_q51_down_soa_sk5_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst,
    const block_q8_1* __restrict__ hq, NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    const int tid = threadIdx.x;
    const int r_loc = tid & 63;
    const int kp = tid >> 6;         // 0..4 (5 K-parts)
    const int N = FIXED_2560_640 ? 2560 : (int) L.n_embd;
    const int K_blk = FIXED_2560_640 ? 20 : (int) (L.n_ff / 32);
    const int r = blockIdx.x * 64 + r_loc;
    const bool valid_row = FIXED_2560_640 ? true : (r < N);

    const uint8_t* blob = (const uint8_t*) grp_ptr[g] + L.down_off;
    const uint4* Wqs = reinterpret_cast<const uint4*>(blob);
    const uint2* Wdmqh = reinterpret_cast<const uint2*>(blob + (size_t) 16 * K_blk * N);

    union Q51Sk5Smem {
        struct {
            __align__(16) __half2 sxh[4][20][16];
            float sxd[4][20];
            float sxs[4][20];
        } x;
        float part[4][5][64];
    };
    __shared__ Q51Sk5Smem sm;
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);

    for (int eb = e0; eb < e1; eb += 4) {
        const int ne = (eb + 4 <= e1) ? 4 : (e1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = FIXED_2560_640 ? 20 : ((b0 + 20 <= K_blk) ? 20 : (K_blk - b0));
            if constexpr (FIXED_2560_640) {
                const int db = tid >> 4, sub = (tid & 15) << 1;
                for (int t = 0; t < ne; ++t) {
                    const block_q8_1* xt = hq + (size_t) (eb + t) * 20;
                    sm.x.sxh[t][db][tid & 15] = iq_q81_pair_h2(xt[db].qs + sub, c1152, inv128);
                    if (tid < 20) {
                        const float2 ds = __half22float2(xt[tid].ds);
                        sm.x.sxd[t][tid] = ds.x * 128.0f;
                        sm.x.sxs[t][tid] = ds.y;
                    }
                }
                if (ne == 3) {
                    sm.x.sxh[3][db][tid & 15] = __float2half2_rn(0.0f);
                    if (tid < 20) { sm.x.sxd[3][tid] = 0.0f; sm.x.sxs[3][tid] = 0.0f; }
                }
            } else {
                for (int t = 0; t < ne; ++t) {
                    const block_q8_1* xt = hq + (size_t) (eb + t) * K_blk + b0;
                    for (int i = tid; i < nb * 16; i += 320) {
                        const int db = i >> 4, sub = (i & 15) << 1;
                        sm.x.sxh[t][db][i & 15] = iq_q81_pair_h2(xt[db].qs + sub, c1152, inv128);
                    }
                    if (tid < nb) {
                        const float2 ds = __half22float2(xt[tid].ds);
                        sm.x.sxd[t][tid] = ds.x * 128.0f;
                        sm.x.sxs[t][tid] = ds.y;
                    }
                }
                if (ne == 3) {
                    for (int i = tid; i < nb * 16; i += 320) sm.x.sxh[3][i >> 4][i & 15] = __float2half2_rn(0.0f);
                    if (tid < nb) { sm.x.sxd[3][tid] = 0.0f; sm.x.sxs[3][tid] = 0.0f; }
                }
            }
            __syncthreads();
            if (valid_row) {
                const uint4* wqs_p = Wqs + (size_t) (b0 + kp) * N + r;
                const uint2* wdm_p = Wdmqh + (size_t) (b0 + kp) * N + r;
                const uint4* x_p = reinterpret_cast<const uint4*>(&sm.x.sxh[0][kp][0]);
                const float* sxd_p = &sm.x.sxd[0][kp];
                const float* sxs_p = &sm.x.sxs[0][kp];
                if (ne == 1) {
#pragma unroll 1
                    for (int db = kp; db < nb; db += 5, wqs_p += 5 * N, wdm_p += 5 * N, x_p += 20, sxd_p += 5, sxs_p += 5) {
                        const uint4 q4 = __ldg(wqs_p);
                        const uint2 dmqh = __ldg(wdm_p);
                        const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&dmqh.x));
                        const uint32_t qh = dmqh.y;
                        const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                        __half2 h0 = __float2half2_rn(0.0f);
                        const uint4 x0a = x_p[0], x0b = x_p[1], x0c = x_p[2], x0d = x_p[3];
                        const uint32_t xlo[8] = {x0a.x, x0a.y, x0a.z, x0a.w, x0b.x, x0b.y, x0b.z, x0b.w};
                        const uint32_t xhi[8] = {x0c.x, x0c.y, x0c.z, x0c.w, x0d.x, x0d.y, x0d.z, x0d.w};
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const uint32_t w = wv[u];
                            const uint32_t lo = (w & 0x0f0f0f0fu) | ((qh << (4 - u)) & 0x10101010u);
                            const uint32_t hi = ((w >> 4) & 0x0f0f0f0fu) | ((qh >> u) & 0x10101010u);
                            const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                            const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                            const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                            const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                            h0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&la), c1024), *reinterpret_cast<const __half2*>(&xlo[u * 2 + 0]), h0);
                            h0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&lb), c1024), *reinterpret_cast<const __half2*>(&xlo[u * 2 + 1]), h0);
                            h0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&ha), c1024), *reinterpret_cast<const __half2*>(&xhi[u * 2 + 0]), h0);
                            h0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&hb), c1024), *reinterpret_cast<const __half2*>(&xhi[u * 2 + 1]), h0);
                        }
                        const float2 f0 = __half22float2(h0);
                        acc0 = fmaf(dm.y, sxs_p[0], fmaf(dm.x * sxd_p[0], f0.x + f0.y, acc0));
                    }
                } else if (ne == 2) {
#pragma unroll 1
                    for (int db = kp; db < nb; db += 5, wqs_p += 5 * N, wdm_p += 5 * N, x_p += 20, sxd_p += 5, sxs_p += 5) {
                        const uint4 q4 = __ldg(wqs_p);
                        const uint2 dmqh = __ldg(wdm_p);
                        const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&dmqh.x));
                        const uint32_t qh = dmqh.y;
                        const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                        __half2 h0 = __float2half2_rn(0.0f), h1 = h0;
#pragma unroll
                        for (int p = 0; p < 2; ++p) {
                            const uint4 x0l = x_p[p];
                            const uint4 x1l = x_p[80 + p];
                            const uint4 x0h = x_p[2 + p];
                            const uint4 x1h = x_p[82 + p];
                            const uint32_t x0la[4] = {x0l.x, x0l.y, x0l.z, x0l.w};
                            const uint32_t x1la[4] = {x1l.x, x1l.y, x1l.z, x1l.w};
                            const uint32_t x0ha[4] = {x0h.x, x0h.y, x0h.z, x0h.w};
                            const uint32_t x1ha[4] = {x1h.x, x1h.y, x1h.z, x1h.w};
#pragma unroll
                            for (int q = 0; q < 2; ++q) {
                                const int u = p * 2 + q;
                                const uint32_t w = wv[u];
                                const uint32_t lo = (w & 0x0f0f0f0fu) | ((qh << (4 - u)) & 0x10101010u);
                                const uint32_t hi = ((w >> 4) & 0x0f0f0f0fu) | ((qh >> u) & 0x10101010u);
                                const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                                const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                                const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                                const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                                const __half2 wla = __hsub2(*reinterpret_cast<const __half2*>(&la), c1024);
                                const __half2 wlb = __hsub2(*reinterpret_cast<const __half2*>(&lb), c1024);
                                const __half2 wha = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                                const __half2 whb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
                                h0 = __hfma2(wla, *reinterpret_cast<const __half2*>(&x0la[q * 2 + 0]), h0);
                                h0 = __hfma2(wlb, *reinterpret_cast<const __half2*>(&x0la[q * 2 + 1]), h0);
                                h0 = __hfma2(wha, *reinterpret_cast<const __half2*>(&x0ha[q * 2 + 0]), h0);
                                h0 = __hfma2(whb, *reinterpret_cast<const __half2*>(&x0ha[q * 2 + 1]), h0);
                                h1 = __hfma2(wla, *reinterpret_cast<const __half2*>(&x1la[q * 2 + 0]), h1);
                                h1 = __hfma2(wlb, *reinterpret_cast<const __half2*>(&x1la[q * 2 + 1]), h1);
                                h1 = __hfma2(wha, *reinterpret_cast<const __half2*>(&x1ha[q * 2 + 0]), h1);
                                h1 = __hfma2(whb, *reinterpret_cast<const __half2*>(&x1ha[q * 2 + 1]), h1);
                            }
                        }
                        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
                        acc0 = fmaf(dm.y, sxs_p[0], fmaf(dm.x * sxd_p[0], f0.x + f0.y, acc0));
                        acc1 = fmaf(dm.y, sxs_p[20], fmaf(dm.x * sxd_p[20], f1.x + f1.y, acc1));
                    }
                } else {
#pragma unroll 1
                    for (int db = kp; db < nb; db += 5, wqs_p += 5 * N, wdm_p += 5 * N, x_p += 20, sxd_p += 5, sxs_p += 5) {
                        const uint4 q4 = __ldg(wqs_p);
                        const uint2 dmqh = __ldg(wdm_p);
                        const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&dmqh.x));
                        const uint32_t qh = dmqh.y;
                        const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                        __half2 h0 = __float2half2_rn(0.0f), h1 = h0, h2 = h0, h3 = h0;
#pragma unroll
                        for (int p = 0; p < 2; ++p) {
                            {
                                const uint4 x0l = x_p[p];
                                const uint4 x1l = x_p[80 + p];
                                const uint4 x2l = x_p[160 + p];
                                const uint4 x3l = x_p[240 + p];
                                const uint32_t x0la[4] = {x0l.x, x0l.y, x0l.z, x0l.w};
                                const uint32_t x1la[4] = {x1l.x, x1l.y, x1l.z, x1l.w};
                                const uint32_t x2la[4] = {x2l.x, x2l.y, x2l.z, x2l.w};
                                const uint32_t x3la[4] = {x3l.x, x3l.y, x3l.z, x3l.w};
#pragma unroll
                                for (int q = 0; q < 2; ++q) {
                                    const int u = p * 2 + q;
                                    const uint32_t lo = (wv[u] & 0x0f0f0f0fu) | ((qh << (4 - u)) & 0x10101010u);
                                    const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                                    const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                                    const __half2 wla = __hsub2(*reinterpret_cast<const __half2*>(&la), c1024);
                                    const __half2 wlb = __hsub2(*reinterpret_cast<const __half2*>(&lb), c1024);
                                    h0 = __hfma2(wla, *reinterpret_cast<const __half2*>(&x0la[q * 2 + 0]), h0);
                                    h0 = __hfma2(wlb, *reinterpret_cast<const __half2*>(&x0la[q * 2 + 1]), h0);
                                    h1 = __hfma2(wla, *reinterpret_cast<const __half2*>(&x1la[q * 2 + 0]), h1);
                                    h1 = __hfma2(wlb, *reinterpret_cast<const __half2*>(&x1la[q * 2 + 1]), h1);
                                    h2 = __hfma2(wla, *reinterpret_cast<const __half2*>(&x2la[q * 2 + 0]), h2);
                                    h2 = __hfma2(wlb, *reinterpret_cast<const __half2*>(&x2la[q * 2 + 1]), h2);
                                    h3 = __hfma2(wla, *reinterpret_cast<const __half2*>(&x3la[q * 2 + 0]), h3);
                                    h3 = __hfma2(wlb, *reinterpret_cast<const __half2*>(&x3la[q * 2 + 1]), h3);
                                }
                            }
                            {
                                const uint4 x0h = x_p[2 + p];
                                const uint4 x1h = x_p[82 + p];
                                const uint4 x2h = x_p[162 + p];
                                const uint4 x3h = x_p[242 + p];
                                const uint32_t x0ha[4] = {x0h.x, x0h.y, x0h.z, x0h.w};
                                const uint32_t x1ha[4] = {x1h.x, x1h.y, x1h.z, x1h.w};
                                const uint32_t x2ha[4] = {x2h.x, x2h.y, x2h.z, x2h.w};
                                const uint32_t x3ha[4] = {x3h.x, x3h.y, x3h.z, x3h.w};
#pragma unroll
                                for (int q = 0; q < 2; ++q) {
                                    const int u = p * 2 + q;
                                    const uint32_t hi = ((wv[u] >> 4) & 0x0f0f0f0fu) | ((qh >> u) & 0x10101010u);
                                    const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                                    const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                                    const __half2 wha = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                                    const __half2 whb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
                                    h0 = __hfma2(wha, *reinterpret_cast<const __half2*>(&x0ha[q * 2 + 0]), h0);
                                    h0 = __hfma2(whb, *reinterpret_cast<const __half2*>(&x0ha[q * 2 + 1]), h0);
                                    h1 = __hfma2(wha, *reinterpret_cast<const __half2*>(&x1ha[q * 2 + 0]), h1);
                                    h1 = __hfma2(whb, *reinterpret_cast<const __half2*>(&x1ha[q * 2 + 1]), h1);
                                    h2 = __hfma2(wha, *reinterpret_cast<const __half2*>(&x2ha[q * 2 + 0]), h2);
                                    h2 = __hfma2(whb, *reinterpret_cast<const __half2*>(&x2ha[q * 2 + 1]), h2);
                                    h3 = __hfma2(wha, *reinterpret_cast<const __half2*>(&x3ha[q * 2 + 0]), h3);
                                    h3 = __hfma2(whb, *reinterpret_cast<const __half2*>(&x3ha[q * 2 + 1]), h3);
                                }
                            }
                        }
                        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1), f2 = __half22float2(h2), f3 = __half22float2(h3);
                        acc0 = fmaf(dm.y, sxs_p[0], fmaf(dm.x * sxd_p[0], f0.x + f0.y, acc0));
                        acc1 = fmaf(dm.y, sxs_p[20], fmaf(dm.x * sxd_p[20], f1.x + f1.y, acc1));
                        acc2 = fmaf(dm.y, sxs_p[40], fmaf(dm.x * sxd_p[40], f2.x + f2.y, acc2));
                        acc3 = fmaf(dm.y, sxs_p[60], fmaf(dm.x * sxd_p[60], f3.x + f3.y, acc3));
                    }
                }
            }
            if (!FIXED_2560_640 && b0 + 20 < K_blk) __syncthreads();
        }
        __syncthreads();
        sm.part[0][kp][r_loc] = acc0;
        if (ne > 1) sm.part[1][kp][r_loc] = acc1;
        if (ne > 2) sm.part[2][kp][r_loc] = acc2;
        if (ne > 3) sm.part[3][kp][r_loc] = acc3;
        __syncthreads();
        if (kp == 0 && valid_row) {
            out[(size_t) ent_dst[eb + 0] * N + r] =
                sm.part[0][0][r_loc] + sm.part[0][1][r_loc] + sm.part[0][2][r_loc] + sm.part[0][3][r_loc] + sm.part[0][4][r_loc];
            if (ne > 1)
                out[(size_t) ent_dst[eb + 1] * N + r] =
                    sm.part[1][0][r_loc] + sm.part[1][1][r_loc] + sm.part[1][2][r_loc] + sm.part[1][3][r_loc] + sm.part[1][4][r_loc];
            if (ne > 2)
                out[(size_t) ent_dst[eb + 2] * N + r] =
                    sm.part[2][0][r_loc] + sm.part[2][1][r_loc] + sm.part[2][2][r_loc] + sm.part[2][3][r_loc] + sm.part[2][4][r_loc];
            if (ne > 3)
                out[(size_t) ent_dst[eb + 3] * N + r] =
                    sm.part[3][0][r_loc] + sm.part[3][1][r_loc] + sm.part[3][2][r_loc] + sm.part[3][3][r_loc] + sm.part[3][4][r_loc];
        }
        if (eb + 4 < e1) __syncthreads();
    }
}

// GP100 (sm_60) Q8_0 K-Major SoA Down Split-K5 (320t = 5 K-parts x 64 rows, 4 Q8_0 blocks/thread for n_ff=640!).
template <bool FIXED_2560_640 = false>
__global__ void __launch_bounds__(320, 3) p100_q80_down_soa_sk5_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst,
    const block_q8_1* __restrict__ hq, NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    const int tid = threadIdx.x;
    const int r_loc = tid & 63;
    const int kp = tid >> 6;
    const int N = FIXED_2560_640 ? 2560 : (int) L.n_embd;
    const int K_blk = FIXED_2560_640 ? 20 : (int) (L.n_ff / 32);
    const int r = blockIdx.x * 64 + r_loc;
    const bool valid_row = FIXED_2560_640 ? true : (r < N);

    const uint8_t* blob = (const uint8_t*) grp_ptr[g] + L.down_off;
    const uint4* Wqs = reinterpret_cast<const uint4*>(blob);
    const __half* Wd = reinterpret_cast<const __half*>(blob + (size_t) 32 * K_blk * N);

    union Q80Sk5Smem {
        struct {
            __align__(16) __half2 sxh[4][20][16];
            float sxd[4][20];
        } x;
        float part[4][5][64];
    };
    __shared__ Q80Sk5Smem sm;
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);

    for (int eb = e0; eb < e1; eb += 4) {
        const int ne = (eb + 4 <= e1) ? 4 : (e1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = FIXED_2560_640 ? 20 : ((b0 + 20 <= K_blk) ? 20 : (K_blk - b0));
            if constexpr (FIXED_2560_640) {
                const int db = tid >> 4, sub = (tid & 15) << 1;
                for (int t = 0; t < ne; ++t) {
                    const block_q8_1* xt = hq + (size_t) (eb + t) * 20;
                    sm.x.sxh[t][db][tid & 15] = iq_q81_pair_h2(xt[db].qs + sub, c1152, inv128);
                    if (tid < 20) sm.x.sxd[t][tid] = __low2float(xt[tid].ds) * 128.0f;
                }
                if (ne == 3) {
                    sm.x.sxh[3][db][tid & 15] = __float2half2_rn(0.0f);
                    if (tid < 20) sm.x.sxd[3][tid] = 0.0f;
                }
            } else {
                for (int t = 0; t < ne; ++t) {
                    const block_q8_1* xt = hq + (size_t) (eb + t) * K_blk + b0;
                    for (int i = tid; i < nb * 16; i += 320) {
                        const int db = i >> 4, sub = (i & 15) << 1;
                        sm.x.sxh[t][db][i & 15] = iq_q81_pair_h2(xt[db].qs + sub, c1152, inv128);
                    }
                    if (tid < nb) sm.x.sxd[t][tid] = __low2float(xt[tid].ds) * 128.0f;
                }
                if (ne == 3) {
                    for (int i = tid; i < nb * 16; i += 320) sm.x.sxh[3][i >> 4][i & 15] = __float2half2_rn(0.0f);
                    if (tid < nb) sm.x.sxd[3][tid] = 0.0f;
                }
            }
            __syncthreads();
            if (valid_row) {
                const uint4* wqs_p = Wqs + ((size_t) (b0 + kp) * 2) * N + r;
                const __half* wd_p = Wd + (size_t) (b0 + kp) * N + r;
                const uint4* x_p = reinterpret_cast<const uint4*>(&sm.x.sxh[0][kp][0]);
                const float* sxd_p = &sm.x.sxd[0][kp];
                if (ne == 1) {
#pragma unroll 1
                    for (int db = kp; db < nb; db += 5, wqs_p += 10 * N, wd_p += 5 * N, x_p += 20, sxd_p += 5) {
                        const float d = __half2float(__ldg(wd_p));
                        __half2 h0 = __float2half2_rn(0.0f);
#pragma unroll
                        for (int hf = 0; hf < 2; ++hf) {
                            const uint4 qv = __ldg(wqs_p + hf * N);
                            const uint4 xa = x_p[hf * 2 + 0];
                            const uint4 xb = x_p[hf * 2 + 1];
                            const uint32_t xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                            const uint32_t ws[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
                            for (int u = 0; u < 4; ++u) {
                                const uint32_t vx = ws[u];
                                const uint32_t a = __byte_perm(vx, 0x64006400u, 0x5150);
                                const uint32_t bv = __byte_perm(vx, 0x64006400u, 0x5352);
                                h0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&a), c1152), *reinterpret_cast<const __half2*>(&xv[u * 2 + 0]), h0);
                                h0 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&bv), c1152), *reinterpret_cast<const __half2*>(&xv[u * 2 + 1]), h0);
                            }
                        }
                        const float2 f0 = __half22float2(h0);
                        acc0 = fmaf(d * sxd_p[0], f0.x + f0.y, acc0);
                    }
                } else if (ne == 2) {
#pragma unroll 1
                    for (int db = kp; db < nb; db += 5, wqs_p += 10 * N, wd_p += 5 * N, x_p += 20, sxd_p += 5) {
                        const float d = __half2float(__ldg(wd_p));
                        __half2 h0 = __float2half2_rn(0.0f), h1 = h0;
#pragma unroll
                        for (int hf = 0; hf < 2; ++hf) {
                            const uint4 qv = __ldg(wqs_p + hf * N);
                            const uint32_t ws[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
                            for (int p = 0; p < 2; ++p) {
                                const uint4 x0v = x_p[hf * 2 + p];
                                const uint4 x1v = x_p[80 + hf * 2 + p];
                                const uint32_t x0a[4] = {x0v.x, x0v.y, x0v.z, x0v.w};
                                const uint32_t x1a[4] = {x1v.x, x1v.y, x1v.z, x1v.w};
#pragma unroll
                                for (int q = 0; q < 2; ++q) {
                                    const uint32_t vx = ws[p * 2 + q];
                                    const uint32_t a = __byte_perm(vx, 0x64006400u, 0x5150);
                                    const uint32_t bv = __byte_perm(vx, 0x64006400u, 0x5352);
                                    const __half2 wa = __hsub2(*reinterpret_cast<const __half2*>(&a), c1152);
                                    const __half2 wb = __hsub2(*reinterpret_cast<const __half2*>(&bv), c1152);
                                    h0 = __hfma2(wa, *reinterpret_cast<const __half2*>(&x0a[q * 2 + 0]), h0);
                                    h0 = __hfma2(wb, *reinterpret_cast<const __half2*>(&x0a[q * 2 + 1]), h0);
                                    h1 = __hfma2(wa, *reinterpret_cast<const __half2*>(&x1a[q * 2 + 0]), h1);
                                    h1 = __hfma2(wb, *reinterpret_cast<const __half2*>(&x1a[q * 2 + 1]), h1);
                                }
                            }
                        }
                        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
                        acc0 = fmaf(d * sxd_p[0], f0.x + f0.y, acc0);
                        acc1 = fmaf(d * sxd_p[20], f1.x + f1.y, acc1);
                    }
                } else {
#pragma unroll 1
                    for (int db = kp; db < nb; db += 5, wqs_p += 10 * N, wd_p += 5 * N, x_p += 20, sxd_p += 5) {
                        const float d = __half2float(__ldg(wd_p));
                        __half2 h0 = __float2half2_rn(0.0f), h1 = h0, h2 = h0, h3 = h0;
#pragma unroll
                        for (int hf = 0; hf < 2; ++hf) {
                            const uint4 qv = __ldg(wqs_p + hf * N);
                            const uint32_t ws[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
                            for (int p = 0; p < 2; ++p) {
                                const uint4 x0v = x_p[hf * 2 + p];
                                const uint4 x1v = x_p[80 + hf * 2 + p];
                                const uint4 x2v = x_p[160 + hf * 2 + p];
                                const uint4 x3v = x_p[240 + hf * 2 + p];
                                const uint32_t x0a[4] = {x0v.x, x0v.y, x0v.z, x0v.w};
                                const uint32_t x1a[4] = {x1v.x, x1v.y, x1v.z, x1v.w};
                                const uint32_t x2a[4] = {x2v.x, x2v.y, x2v.z, x2v.w};
                                const uint32_t x3a[4] = {x3v.x, x3v.y, x3v.z, x3v.w};
#pragma unroll
                                for (int q = 0; q < 2; ++q) {
                                    const uint32_t vx = ws[p * 2 + q];
                                    const uint32_t a = __byte_perm(vx, 0x64006400u, 0x5150);
                                    const uint32_t bv = __byte_perm(vx, 0x64006400u, 0x5352);
                                    const __half2 wa = __hsub2(*reinterpret_cast<const __half2*>(&a), c1152);
                                    const __half2 wb = __hsub2(*reinterpret_cast<const __half2*>(&bv), c1152);
                                    h0 = __hfma2(wa, *reinterpret_cast<const __half2*>(&x0a[q * 2 + 0]), h0);
                                    h0 = __hfma2(wb, *reinterpret_cast<const __half2*>(&x0a[q * 2 + 1]), h0);
                                    h1 = __hfma2(wa, *reinterpret_cast<const __half2*>(&x1a[q * 2 + 0]), h1);
                                    h1 = __hfma2(wb, *reinterpret_cast<const __half2*>(&x1a[q * 2 + 1]), h1);
                                    h2 = __hfma2(wa, *reinterpret_cast<const __half2*>(&x2a[q * 2 + 0]), h2);
                                    h2 = __hfma2(wb, *reinterpret_cast<const __half2*>(&x2a[q * 2 + 1]), h2);
                                    h3 = __hfma2(wa, *reinterpret_cast<const __half2*>(&x3a[q * 2 + 0]), h3);
                                    h3 = __hfma2(wb, *reinterpret_cast<const __half2*>(&x3a[q * 2 + 1]), h3);
                                }
                            }
                        }
                        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1), f2 = __half22float2(h2), f3 = __half22float2(h3);
                        acc0 = fmaf(d * sxd_p[0], f0.x + f0.y, acc0);
                        acc1 = fmaf(d * sxd_p[20], f1.x + f1.y, acc1);
                        acc2 = fmaf(d * sxd_p[40], f2.x + f2.y, acc2);
                        acc3 = fmaf(d * sxd_p[60], f3.x + f3.y, acc3);
                    }
                }
            }
            if (!FIXED_2560_640 && b0 + 20 < K_blk) __syncthreads();
        }
        __syncthreads();
        sm.part[0][kp][r_loc] = acc0;
        if (ne > 1) sm.part[1][kp][r_loc] = acc1;
        if (ne > 2) sm.part[2][kp][r_loc] = acc2;
        if (ne > 3) sm.part[3][kp][r_loc] = acc3;
        __syncthreads();
        if (kp == 0 && valid_row) {
            out[(size_t) ent_dst[eb + 0] * N + r] =
                sm.part[0][0][r_loc] + sm.part[0][1][r_loc] + sm.part[0][2][r_loc] + sm.part[0][3][r_loc] + sm.part[0][4][r_loc];
            if (ne > 1)
                out[(size_t) ent_dst[eb + 1] * N + r] =
                    sm.part[1][0][r_loc] + sm.part[1][1][r_loc] + sm.part[1][2][r_loc] + sm.part[1][3][r_loc] + sm.part[1][4][r_loc];
            if (ne > 2)
                out[(size_t) ent_dst[eb + 2] * N + r] =
                    sm.part[2][0][r_loc] + sm.part[2][1][r_loc] + sm.part[2][2][r_loc] + sm.part[2][3][r_loc] + sm.part[2][4][r_loc];
            if (ne > 3)
                out[(size_t) ent_dst[eb + 3] * N + r] =
                    sm.part[3][0][r_loc] + sm.part[3][1][r_loc] + sm.part[3][2][r_loc] + sm.part[3][3][r_loc] + sm.part[3][4][r_loc];
        }
        if (eb + 4 < e1) __syncthreads();
    }
}

// Non-SoA GP100 (sm_60) 8-lane Q4_K HFMA2 Gate+Up with 4-token register reuse (eb += 4) for PCIe-staged RAM experts.
__global__ void __launch_bounds__(256, 2) p100_q4k_gu_hfma2_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok,
    const block_q8_1* __restrict__ xq, NativeExpertLayout L,
    float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    const int tid = threadIdx.x;
    const int row = blockIdx.x * 32 + (tid >> 3);
    const int sub = tid & 7;
    const int N = 2 * (int) L.n_ff;
    const bool valid = row < N;
    const bool is_up = row >= (int) L.n_ff;
    const int r_out = is_up ? (row - (int) L.n_ff) : row;
    const int K_blk = (int) (L.n_embd / 256);
    const int K_q8 = (int) (L.n_embd / 32);

    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const block_q4_K* wr = valid
        ? reinterpret_cast<const block_q4_K*>(blob + (is_up ? L.up_off : 0) + (size_t) r_out * L.gu_row)
        : nullptr;

    __shared__ __half2 sxh[4][1280];
    __shared__ float sxd[4][80];
    __shared__ float sxs[4][80];
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);

    for (int eb = e0; eb < e1; eb += 4) {
        const int ne = (eb + 4 <= e1) ? 4 : (e1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 10) {
            const int nb = (b0 + 10 <= K_blk) ? 10 : (K_blk - b0);
            const int nq8 = nb * 8, nh2 = nb * 128;
            for (int t = 0; t < ne; ++t) {
                const block_q8_1* xb = xq + (size_t) ent_tok[eb + t] * K_q8 + b0 * 8;
                for (int i = tid; i < nh2; i += 256) {
                    const int qb = i >> 4, j = (i & 15) << 1;
                    sxh[t][i] = iq_q81_pair_h2(xb[qb].qs + j, c1152, inv128);
                }
                if (tid < nq8) {
                    const float2 ds = __half22float2(xb[tid].ds);
                    sxd[t][tid] = ds.x * 128.0f;
                    sxs[t][tid] = ds.y;
                }
            }
            if (ne == 3) {
                for (int i = tid; i < nh2; i += 256) sxh[3][i] = __float2half2_rn(0.0f);
                if (tid < nq8) { sxd[3][tid] = 0.0f; sxs[3][tid] = 0.0f; }
            }
            __syncthreads();
            if (valid) {
                const int nch = nb * 4;
                if (ne == 1) {
                    for (int ch = sub; ch < nch; ch += 8) {
                        const int db_k = ch >> 2, c = ch & 3;
                        const block_q4_K* bq = wr + b0 + db_k;
                        const float2 dm = __half22float2(__ldg(&bq->dm));
                        const uint32_t* sc32 = reinterpret_cast<const uint32_t*>(bq->scales);
                        const uint32_t s0 = __ldg(sc32 + 0), s1 = __ldg(sc32 + 1), s2 = __ldg(sc32 + 2);
                        const uint32_t sc_w[2] = {s0 & 0x3f3f3f3fu, (s2 & 0x0f0f0f0fu) | ((s0 >> 2) & 0x30303030u)};
                        const uint32_t m_w[2]  = {s1 & 0x3f3f3f3fu, ((s2 >> 4) & 0x0f0f0f0fu) | ((s1 >> 2) & 0x30303030u)};
                        const int sb0 = c * 2, sb1 = c * 2 + 1;
                        const float sc0 = (float) ((sc_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                        const float sc1 = (float) ((sc_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                        const float mn0 = (float) ((m_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                        const float mn1 = (float) ((m_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                        const uint32_t* q32 = reinterpret_cast<const uint32_t*>(bq->qs + c * 32);
                        __half2 p0a = __float2half2_rn(0.0f), p0b = p0a;
                        const int x0_base = (db_k * 8 + sb0) * 16, x1_base = (db_k * 8 + sb1) * 16;
#pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            const uint32_t w = __ldg(q32 + u);
                            const uint32_t lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu;
                            const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                            const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                            const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                            const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                            p0a = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&la), c1024), sxh[0][x0_base + u * 2 + 0], p0a);
                            p0a = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&lb), c1024), sxh[0][x0_base + u * 2 + 1], p0a);
                            p0b = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&ha), c1024), sxh[0][x1_base + u * 2 + 0], p0b);
                            p0b = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&hb), c1024), sxh[0][x1_base + u * 2 + 1], p0b);
                        }
                        const float2 f0a = __half22float2(p0a), f0b = __half22float2(p0b);
                        const float sd0 = fmaf(sc0 * sxd[0][db_k * 8 + sb0], f0a.x + f0a.y, sc1 * sxd[0][db_k * 8 + sb1] * (f0b.x + f0b.y));
                        const float sm0 = fmaf(mn0, sxs[0][db_k * 8 + sb0], mn1 * sxs[0][db_k * 8 + sb1]);
                        acc0 = fmaf(dm.x, sd0, fmaf(-dm.y, sm0, acc0));
                    }
                } else if (ne == 2) {
                    for (int ch = sub; ch < nch; ch += 8) {
                        const int db_k = ch >> 2, c = ch & 3;
                        const block_q4_K* bq = wr + b0 + db_k;
                        const float2 dm = __half22float2(__ldg(&bq->dm));
                        const uint32_t* sc32 = reinterpret_cast<const uint32_t*>(bq->scales);
                        const uint32_t s0 = __ldg(sc32 + 0), s1 = __ldg(sc32 + 1), s2 = __ldg(sc32 + 2);
                        const uint32_t sc_w[2] = {s0 & 0x3f3f3f3fu, (s2 & 0x0f0f0f0fu) | ((s0 >> 2) & 0x30303030u)};
                        const uint32_t m_w[2]  = {s1 & 0x3f3f3f3fu, ((s2 >> 4) & 0x0f0f0f0fu) | ((s1 >> 2) & 0x30303030u)};
                        const int sb0 = c * 2, sb1 = c * 2 + 1;
                        const float sc0 = (float) ((sc_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                        const float sc1 = (float) ((sc_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                        const float mn0 = (float) ((m_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                        const float mn1 = (float) ((m_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                        const uint32_t* q32 = reinterpret_cast<const uint32_t*>(bq->qs + c * 32);
                        __half2 p0a = __float2half2_rn(0.0f), p0b = p0a, p1a = p0a, p1b = p0a;
                        const int x0_base = (db_k * 8 + sb0) * 16, x1_base = (db_k * 8 + sb1) * 16;
#pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            const uint32_t w = __ldg(q32 + u);
                            const uint32_t lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu;
                            const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                            const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                            const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                            const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                            const __half2 wla = __hsub2(*reinterpret_cast<const __half2*>(&la), c1024);
                            const __half2 wlb = __hsub2(*reinterpret_cast<const __half2*>(&lb), c1024);
                            const __half2 wha = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                            const __half2 whb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
                            p0a = __hfma2(wla, sxh[0][x0_base + u * 2 + 0], p0a); p0a = __hfma2(wlb, sxh[0][x0_base + u * 2 + 1], p0a);
                            p0b = __hfma2(wha, sxh[0][x1_base + u * 2 + 0], p0b); p0b = __hfma2(whb, sxh[0][x1_base + u * 2 + 1], p0b);
                            p1a = __hfma2(wla, sxh[1][x0_base + u * 2 + 0], p1a); p1a = __hfma2(wlb, sxh[1][x0_base + u * 2 + 1], p1a);
                            p1b = __hfma2(wha, sxh[1][x1_base + u * 2 + 0], p1b); p1b = __hfma2(whb, sxh[1][x1_base + u * 2 + 1], p1b);
                        }
                        const float2 f0a = __half22float2(p0a), f0b = __half22float2(p0b);
                        const float2 f1a = __half22float2(p1a), f1b = __half22float2(p1b);
                        acc0 = fmaf(dm.x, fmaf(sc0 * sxd[0][db_k * 8 + sb0], f0a.x + f0a.y, sc1 * sxd[0][db_k * 8 + sb1] * (f0b.x + f0b.y)),
                                    fmaf(-dm.y, fmaf(mn0, sxs[0][db_k * 8 + sb0], mn1 * sxs[0][db_k * 8 + sb1]), acc0));
                        acc1 = fmaf(dm.x, fmaf(sc0 * sxd[1][db_k * 8 + sb0], f1a.x + f1a.y, sc1 * sxd[1][db_k * 8 + sb1] * (f1b.x + f1b.y)),
                                    fmaf(-dm.y, fmaf(mn0, sxs[1][db_k * 8 + sb0], mn1 * sxs[1][db_k * 8 + sb1]), acc1));
                    }
                } else {
                    for (int ch = sub; ch < nch; ch += 8) {
                        const int db_k = ch >> 2, c = ch & 3;
                        const block_q4_K* bq = wr + b0 + db_k;
                        const float2 dm = __half22float2(__ldg(&bq->dm));
                        const uint32_t* sc32 = reinterpret_cast<const uint32_t*>(bq->scales);
                        const uint32_t s0 = __ldg(sc32 + 0), s1 = __ldg(sc32 + 1), s2 = __ldg(sc32 + 2);
                        const uint32_t sc_w[2] = {s0 & 0x3f3f3f3fu, (s2 & 0x0f0f0f0fu) | ((s0 >> 2) & 0x30303030u)};
                        const uint32_t m_w[2]  = {s1 & 0x3f3f3f3fu, ((s2 >> 4) & 0x0f0f0f0fu) | ((s1 >> 2) & 0x30303030u)};
                        const int sb0 = c * 2, sb1 = c * 2 + 1;
                        const float sc0 = (float) ((sc_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                        const float sc1 = (float) ((sc_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                        const float mn0 = (float) ((m_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                        const float mn1 = (float) ((m_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                        const uint32_t* q32 = reinterpret_cast<const uint32_t*>(bq->qs + c * 32);
                        __half2 pa0 = __float2half2_rn(0.0f), pa1 = pa0, pa2 = pa0, pa3 = pa0;
                        __half2 pb0 = pa0, pb1 = pa0, pb2 = pa0, pb3 = pa0;
                        const int x0_base = (db_k * 8 + sb0) * 16, x1_base = (db_k * 8 + sb1) * 16;
#pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            const uint32_t w = __ldg(q32 + u);
                            const uint32_t lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu;
                            const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                            const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                            const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                            const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                            const __half2 wla = __hsub2(*reinterpret_cast<const __half2*>(&la), c1024);
                            const __half2 wlb = __hsub2(*reinterpret_cast<const __half2*>(&lb), c1024);
                            const __half2 wha = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                            const __half2 whb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
                            pa0 = __hfma2(wla, sxh[0][x0_base + u * 2 + 0], pa0); pa0 = __hfma2(wlb, sxh[0][x0_base + u * 2 + 1], pa0);
                            pb0 = __hfma2(wha, sxh[0][x1_base + u * 2 + 0], pb0); pb0 = __hfma2(whb, sxh[0][x1_base + u * 2 + 1], pb0);
                            pa1 = __hfma2(wla, sxh[1][x0_base + u * 2 + 0], pa1); pa1 = __hfma2(wlb, sxh[1][x0_base + u * 2 + 1], pa1);
                            pb1 = __hfma2(wha, sxh[1][x1_base + u * 2 + 0], pb1); pb1 = __hfma2(whb, sxh[1][x1_base + u * 2 + 1], pb1);
                            pa2 = __hfma2(wla, sxh[2][x0_base + u * 2 + 0], pa2); pa2 = __hfma2(wlb, sxh[2][x0_base + u * 2 + 1], pa2);
                            pb2 = __hfma2(wha, sxh[2][x1_base + u * 2 + 0], pb2); pb2 = __hfma2(whb, sxh[2][x1_base + u * 2 + 1], pb2);
                            pa3 = __hfma2(wla, sxh[3][x0_base + u * 2 + 0], pa3); pa3 = __hfma2(wlb, sxh[3][x0_base + u * 2 + 1], pa3);
                            pb3 = __hfma2(wha, sxh[3][x1_base + u * 2 + 0], pb3); pb3 = __hfma2(whb, sxh[3][x1_base + u * 2 + 1], pb3);
                        }
                        const float2 fa0 = __half22float2(pa0), fb0 = __half22float2(pb0);
                        acc0 = fmaf(dm.x, fmaf(sc0 * sxd[0][db_k * 8 + sb0], fa0.x + fa0.y, sc1 * sxd[0][db_k * 8 + sb1] * (fb0.x + fb0.y)),
                                    fmaf(-dm.y, fmaf(mn0, sxs[0][db_k * 8 + sb0], mn1 * sxs[0][db_k * 8 + sb1]), acc0));
                        const float2 fa1 = __half22float2(pa1), fb1 = __half22float2(pb1);
                        acc1 = fmaf(dm.x, fmaf(sc0 * sxd[1][db_k * 8 + sb0], fa1.x + fa1.y, sc1 * sxd[1][db_k * 8 + sb1] * (fb1.x + fb1.y)),
                                    fmaf(-dm.y, fmaf(mn0, sxs[1][db_k * 8 + sb0], mn1 * sxs[1][db_k * 8 + sb1]), acc1));
                        const float2 fa2 = __half22float2(pa2), fb2 = __half22float2(pb2);
                        acc2 = fmaf(dm.x, fmaf(sc0 * sxd[2][db_k * 8 + sb0], fa2.x + fa2.y, sc1 * sxd[2][db_k * 8 + sb1] * (fb2.x + fb2.y)),
                                    fmaf(-dm.y, fmaf(mn0, sxs[2][db_k * 8 + sb0], mn1 * sxs[2][db_k * 8 + sb1]), acc2));
                        const float2 fa3 = __half22float2(pa3), fb3 = __half22float2(pb3);
                        acc3 = fmaf(dm.x, fmaf(sc0 * sxd[3][db_k * 8 + sb0], fa3.x + fa3.y, sc1 * sxd[3][db_k * 8 + sb1] * (fb3.x + fb3.y)),
                                    fmaf(-dm.y, fmaf(mn0, sxs[3][db_k * 8 + sb0], mn1 * sxs[3][db_k * 8 + sb1]), acc3));
                    }
                }
            }
            if (b0 + 10 < K_blk) __syncthreads();
        }
        if (valid) {
            float* dst_p = is_up ? up : gate;
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1); acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2); acc0 += __shfl_xor_sync(0xffffffffu, acc0, 4);
            if (sub == 0) dst_p[(size_t) (eb + 0) * L.n_ff + r_out] = acc0;
            if (ne > 1) {
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1); acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2); acc1 += __shfl_xor_sync(0xffffffffu, acc1, 4);
                if (sub == 0) dst_p[(size_t) (eb + 1) * L.n_ff + r_out] = acc1;
            }
            if (ne > 2) {
                acc2 += __shfl_xor_sync(0xffffffffu, acc2, 1); acc2 += __shfl_xor_sync(0xffffffffu, acc2, 2); acc2 += __shfl_xor_sync(0xffffffffu, acc2, 4);
                if (sub == 0) dst_p[(size_t) (eb + 2) * L.n_ff + r_out] = acc2;
            }
            if (ne > 3) {
                acc3 += __shfl_xor_sync(0xffffffffu, acc3, 1); acc3 += __shfl_xor_sync(0xffffffffu, acc3, 2); acc3 += __shfl_xor_sync(0xffffffffu, acc3, 4);
                if (sub == 0) dst_p[(size_t) (eb + 3) * L.n_ff + r_out] = acc3;
            }
        }
        if (eb + 4 < e1) __syncthreads();
    }
}

// GP100 (sm_60) 8-lane Q5_K HFMA2 Gate+Up with 4-token register reuse (eb += 4) for blk.2 (replaces dp4a emulation!).
__global__ void __launch_bounds__(256, 2) p100_q5k_gu_hfma2_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok,
    const block_q8_1* __restrict__ xq, NativeExpertLayout L,
    float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    const int tid = threadIdx.x;
    const int row = blockIdx.x * 32 + (tid >> 3);
    const int sub = tid & 7;
    const int N = 2 * (int) L.n_ff;
    const bool valid = row < N;
    const bool is_up = row >= (int) L.n_ff;
    const int r_out = is_up ? (row - (int) L.n_ff) : row;
    const int K_blk = (int) (L.n_embd / 256);
    const int K_q8 = (int) (L.n_embd / 32);

    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const block_q5_K* wr = valid
        ? reinterpret_cast<const block_q5_K*>(blob + (is_up ? L.up_off : 0) + (size_t) r_out * L.gu_row)
        : nullptr;

    __shared__ __half2 sxh[4][1280];
    __shared__ float sxd[4][80];
    __shared__ float sxs[4][80];
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);

    for (int eb = e0; eb < e1; eb += 4) {
        const int ne = (eb + 4 <= e1) ? 4 : (e1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 10) {
            const int nb = (b0 + 10 <= K_blk) ? 10 : (K_blk - b0);
            const int nq8 = nb * 8, nh2 = nb * 128;
            for (int t = 0; t < ne; ++t) {
                const block_q8_1* xb = xq + (size_t) ent_tok[eb + t] * K_q8 + b0 * 8;
                for (int i = tid; i < nh2; i += 256) {
                    const int qb = i >> 4, j = (i & 15) << 1;
                    sxh[t][i] = iq_q81_pair_h2(xb[qb].qs + j, c1152, inv128);
                }
                if (tid < nq8) {
                    const float2 ds = __half22float2(xb[tid].ds);
                    sxd[t][tid] = ds.x * 128.0f;
                    sxs[t][tid] = ds.y;
                }
            }
            if (ne == 3) {
                for (int i = tid; i < nh2; i += 256) sxh[3][i] = __float2half2_rn(0.0f);
                if (tid < nq8) { sxd[3][tid] = 0.0f; sxs[3][tid] = 0.0f; }
            }
            __syncthreads();
            if (valid) {
                const int nch = nb * 4;
                for (int ch = sub; ch < nch; ch += 8) {
                    const int db_k = ch >> 2, c = ch & 3;
                    const block_q5_K* bq = wr + b0 + db_k;
                    const float2 dm = __half22float2(__ldg(&bq->dm));
                    const uint32_t* sc32 = reinterpret_cast<const uint32_t*>(bq->scales);
                    const uint32_t s0 = __ldg(sc32 + 0), s1 = __ldg(sc32 + 1), s2 = __ldg(sc32 + 2);
                    const uint32_t sc_w[2] = {s0 & 0x3f3f3f3fu, (s2 & 0x0f0f0f0fu) | ((s0 >> 2) & 0x30303030u)};
                    const uint32_t m_w[2]  = {s1 & 0x3f3f3f3fu, ((s2 >> 4) & 0x0f0f0f0fu) | ((s1 >> 2) & 0x30303030u)};
                    const int sb0 = c * 2, sb1 = c * 2 + 1;
                    const float sc0 = (float) ((sc_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                    const float sc1 = (float) ((sc_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                    const float mn0 = (float) ((m_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                    const float mn1 = (float) ((m_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                    const uint32_t* q32 = reinterpret_cast<const uint32_t*>(bq->qs + c * 32);
                    const uint32_t* qh32 = reinterpret_cast<const uint32_t*>(bq->qh);
                    const int sh0 = 2 * c, sh1 = 2 * c + 1;
                    const int x0_base = (db_k * 8 + sb0) * 16, x1_base = (db_k * 8 + sb1) * 16;
                    __half2 pa[4], pb[4];
#pragma unroll
                    for (int t = 0; t < 4; ++t) { pa[t] = __float2half2_rn(0.0f); pb[t] = __float2half2_rn(0.0f); }
#pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        const uint32_t w = __ldg(q32 + u);
                        const uint32_t qh = __ldg(qh32 + u);
                        const uint32_t lo = (w & 0x0f0f0f0fu) | (((qh >> sh0) & 0x01010101u) << 4);
                        const uint32_t hi = ((w >> 4) & 0x0f0f0f0fu) | (((qh >> sh1) & 0x01010101u) << 4);
                        const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                        const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                        const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                        const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                        const __half2 wla = __hsub2(*reinterpret_cast<const __half2*>(&la), c1024);
                        const __half2 wlb = __hsub2(*reinterpret_cast<const __half2*>(&lb), c1024);
                        const __half2 wha = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                        const __half2 whb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
                        pa[0] = __hfma2(wla, sxh[0][x0_base + u * 2 + 0], pa[0]); pa[0] = __hfma2(wlb, sxh[0][x0_base + u * 2 + 1], pa[0]);
                        pb[0] = __hfma2(wha, sxh[0][x1_base + u * 2 + 0], pb[0]); pb[0] = __hfma2(whb, sxh[0][x1_base + u * 2 + 1], pb[0]);
                        if (ne > 1) {
                            pa[1] = __hfma2(wla, sxh[1][x0_base + u * 2 + 0], pa[1]); pa[1] = __hfma2(wlb, sxh[1][x0_base + u * 2 + 1], pa[1]);
                            pb[1] = __hfma2(wha, sxh[1][x1_base + u * 2 + 0], pb[1]); pb[1] = __hfma2(whb, sxh[1][x1_base + u * 2 + 1], pb[1]);
                        }
                        if (ne > 2) {
                            pa[2] = __hfma2(wla, sxh[2][x0_base + u * 2 + 0], pa[2]); pa[2] = __hfma2(wlb, sxh[2][x0_base + u * 2 + 1], pa[2]);
                            pb[2] = __hfma2(wha, sxh[2][x1_base + u * 2 + 0], pb[2]); pb[2] = __hfma2(whb, sxh[2][x1_base + u * 2 + 1], pb[2]);
                            pa[3] = __hfma2(wla, sxh[3][x0_base + u * 2 + 0], pa[3]); pa[3] = __hfma2(wlb, sxh[3][x0_base + u * 2 + 1], pa[3]);
                            pb[3] = __hfma2(wha, sxh[3][x1_base + u * 2 + 0], pb[3]); pb[3] = __hfma2(whb, sxh[3][x1_base + u * 2 + 1], pb[3]);
                        }
                    }
                    {
                        const float2 fa0 = __half22float2(pa[0]), fb0 = __half22float2(pb[0]);
                        acc0 = fmaf(dm.x, fmaf(sc0 * sxd[0][db_k * 8 + sb0], fa0.x + fa0.y, sc1 * sxd[0][db_k * 8 + sb1] * (fb0.x + fb0.y)),
                                    fmaf(-dm.y, fmaf(mn0, sxs[0][db_k * 8 + sb0], mn1 * sxs[0][db_k * 8 + sb1]), acc0));
                        if (ne > 1) {
                            const float2 fa1 = __half22float2(pa[1]), fb1 = __half22float2(pb[1]);
                            acc1 = fmaf(dm.x, fmaf(sc0 * sxd[1][db_k * 8 + sb0], fa1.x + fa1.y, sc1 * sxd[1][db_k * 8 + sb1] * (fb1.x + fb1.y)),
                                        fmaf(-dm.y, fmaf(mn0, sxs[1][db_k * 8 + sb0], mn1 * sxs[1][db_k * 8 + sb1]), acc1));
                        }
                        if (ne > 2) {
                            const float2 fa2 = __half22float2(pa[2]), fb2 = __half22float2(pb[2]);
                            acc2 = fmaf(dm.x, fmaf(sc0 * sxd[2][db_k * 8 + sb0], fa2.x + fa2.y, sc1 * sxd[2][db_k * 8 + sb1] * (fb2.x + fb2.y)),
                                        fmaf(-dm.y, fmaf(mn0, sxs[2][db_k * 8 + sb0], mn1 * sxs[2][db_k * 8 + sb1]), acc2));
                            const float2 fa3 = __half22float2(pa[3]), fb3 = __half22float2(pb[3]);
                            acc3 = fmaf(dm.x, fmaf(sc0 * sxd[3][db_k * 8 + sb0], fa3.x + fa3.y, sc1 * sxd[3][db_k * 8 + sb1] * (fb3.x + fb3.y)),
                                        fmaf(-dm.y, fmaf(mn0, sxs[3][db_k * 8 + sb0], mn1 * sxs[3][db_k * 8 + sb1]), acc3));
                        }
                    }
                }
            }
            if (b0 + 10 < K_blk) __syncthreads();
        }
        if (valid) {
            float* dst_p = is_up ? up : gate;
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1); acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2); acc0 += __shfl_xor_sync(0xffffffffu, acc0, 4);
            if (sub == 0) dst_p[(size_t) (eb + 0) * L.n_ff + r_out] = acc0;
            if (ne > 1) {
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1); acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2); acc1 += __shfl_xor_sync(0xffffffffu, acc1, 4);
                if (sub == 0) dst_p[(size_t) (eb + 1) * L.n_ff + r_out] = acc1;
            }
            if (ne > 2) {
                acc2 += __shfl_xor_sync(0xffffffffu, acc2, 1); acc2 += __shfl_xor_sync(0xffffffffu, acc2, 2); acc2 += __shfl_xor_sync(0xffffffffu, acc2, 4);
                if (sub == 0) dst_p[(size_t) (eb + 2) * L.n_ff + r_out] = acc2;
            }
            if (ne > 3) {
                acc3 += __shfl_xor_sync(0xffffffffu, acc3, 1); acc3 += __shfl_xor_sync(0xffffffffu, acc3, 2); acc3 += __shfl_xor_sync(0xffffffffu, acc3, 4);
                if (sub == 0) dst_p[(size_t) (eb + 3) * L.n_ff + r_out] = acc3;
            }
        }
        if (eb + 4 < e1) __syncthreads();
    }
}

// Non-SoA GP100 (sm_60) 4-lane Q5_1 SWAR HFMA2 Down with 4-token register reuse (eb += 4) for PCIe-staged RAM experts.
__global__ void __launch_bounds__(256, 2) p100_q51_down_hfma2_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst,
    const block_q8_1* __restrict__ hq, NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    __shared__ __half2 sxh[4][20][16];
    __shared__ float sxd[4][20];
    __shared__ float sxs[4][20];
    const int tid = threadIdx.x;
    const int r = blockIdx.x * 64 + (tid >> 2);
    const int sub = tid & 3;
    const bool valid_r = r < (int) L.n_embd;
    const int K_blk = (int) (L.n_ff / 32);
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const block_q5_1* wr = valid_r ? (const block_q5_1*) (blob + L.down_off + (size_t) r * L.d_row) : nullptr;
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);

    for (int eb = e0; eb < e1; eb += 4) {
        const int ne = (eb + 4 <= e1) ? 4 : (e1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = (b0 + 20 <= K_blk) ? 20 : (K_blk - b0);
            for (int t = 0; t < ne; ++t) {
                const block_q8_1* xt = hq + (size_t) (eb + t) * K_blk + b0;
                for (int i = tid; i < nb * 16; i += 256) {
                    const int db = i >> 4, j = (i & 15) << 1;
                    sxh[t][db][i & 15] = iq_q81_pair_h2(xt[db].qs + j, c1152, inv128);
                }
                if (tid < nb) {
                    const float2 ds = __half22float2(xt[tid].ds);
                    sxd[t][tid] = ds.x * 128.0f;
                    sxs[t][tid] = ds.y;
                }
            }
            if (ne == 3) {
                for (int i = tid; i < nb * 16; i += 256) sxh[3][i >> 4][i & 15] = __float2half2_rn(0.0f);
                if (tid < nb) { sxd[3][tid] = 0.0f; sxs[3][tid] = 0.0f; }
            }
            __syncthreads();
            if (valid_r) {
                for (int db = sub; db < nb; db += 4) {
                    const block_q5_1* bq = wr + b0 + db;
                    const float2 dm = __half22float2(__ldg(&bq->dm));
                    const uint32_t qh = (uint32_t) __ldg(reinterpret_cast<const uint32_t*>(bq->qh));
                    const uint32_t* q32 = reinterpret_cast<const uint32_t*>(bq->qs);
                    __half2 h0 = __float2half2_rn(0.0f), h1 = h0, h2 = h0, h3 = h0;
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const uint32_t w = __ldg(q32 + u);
                        const uint32_t hb_lo = (((qh >> (u * 4)) & 0xfu) * 0x00204081u) & 0x01010101u;
                        const uint32_t hb_hi = (((qh >> (16 + u * 4)) & 0xfu) * 0x00204081u) & 0x01010101u;
                        const uint32_t lo = (w & 0x0f0f0f0fu) | (hb_lo << 4);
                        const uint32_t hi = ((w >> 4) & 0x0f0f0f0fu) | (hb_hi << 4);
                        const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                        const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                        const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                        const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                        const __half2 wla = __hsub2(*reinterpret_cast<const __half2*>(&la), c1024);
                        const __half2 wlb = __hsub2(*reinterpret_cast<const __half2*>(&lb), c1024);
                        const __half2 wha = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                        const __half2 whb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
                        h0 = __hfma2(wla, sxh[0][db][u * 2 + 0], h0); h0 = __hfma2(wlb, sxh[0][db][u * 2 + 1], h0);
                        h0 = __hfma2(wha, sxh[0][db][8 + u * 2 + 0], h0); h0 = __hfma2(whb, sxh[0][db][8 + u * 2 + 1], h0);
                        if (ne > 1) {
                            h1 = __hfma2(wla, sxh[1][db][u * 2 + 0], h1); h1 = __hfma2(wlb, sxh[1][db][u * 2 + 1], h1);
                            h1 = __hfma2(wha, sxh[1][db][8 + u * 2 + 0], h1); h1 = __hfma2(whb, sxh[1][db][8 + u * 2 + 1], h1);
                        }
                        if (ne > 2) {
                            h2 = __hfma2(wla, sxh[2][db][u * 2 + 0], h2); h2 = __hfma2(wlb, sxh[2][db][u * 2 + 1], h2);
                            h2 = __hfma2(wha, sxh[2][db][8 + u * 2 + 0], h2); h2 = __hfma2(whb, sxh[2][db][8 + u * 2 + 1], h2);
                            h3 = __hfma2(wla, sxh[3][db][u * 2 + 0], h3); h3 = __hfma2(wlb, sxh[3][db][u * 2 + 1], h3);
                            h3 = __hfma2(wha, sxh[3][db][8 + u * 2 + 0], h3); h3 = __hfma2(whb, sxh[3][db][8 + u * 2 + 1], h3);
                        }
                    }
                    const float2 f0 = __half22float2(h0);
                    acc0 = fmaf(dm.y, sxs[0][db], fmaf(dm.x * sxd[0][db], f0.x + f0.y, acc0));
                    if (ne > 1) {
                        const float2 f1 = __half22float2(h1);
                        acc1 = fmaf(dm.y, sxs[1][db], fmaf(dm.x * sxd[1][db], f1.x + f1.y, acc1));
                    }
                    if (ne > 2) {
                        const float2 f2 = __half22float2(h2), f3 = __half22float2(h3);
                        acc2 = fmaf(dm.y, sxs[2][db], fmaf(dm.x * sxd[2][db], f2.x + f2.y, acc2));
                        acc3 = fmaf(dm.y, sxs[3][db], fmaf(dm.x * sxd[3][db], f3.x + f3.y, acc3));
                    }
                }
            }
            if (b0 + 20 < K_blk) __syncthreads();
        }
        if (valid_r) {
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1); acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2);
            if (sub == 0) out[(size_t) ent_dst[eb + 0] * L.n_embd + r] = acc0;
            if (ne > 1) {
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1); acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2);
                if (sub == 0) out[(size_t) ent_dst[eb + 1] * L.n_embd + r] = acc1;
            }
            if (ne > 2) {
                acc2 += __shfl_xor_sync(0xffffffffu, acc2, 1); acc2 += __shfl_xor_sync(0xffffffffu, acc2, 2);
                if (sub == 0) out[(size_t) ent_dst[eb + 2] * L.n_embd + r] = acc2;
            }
            if (ne > 3) {
                acc3 += __shfl_xor_sync(0xffffffffu, acc3, 1); acc3 += __shfl_xor_sync(0xffffffffu, acc3, 2);
                if (sub == 0) out[(size_t) ent_dst[eb + 3] * L.n_embd + r] = acc3;
            }
        }
        if (eb + 4 < e1) __syncthreads();
    }
}

// Non-SoA GP100 (sm_60) 4-lane Q8_0 HFMA2 Down with 4-token register reuse (eb += 4) for PCIe-staged RAM experts.
__global__ void __launch_bounds__(256, 2) p100_q80_down_hfma2_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst,
    const block_q8_1* __restrict__ hq, NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e0 >= e1) return;

    __shared__ __half2 sxh[4][20][16];
    __shared__ float sxd[4][20];
    const int tid = threadIdx.x;
    const int r = blockIdx.x * 64 + (tid >> 2);
    const int sub = tid & 3;
    const bool valid_r = r < (int) L.n_embd;
    const int K_blk = (int) (L.n_ff / 32);
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const block_q8_0* wr = valid_r ? (const block_q8_0*) (blob + L.down_off + (size_t) r * L.d_row) : nullptr;
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 inv128 = __floats2half2_rn(1.0f / 128.0f, 1.0f / 128.0f);

    for (int eb = e0; eb < e1; eb += 4) {
        const int ne = (eb + 4 <= e1) ? 4 : (e1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = (b0 + 20 <= K_blk) ? 20 : (K_blk - b0);
            for (int t = 0; t < ne; ++t) {
                const block_q8_1* xt = hq + (size_t) (eb + t) * K_blk + b0;
                for (int i = tid; i < nb * 16; i += 256) {
                    const int db = i >> 4, j = (i & 15) << 1;
                    sxh[t][db][i & 15] = iq_q81_pair_h2(xt[db].qs + j, c1152, inv128);
                }
                if (tid < nb) sxd[t][tid] = __low2float(xt[tid].ds) * 128.0f;
            }
            if (ne == 3) {
                for (int i = tid; i < nb * 16; i += 256) sxh[3][i >> 4][i & 15] = __float2half2_rn(0.0f);
                if (tid < nb) sxd[3][tid] = 0.0f;
            }
            __syncthreads();
            if (valid_r) {
                for (int db = sub; db < nb; db += 4) {
                    const block_q8_0* bq = wr + b0 + db;
                    const float d = __half2float(__ldg(&bq->d));
                    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(bq->qs);
                    __half2 h0 = __float2half2_rn(0.0f), h1 = h0, h2 = h0, h3 = h0;
#pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        const uint32_t lo16 = __ldg(q16 + u * 2 + 0);
                        const uint32_t hi16 = __ldg(q16 + u * 2 + 1);
                        const uint32_t vx = (lo16 | (hi16 << 16)) ^ 0x80808080u;
                        const uint32_t a = __byte_perm(vx, 0x64006400u, 0x5150);
                        const uint32_t bv = __byte_perm(vx, 0x64006400u, 0x5352);
                        const __half2 wa = __hsub2(*reinterpret_cast<const __half2*>(&a), c1152);
                        const __half2 wb = __hsub2(*reinterpret_cast<const __half2*>(&bv), c1152);
                        h0 = __hfma2(wa, sxh[0][db][u * 2 + 0], h0); h0 = __hfma2(wb, sxh[0][db][u * 2 + 1], h0);
                        if (ne > 1) {
                            h1 = __hfma2(wa, sxh[1][db][u * 2 + 0], h1); h1 = __hfma2(wb, sxh[1][db][u * 2 + 1], h1);
                        }
                        if (ne > 2) {
                            h2 = __hfma2(wa, sxh[2][db][u * 2 + 0], h2); h2 = __hfma2(wb, sxh[2][db][u * 2 + 1], h2);
                            h3 = __hfma2(wa, sxh[3][db][u * 2 + 0], h3); h3 = __hfma2(wb, sxh[3][db][u * 2 + 1], h3);
                        }
                    }
                    const float2 f0 = __half22float2(h0);
                    acc0 = fmaf(d * sxd[0][db], f0.x + f0.y, acc0);
                    if (ne > 1) {
                        const float2 f1 = __half22float2(h1);
                        acc1 = fmaf(d * sxd[1][db], f1.x + f1.y, acc1);
                    }
                    if (ne > 2) {
                        const float2 f2 = __half22float2(h2), f3 = __half22float2(h3);
                        acc2 = fmaf(d * sxd[2][db], f2.x + f2.y, acc2);
                        acc3 = fmaf(d * sxd[3][db], f3.x + f3.y, acc3);
                    }
                }
            }
            if (b0 + 20 < K_blk) __syncthreads();
        }
        if (valid_r) {
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1); acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2);
            if (sub == 0) out[(size_t) ent_dst[eb + 0] * L.n_embd + r] = acc0;
            if (ne > 1) {
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1); acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2);
                if (sub == 0) out[(size_t) ent_dst[eb + 1] * L.n_embd + r] = acc1;
            }
            if (ne > 2) {
                acc2 += __shfl_xor_sync(0xffffffffu, acc2, 1); acc2 += __shfl_xor_sync(0xffffffffu, acc2, 2);
                if (sub == 0) out[(size_t) ent_dst[eb + 2] * L.n_embd + r] = acc2;
            }
            if (ne > 3) {
                acc3 += __shfl_xor_sync(0xffffffffu, acc3, 1); acc3 += __shfl_xor_sync(0xffffffffu, acc3, 2);
                if (sub == 0) out[(size_t) ent_dst[eb + 3] * L.n_embd + r] = acc3;
            }
        }
        if (eb + 4 < e1) __syncthreads();
    }
}

bool iq_p100_hfma2_on() {
    static int cache[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    if (cache[dev] == 0) {
        int maj = 0, min = 0;
        cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev);
        const char* v = std::getenv("MAYA_P100_HFMA2");
        const bool on = (maj == 6 && min == 0) && (v == nullptr || std::atoi(v) != 0);
        cache[dev] = on ? 1 : -1;
    }
    return cache[dev] > 0;
}

// A mixed layer's half (gate: UP = false, up: UP = true) by its own type TG: the same rows, dot and stores as
// native_gu_kernel's for that half.  native_expert_grouped launches it twice where the gate and up types differ.
template<int TG, bool UP>
__global__ void __launch_bounds__(256) native_gu_half_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                             const int32_t* __restrict__ grp_start,
                                                             const int32_t* __restrict__ n_groups,
                                                             const int32_t* __restrict__ ent_tok,
                                                             const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                             float* __restrict__ dst) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * GU_ROWS + warp;              // 0 .. n_ff
    if (r >= L.n_ff) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = UP ? blob + L.up_off + (size_t) r * L.u_row : blob + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s = row_dot<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, lane);
        if (lane == 0) dst[(size_t) e * L.n_ff + r] = s;
    }
}

template<bool UP>
bool launch_gu_half(int ty, dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                    const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X, const NativeExpertLayout& L,
                    float* dst) {
#define MAYA_GU_HALF(T) case T: native_gu_half_kernel<T, UP><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, \
                                                                                  ent_tok, X, L, dst); return true;
    switch (ty) {
        MAYA_GU_HALF(10) MAYA_GU_HALF(11) MAYA_GU_HALF(12) MAYA_GU_HALF(13) MAYA_GU_HALF(14)
        MAYA_GU_HALF(16) MAYA_GU_HALF(17) MAYA_GU_HALF(18) MAYA_GU_HALF(19) MAYA_GU_HALF(20)
        MAYA_GU_HALF(21) MAYA_GU_HALF(22) MAYA_GU_HALF(23) MAYA_GU_HALF(29)
        default: return false;
    }
#undef MAYA_GU_HALF
}

__global__ void swiglu_entries_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                                      float limit, long long n_cap, const int32_t* __restrict__ n_groups,
                                      const int32_t* __restrict__ grp_start, int n_ff) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    // Live end entry = grp_start[*n_groups] (VRAM and PCIe layouts). ng==0 → no work (common at 100% hit).
    long long n = n_cap;
    if (n_groups != nullptr && grp_start != nullptr && n_ff > 0) {
        const int ng = *n_groups;
        if (ng <= 0) return;
        n = (long long) grp_start[ng] * (long long) n_ff;
    }
    if (i >= n || i >= n_cap) return;
    // limit = +inf is the plain unclamped path (qwen4exp, __expf); a finite limit is GLM's clamped
    // swiglu, which needs the precise expf - glm_ffn's parity tolerance is tighter than the
    // intrinsic's argument-scaled error at the |g| ~ 30 the model's gates reach
    if (limit == INFINITY) {
        const float g = gate[i];
        h[i] = (g / (1.0f + __expf(-g))) * up[i];
        return;
    }
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    h[i] = (g / (1.0f + expf(-g))) * u;
}

template<int TD>
__global__ void __launch_bounds__(256) native_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent_dst,
                                                          const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                          float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    if (r >= L.n_embd) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s = row_dot<TD>(wr, hq + (size_t) e * hb, nb, lane);
        if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s;
    }
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
__global__ void quantize_q8_1_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xi = x[i];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = fminf(amax / 127.0f, 65504.0f);   // finite in FP16 (upstream #1448); below that nothing changes
    const int8_t q = amax == 0.0f ? 0 : fmaxf(-127.0f, fminf(127.0f, roundf(xi / d)));
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}

// Fused SwiGLU + Q8_1 quantization bounded by grp_start[*n_groups] (eliminates intermediate h write/read + 1 launch/layer).
__global__ void swiglu_quant_q8_1_bounded_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                                 block_q8_1* __restrict__ y, float limit, long long n_cap,
                                                 const int32_t* __restrict__ n_groups,
                                                 const int32_t* __restrict__ grp_start, int n_ff) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long n = n_cap;
    if (n_groups != nullptr && grp_start != nullptr && n_ff > 0) {
        const int ng = *n_groups;
        if (ng <= 0) return;
        n = (long long) grp_start[ng] * (long long) n_ff;
    }
    if (i >= n || i >= n_cap) return;
    float xi;
    if (limit == INFINITY) {
        const float g = gate[i];
        xi = (g / (1.0f + __expf(-g))) * up[i];
    } else {
        const float g = fminf(gate[i], limit);
        const float u = fminf(fmaxf(up[i], -limit), limit);
        xi = (g / (1.0f + expf(-g))) * u;
    }
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(xi / d)));
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}

// Grouped-expert path: same q8_1 quant, but idle when the live plan has 0 groups or i is past grp_start[ng].
__global__ void quantize_q8_1_bounded_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, long long n_cap,
                                              const int32_t* __restrict__ n_groups, const int32_t* __restrict__ grp_start,
                                              int n_ff) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long n = n_cap;
    if (n_groups != nullptr && grp_start != nullptr && n_ff > 0) {
        const int ng = *n_groups;
        if (ng <= 0) return;
        n = (long long) grp_start[ng] * (long long) n_ff;
    }
    if (i >= n || i >= n_cap) return;
    const float xi = x[i];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : fmaxf(-127.0f, fminf(127.0f, roundf(xi / d)));
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template<typename dst_t> __device__ __forceinline__ dst_t cvt(float v);
template<> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template<> __device__ __forceinline__ __half cvt<__half>(float v) { return __float2half(v); }

template<typename dst_t>
__device__ void dq_iq2_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_s(const void* vx, int64_t ibs, dst_t* yy, int tid, const uint64_t* codebook = iq2s_grid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint64_t grid = codebook[x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)];
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j)
        y[j] = cvt<dst_t>(d * (uint8_t) (grid >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq3_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid, const uint32_t* codebook = iq3xxs_grid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint32_t grid1 = codebook[q3[2 * il + 0]];
    const uint32_t grid2 = codebook[q3[2 * il + 1]];
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq3_s(const void* vx, int64_t ibs, dst_t* yy, int tid, const uint32_t* codebook = iq3s_grid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint32_t grid1 = codebook[qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)];
    const uint32_t grid2 = codebook[qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)];
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
// One 128-value half of an IQ3_S superblock: `h` (0 or 1) selects the sub-blocks ib = 4*h .. 4*h + 3, whose
// values land at the chunk-local positions 0 .. 127 (a WK-128 chunk's own column space, not the superblock's
// 256).  `sub` in 0..15 picks the (il, ib&3) pair that dq_iq3_s reads from a lane's tid, so a warp's lanes
// 0..15 and 16..31 decode two rows at once and a lane still decodes the eight values per row that dq_iq3_s
// gives each of its 32.  d, qs, qh, scales and signs stay indexed by the superblock's sub-block ib, and the
// float d and the sign application are dq_iq3_s's own expressions.
template<typename dst_t>
__device__ void dq_iq3_s_half(const void* vx, int64_t ibs, int h, int sub, dst_t* yy, const uint32_t* codebook = iq3s_grid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int il = sub >> 2, ib4 = sub & 3, ib = 4 * h + ib4;
    dst_t* y = yy + 32 * ib4 + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint32_t grid1 = codebook[qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)];
    const uint32_t grid2 = codebook[qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)];
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    if constexpr (std::is_same_v<dst_t, __half>) {
        // the eight values are contiguous and eight-aligned: four aligned half2 stores, not eight
#pragma unroll
        for (int j = 0; j < 4; j += 2) {
            const float l0 = d * (float) (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            const float l1 = d * (float) (uint8_t) (grid1 >> (8 * j + 8)) * (signs & kmask_iq2xs[j + 1] ? -1.f : 1.f);
            const float h0 = d * (float) (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            const float h1 = d * (float) (uint8_t) (grid2 >> (8 * j + 8)) * (signs & kmask_iq2xs[j + 5] ? -1.f : 1.f);
            *reinterpret_cast<__half2*>(y + j) = __floats2half2_rn(l0, l1);
            *reinterpret_cast<__half2*>(y + j + 4) = __floats2half2_rn(h0, h1);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y[j + 0] = cvt<dst_t>(d * (float) (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
            y[j + 4] = cvt<dst_t>(d * (float) (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
        }
    }
}
// The same 128-value half for IQ3_XXS: `h` and `sub` give the sub-block ib = 4*h + (sub & 3) and the value group
// il = sub >> 2 exactly as dq_iq3_s_half, and the codebook, the four-bit gas scale (aux32 bits 28..31) and the
// seven-bit sign group (bits 7*il .. 7*il + 6) are dq_iq3_xxs's own expressions.  Its grid1 holds y[0..3] and
// grid2 y[4..7], so the half path splits the same way dq_iq3_s_half's does.
template<typename dst_t>
__device__ void dq_iq3_xxs_half(const void* vx, int64_t ibs, int h, int sub, dst_t* yy, const uint32_t* codebook = iq3xxs_grid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int il = sub >> 2, ib4 = sub & 3, ib = 4 * h + ib4;
    dst_t* y = yy + 32 * ib4 + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint32_t grid1 = codebook[q3[2 * il + 0]];
    const uint32_t grid2 = codebook[q3[2 * il + 1]];
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    if constexpr (std::is_same_v<dst_t, __half>) {
        // the eight values are contiguous and eight-aligned: four aligned half2 stores, not eight
#pragma unroll
        for (int j = 0; j < 4; j += 2) {
            const float l0 = d * (float) (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            const float l1 = d * (float) (uint8_t) (grid1 >> (8 * j + 8)) * (signs & kmask_iq2xs[j + 1] ? -1.f : 1.f);
            const float h0 = d * (float) (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            const float h1 = d * (float) (uint8_t) (grid2 >> (8 * j + 8)) * (signs & kmask_iq2xs[j + 5] ? -1.f : 1.f);
            *reinterpret_cast<__half2*>(y + j) = __floats2half2_rn(l0, l1);
            *reinterpret_cast<__half2*>(y + j + 4) = __floats2half2_rn(h0, h1);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y[j + 0] = cvt<dst_t>(d * (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
            y[j + 4] = cvt<dst_t>(d * (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
        }
    }
}
// And the same 128-value half for IQ2_S: `h`, `sub`, `il`, `ib4` and the output mapping are dq_iq3_s_half's, while
// the 1024-entry 64-bit grid, the nibble scale (bits 0..3 for il 0/1, 4..7 for il 2/3) and the sign byte at
// qs[QK_K/8 + 4*ib + il] are dq_iq2_s's own expressions.  Its eight values are one contiguous group, so the half
// path is four half2 stores with nothing split between two grids.
template<typename dst_t>
__device__ void dq_iq2_s_half(const void* vx, int64_t ibs, int h, int sub, dst_t* yy, const uint64_t* codebook = iq2s_grid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int il = sub >> 2, ib4 = sub & 3, ib = 4 * h + ib4;
    dst_t* y = yy + 32 * ib4 + 8 * il;
    const uint64_t grid = codebook[x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)];
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    if constexpr (std::is_same_v<dst_t, __half>) {
        // the eight values are contiguous and eight-aligned: four aligned half2 stores, not eight
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const float l0 = d * (float) (uint8_t) (grid >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            const float l1 = d * (float) (uint8_t) (grid >> (8 * j + 8)) * (signs & kmask_iq2xs[j + 1] ? -1.f : 1.f);
            *reinterpret_cast<__half2*>(y + j) = __floats2half2_rn(l0, l1);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j)
            y[j] = cvt<dst_t>(d * (uint8_t) (grid >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq1_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_s* x = (const block_iq1_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const float delta = x[ibs].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float) x[ibs].d * (2 * ((x[ibs].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[ib] >> 3 * il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template<typename dst_t>
__device__ void dq_iq1_m(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template<typename dst_t>
__device__ void dq_iq4_nl(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template<typename dst_t>
__device__ void dq_q3_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template<typename dst_t>
__device__ void dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid, const int8_t* codebook = kvalues_iq4nl) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * codebook[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * codebook[q4[j] >> 4]);
    }
}
template<typename dst_t>
__device__ void dq_q2_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        yy[b * 64 + i] = cvt<dst_t>(d * (float) (code - 1));
    }
}

// Q5_0 (llama.cpp's dequantize_q5_0): 8 blocks of 32 per "superblock" of 256, laid out like dq_iq4_nl
template<typename dst_t>
__device__ void dq_q5_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_0* x = (const block_q5_0*) vx + ibs * (QK_K / QK5_0);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* qs = x[ib].qs + 4 * il;
    const uint32_t qh = (uint32_t) x[ib].qh[0] | ((uint32_t) x[ib].qh[1] << 8) | ((uint32_t) x[ib].qh[2] << 16) |
                        ((uint32_t) x[ib].qh[3] << 24);
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        const int k = 4 * (int) il + j;
        const int x0 = (qs[j] & 0xf) | (((qh >> k) << 4) & 0x10);
        const int x1 = (qs[j] >> 4) | ((qh >> (k + 12)) & 0x10);
        y[j + 0] = cvt<dst_t>((x0 - 16) * d);
        y[j + 16] = cvt<dst_t>((x1 - 16) * d);
    }
}

// Q5_1 dequantizer: 8 blocks of 32 per 256-element superblock (w = d * q5 + m)
template<typename dst_t>
__device__ void dq_q5_1(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_1* x = (const block_q5_1*) vx + ibs * (QK_K / QK5_1);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* qs = x[ib].qs + 4 * il;
    const uint32_t qh = (uint32_t) x[ib].qh[0] | ((uint32_t) x[ib].qh[1] << 8) | ((uint32_t) x[ib].qh[2] << 16) |
                        ((uint32_t) x[ib].qh[3] << 24);
    const float2 dm = __half22float2(x[ib].dm);
    for (int j = 0; j < 4; ++j) {
        const int k = 4 * (int) il + j;
        const int x0 = (qs[j] & 0xf) | (((qh >> k) << 4) & 0x10);
        const int x1 = (qs[j] >> 4) | ((qh >> (k + 12)) & 0x10);
        y[j + 0] = cvt<dst_t>(x0 * dm.x + dm.y);
        y[j + 16] = cvt<dst_t>(x1 * dm.x + dm.y);
    }
}

// Q8_0 dequantizer: 8 blocks of 32 per 256-element superblock
template<typename dst_t>
__device__ void dq_q8_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q8_0* x = (const block_q8_0*) vx + ibs * (QK_K / QK8_0);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const int8_t* qs = x[ib].qs + 8 * il;
    const float d = __half2float(x[ib].d);
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>((float) qs[j] * d);
}

// Q4_K dequantizer: 256 elements per block, 32 threads x 8 elements
template<typename dst_t>
__device__ void dq_q4_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q4_K* x = (const block_q4_K*) vx + ibs;
    const float2 dm = __half22float2(x->dm);
    const int il = tid / 8, ir = tid % 8; // 4 chunks of 64, 8 threads per chunk -> 4 lo + 4 hi
    const int is = 2 * il;
    auto get_sm = [&](int j, int& sc, int& m) {
        if (j < 4) { sc = x->scales[j] & 63; m = x->scales[j + 4] & 63; }
        else { sc = (x->scales[j + 4] & 0xf) | ((x->scales[j - 4] >> 6) << 4);
               m  = (x->scales[j + 4] >> 4)  | ((x->scales[j] >> 6) << 4); }
    };
    int sc0, m0, sc1, m1;
    get_sm(is + 0, sc0, m0);
    get_sm(is + 1, sc1, m1);
    const float d0 = dm.x * sc0, mn0 = dm.y * m0;
    const float d1 = dm.x * sc1, mn1 = dm.y * m1;
    const uint8_t* q = x->qs + il * 32 + ir * 4;
    dst_t* y = yy + il * 64 + ir * 4;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d0 * (float) (q[j] & 0xf) - mn0);
        y[j + 32] = cvt<dst_t>(d1 * (float) (q[j] >> 4) - mn1);
    }
}

// Q5_K dequantizer: 256 elements per block, 32 threads x 8 elements
template<typename dst_t>
__device__ void dq_q5_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_K* x = (const block_q5_K*) vx + ibs;
    const float2 dm = __half22float2(x->dm);
    const int il = tid / 8, ir = tid % 8;
    const int is = 2 * il;
    auto get_sm = [&](int j, int& sc, int& m) {
        if (j < 4) { sc = x->scales[j] & 63; m = x->scales[j + 4] & 63; }
        else { sc = (x->scales[j + 4] & 0xf) | ((x->scales[j - 4] >> 6) << 4);
               m  = (x->scales[j + 4] >> 4)  | ((x->scales[j] >> 6) << 4); }
    };
    int sc0, m0, sc1, m1;
    get_sm(is + 0, sc0, m0);
    get_sm(is + 1, sc1, m1);
    const float d0 = dm.x * sc0, mn0 = dm.y * m0;
    const float d1 = dm.x * sc1, mn1 = dm.y * m1;
    const uint8_t* ql = x->qs + il * 32 + ir * 4;
    const uint8_t* qh = x->qh + ir * 4;
    const uint8_t hm0 = (uint8_t) (1u << (2 * il)), hm1 = (uint8_t) (1u << (2 * il + 1));
    dst_t* y = yy + il * 64 + ir * 4;
    for (int j = 0; j < 4; ++j) {
        y[j + 0]  = cvt<dst_t>(d0 * (float) ((ql[j] & 0xf) + ((qh[j] & hm0) ? 16 : 0)) - mn0);
        y[j + 32] = cvt<dst_t>(d1 * (float) ((ql[j] >> 4)  + ((qh[j] & hm1) ? 16 : 0)) - mn1);
    }
}

template<typename dst_t>
__device__ __forceinline__ void dq_dispatch(int ty, const void* vx, int64_t ibs, dst_t* y, int tid) {
    switch (ty) {
        case 6: dq_q5_0(vx, ibs, y, tid); break;
        case 7: dq_q5_1(vx, ibs, y, tid); break;
        case 8: dq_q8_0(vx, ibs, y, tid); break;
        case 12: dq_q4_k(vx, ibs, y, tid); break;
        case 13: dq_q5_k(vx, ibs, y, tid); break;
        case 16: dq_iq2_xxs(vx, ibs, y, tid); break;
        case 17: dq_iq2_xs(vx, ibs, y, tid); break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid); break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 19: dq_iq1_s(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template<typename dst_t>
__global__ void dequant_flat_kernel(int ty, const void* __restrict__ vx, dst_t* __restrict__ y) {
    const int64_t i = blockIdx.x;
    dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, threadIdx.x);
}
// gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
__global__ void dequant_gu_kernel(int ty, const void* __restrict__ gate, const void* __restrict__ up, int64_t per_row,
                                  __half* __restrict__ y) {
    const int64_t i = blockIdx.x;
    const int parity = blockIdx.y;
    const int64_t r = i / per_row, c = i % per_row;
    dq_dispatch<__half>(ty, parity ? up : gate, i, y + ((2 * r + parity) * per_row + c) * QK_K, threadIdx.x);
}

// dequant_gu_kernel for a mixed layer: the gate blocks by tg, the up blocks by tu (the same interleaved rows)
__global__ void dequant_gu2_kernel(int tg, int tu, const void* __restrict__ gate, const void* __restrict__ up,
                                   int64_t per_row, __half* __restrict__ y) {
    const int64_t i = blockIdx.x;
    const int parity = blockIdx.y;
    const int64_t r = i / per_row, c = i % per_row;
    dq_dispatch<__half>(parity ? tu : tg, parity ? up : gate, i, y + ((2 * r + parity) * per_row + c) * QK_K,
                        threadIdx.x);
}

bool is_iq(int t) { return t == 16 || t == 17 || t == 18 || t == 19 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11; }
bool is_dq(int t) { return is_iq(t) || t == 6 || t == 7 || t == 8 || t == 12 || t == 13; }

}  // namespace

bool iq_supported(int t) noexcept { return is_iq(t); }
bool iq_dequant_supported(int t) noexcept { return is_dq(t); }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 2: return (size_t) (n / 32) * sizeof(block_q4_0);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 19: return (size_t) (n / 256) * sizeof(block_iq1_s);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 10: return (size_t) (n / 256) * sizeof(block_q2_K);   // glm-dsa B3.5: native_expert_layout's Q2_K rows
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    quantize_q8_1_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, (block_q8_1*) y, n);
    check("quantize_q8_1_rows");
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const dim3 grid((unsigned) ((n_out + 3) / 4)), block(32, 4);
    const size_t rb = iq_row_bytes(t, n_in);
    cudaStream_t s = (cudaStream_t) stream;
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    switch (t) {
        case 16: launch_mmvq<16>(W, rb, X, y, n_in, n_out, ncols, grid, block, s); break;
        case 17: launch_mmvq<17>(W, rb, X, y, n_in, n_out, ncols, grid, block, s); break;
        case 18: launch_mmvq<18>(W, rb, X, y, n_in, n_out, ncols, grid, block, s); break;
        case 20: launch_mmvq<20>(W, rb, X, y, n_in, n_out, ncols, grid, block, s); break;
        case 21: launch_mmvq<21>(W, rb, X, y, n_in, n_out, ncols, grid, block, s); break;
        case 22: launch_mmvq<22>(W, rb, X, y, n_in, n_out, ncols, grid, block, s); break;
        case 23: launch_mmvq<23>(W, rb, X, y, n_in, n_out, ncols, grid, block, s); break;
        case 29: launch_mmvq<29>(W, rb, X, y, n_in, n_out, ncols, grid, block, s); break;
        case 19: launch_mmvq<19>(W, rb, X, y, n_in, n_out, ncols, grid, block, s); break;
        case 42: launch_mmvq<42>(W, rb, X, y, n_in, n_out, ncols, grid, block, s); break;
        case 13: launch_mmvq<13>(W, rb, X, y, n_in, n_out, ncols, grid, block, s); break;
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    // (+ Q5_0 / Q5_1 / Q8_0 / Q4_K / Q5_K: routed expert & embedding types of unsloth's Qwen3.8 UD-IQ2_XS & UD-Q4_K_XL)
    if (n % 256 != 0 || !is_dq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    dequant_flat_kernel<__half><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, (__half*) dst);
    check("iq_dequant_f16");
}

namespace {
__global__ void embed_rows_kernel(int ty, const uint8_t* __restrict__ table, size_t row_bytes,
                                  const int32_t* __restrict__ tokens, int64_t n_embd, float* __restrict__ y) {
    const int t = blockIdx.y;
    const int64_t b = blockIdx.x;
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    dq_dispatch<float>(ty, row, b, y + (size_t) t * n_embd + b * QK_K, threadIdx.x);
}
}  // namespace

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !is_dq(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    embed_rows_kernel<<<dim3((unsigned) (n_embd / 256), (unsigned) n_tok), 32, 0, (cudaStream_t) stream>>>(
        t, (const uint8_t*) table, row_bytes, tokens, n_embd, out);
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !is_dq(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    dequant_flat_kernel<float><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, dst);
    check("iq_dequant_f32");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    const int64_t per_row = n_embd / 256;
    dequant_gu_kernel<<<dim3((unsigned) (n_ff * per_row), 2), 32, 0, (cudaStream_t) stream>>>(t, gate, up, per_row,
                                                                                           (__half*) dst);
    check("iq_dequant_gu_f16");
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    L.u_type = gu_type;
    L.u_row = L.gu_row;
    return L;
}

NativeExpertLayout native_expert_layout(int gu_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L = native_expert_layout(gu_type, d_type, n_embd, n_ff);
    if (up_type == gu_type || up_type < 0) return L;
    L.u_type = up_type;
    L.u_row = iq_row_bytes(up_type, n_embd);
    L.down_off = L.up_off + (size_t) n_ff * L.u_row;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

void iq_dequant_gu_f16_mixed(int gate_type, int up_type, const void* gate, const void* up, int64_t n_ff,
                             int64_t n_embd, uint16_t* dst, void* stream) {
    if (up_type == gate_type) {
        iq_dequant_gu_f16(gate_type, gate, up, n_ff, n_embd, dst, stream);
        return;
    }
    const int64_t per_row = n_embd / 256;
    dequant_gu2_kernel<<<dim3((unsigned) (n_ff * per_row), 2), 32, 0, (cudaStream_t) stream>>>(
        gate_type, up_type, gate, up, per_row, (__half*) dst);
    check("iq_dequant_gu_f16_mixed");
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}

static bool iq_gu_soa_ok(const NativeExpertLayout& L) noexcept {
    if (L.gu_type == 12 && !iq_p100_hfma2_on()) return false;
    return (L.u_type < 0 || L.u_type == L.gu_type) &&
           (L.gu_type == 12 || L.gu_type == 16 || L.gu_type == 22) &&
           L.n_embd >= 256 && (L.n_embd & 255) == 0 && ((2 * L.n_ff) & 31) == 0;
}

static bool iq_dn_soa_ok(const NativeExpertLayout& L) noexcept {
    if ((L.d_type == 7 || L.d_type == 8) && !iq_p100_hfma2_on()) return false;
    return (L.down_off & 15) == 0 && L.n_ff >= 64 && (L.n_ff & 63) == 0 &&
           (L.d_type == 7 || L.d_type == 8 || L.d_type == 20 || L.d_type == 42);
}

// ---- Fast GP100 K-Major SoA -> FP16 dequantizers for zero-PCIe VRAM-resident prefill ----
__global__ void __launch_bounds__(128) dequant_iq2xxs_gu_soa_f16_kernel(
    const uint8_t* __restrict__ blob, int n_ff, int n_embd, uint16_t* __restrict__ dst) {
    const int N = 2 * n_ff;
    const int K_blk = n_embd / 256;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row = (int) blockIdx.x * 32 + lane;
    const int b = (int) blockIdx.y;
    const uint4* Wq = reinterpret_cast<const uint4*>(blob);
    const __half* Wd = reinterpret_cast<const __half*>(blob + (size_t) 64 * K_blk * N);
    __shared__ uint4 sgrid[256];
    __shared__ uint4 ssign[128];
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    for (int i = tid; i < 256; i += 128) {
        const uint2 gp = ((const uint2*) iq2xxs_grid)[i];
        const uint32_t ha = __byte_perm(gp.x, 0x64006400u, 0x5150);
        const uint32_t hb = __byte_perm(gp.x, 0x64006400u, 0x5352);
        const uint32_t hc = __byte_perm(gp.y, 0x64006400u, 0x5150);
        const uint32_t hd = __byte_perm(gp.y, 0x64006400u, 0x5352);
        const __half2 va = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
        const __half2 vb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
        const __half2 vc = __hsub2(*reinterpret_cast<const __half2*>(&hc), c1024);
        const __half2 vd = __hsub2(*reinterpret_cast<const __half2*>(&hd), c1024);
        uint4 out;
        out.x = *reinterpret_cast<const uint32_t*>(&va);
        out.y = *reinterpret_cast<const uint32_t*>(&vb);
        out.z = *reinterpret_cast<const uint32_t*>(&vc);
        out.w = *reinterpret_cast<const uint32_t*>(&vd);
        sgrid[i] = out;
    }
    if (tid < 128) {
        const uint32_t sg = (uint32_t) tid | ((__popc((uint32_t) tid) & 1u) << 7);
        ssign[tid] = make_sign_mask4(sg);
    }
    __syncthreads();
    if (row >= N) return;
    const int dst_row = (row < n_ff) ? (2 * row) : (2 * (row - n_ff) + 1);
    const float db = __half2float(__ldg(Wd + (size_t) b * N + row)) * 0.125f;
    const uint4 wv = __ldg(Wq + ((size_t) b * 4 + warp) * N + row);
    const uint32_t idx_a[2] = {wv.x, wv.z};
    const uint32_t aux_a[2] = {wv.y, wv.w};
    uint4* out4 = reinterpret_cast<uint4*>(dst + (size_t) dst_row * n_embd + b * 256 + warp * 64);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t idx32 = idx_a[h];
        const uint32_t aux32 = aux_a[h];
        const __half2 sc = __float2half2_rn(db * (float) ((aux32 >> 27) | 1u));
#pragma unroll
        for (int k0 = 0; k0 < 4; ++k0) {
            const uint4 gv = sgrid[(idx32 >> (k0 * 8)) & 0xffu];
            const uint4 sm = ssign[(aux32 >> (7 * k0)) & 0x7fu];
            const uint32_t w0 = gv.x ^ sm.x, w1 = gv.y ^ sm.y, w2 = gv.z ^ sm.z, w3 = gv.w ^ sm.w;
            const __half2 r0 = __hmul2(*reinterpret_cast<const __half2*>(&w0), sc);
            const __half2 r1 = __hmul2(*reinterpret_cast<const __half2*>(&w1), sc);
            const __half2 r2 = __hmul2(*reinterpret_cast<const __half2*>(&w2), sc);
            const __half2 r3 = __hmul2(*reinterpret_cast<const __half2*>(&w3), sc);
            out4[h * 4 + k0] = make_uint4(*reinterpret_cast<const uint32_t*>(&r0), *reinterpret_cast<const uint32_t*>(&r1),
                                          *reinterpret_cast<const uint32_t*>(&r2), *reinterpret_cast<const uint32_t*>(&r3));
        }
    }
}

__global__ void __launch_bounds__(128) dequant_iq2s_gu_soa_f16_kernel(
    const uint8_t* __restrict__ blob, int n_ff, int n_embd, uint16_t* __restrict__ dst) {
    const int N = 2 * n_ff;
    const int K_blk = n_embd / 256;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row = (int) blockIdx.x * 32 + lane;
    const int b = (int) blockIdx.y;
    const uint4* Wqs = reinterpret_cast<const uint4*>(blob);
    const uint4* Wqhsc = reinterpret_cast<const uint4*>(blob + (size_t) 64 * K_blk * N);
    const __half* Wd = reinterpret_cast<const __half*>(blob + (size_t) 80 * K_blk * N);
    __shared__ uint4 ssign[256];
    for (int i = tid; i < 256; i += 128) ssign[i] = make_sign_mask4((uint32_t) i);
    __syncthreads();
    if (row >= N) return;
    const int dst_row = (row < n_ff) ? (2 * row) : (2 * (row - n_ff) + 1);
    const float db = __half2float(__ldg(Wd + (size_t) b * N + row)) * 0.125f;
    const uint4 qhsc = __ldg(Wqhsc + (size_t) b * N + row);
    const uint4* Wb = Wqs + (size_t) b * 4 * N + row;
    // warp 0..3 handles sub-blocks sb = 2*warp, 2*warp+1 (2 subblocks = 64 values per warp)
    const int half_b = warp >> 1;   // 0 -> sb 0..3, 1 -> sb 4..7
    const uint4 gv_w = __ldg(Wb + half_b * N);
    const uint4 sn_w = __ldg(Wb + (2 + half_b) * N);
    const uint32_t g_pair[2] = {(warp & 1) ? gv_w.z : gv_w.x, (warp & 1) ? gv_w.w : gv_w.y};
    const uint32_t s_pair[2] = {(warp & 1) ? sn_w.z : sn_w.x, (warp & 1) ? sn_w.w : sn_w.y};
    const uint32_t qh_w = (half_b == 0) ? qhsc.x : qhsc.y;
    const uint32_t sc_w = (half_b == 0) ? qhsc.z : qhsc.w;
    uint4* out4 = reinterpret_cast<uint4*>(dst + (size_t) dst_row * n_embd + b * 256 + warp * 64);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int sb = warp * 2 + h;
        const uint32_t g32 = g_pair[h];
        const uint32_t s32 = s_pair[h];
        const uint32_t qh = (qh_w >> ((sb & 3) * 8)) & 0xffu;
        const uint32_t sc = (sc_w >> ((sb & 3) * 8)) & 0xffu;
        const __half2 sc0 = __float2half2_rn(db * (float) ((sc & 0x0fu) * 2u + 1u));
        const __half2 sc1 = __float2half2_rn(db * (float) ((sc >> 4) * 2u + 1u));
#pragma unroll
        for (int k0 = 0; k0 < 4; ++k0) {
            const uint32_t gidx = ((g32 >> (k0 * 8)) & 0xffu) | ((qh << (8 - 2 * k0)) & 0x300u);
            const uint4 gv = __ldg(&g_iq2s_h2[gidx]);
            const uint4 sm = ssign[(s32 >> (k0 * 8)) & 0xffu];
            const __half2 sck = (k0 < 2) ? sc0 : sc1;
            const uint32_t w0 = gv.x ^ sm.x, w1 = gv.y ^ sm.y, w2 = gv.z ^ sm.z, w3 = gv.w ^ sm.w;
            const __half2 r0 = __hmul2(*reinterpret_cast<const __half2*>(&w0), sck);
            const __half2 r1 = __hmul2(*reinterpret_cast<const __half2*>(&w1), sck);
            const __half2 r2 = __hmul2(*reinterpret_cast<const __half2*>(&w2), sck);
            const __half2 r3 = __hmul2(*reinterpret_cast<const __half2*>(&w3), sck);
            out4[h * 4 + k0] = make_uint4(*reinterpret_cast<const uint32_t*>(&r0), *reinterpret_cast<const uint32_t*>(&r1),
                                          *reinterpret_cast<const uint32_t*>(&r2), *reinterpret_cast<const uint32_t*>(&r3));
        }
    }
}

__global__ void __launch_bounds__(64) dequant_q20_dn_soa_f16_kernel(
    const uint8_t* __restrict__ dn_dev, int N, int n_ff, uint16_t* __restrict__ dst) {
    const int K_blk = n_ff / 64;
    const int r = (int) blockIdx.x * 64 + threadIdx.x;
    const int b = (int) blockIdx.y;
    if (r >= N) return;
    const uint4* Wq = reinterpret_cast<const uint4*>(dn_dev);
    const __half* Wd = reinterpret_cast<const __half*>(dn_dev + (size_t) 16 * K_blk * N);
    const uint4 q4 = __ldg(Wq + (size_t) b * N + r);
    const __half dh = __ldg(Wd + (size_t) b * N + r);
    const uint32_t d = __half_as_ushort(dh);
    const uint32_t z = __half_as_ushort(__hmul(dh, __float2half(0.0f)));
    const uint32_t twice = __half_as_ushort(__hadd(dh, dh));
    const uint32_t table01 = (d ^ 0x8000u) | (z << 16);
    const uint32_t table23 = d | (twice << 16);
    const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
    uint4* out4 = reinterpret_cast<uint4*>(dst + (size_t) r * n_ff + b * 64);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const uint32_t w = wv[u];
        uint32_t h[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t qb = (w >> (c * 8)) & 0xffu;
            const unsigned a0 = (qb & 3u) << 1, b0 = ((qb >> 2) & 3u) << 1;
            const unsigned sel0 = 0x1010u | a0 | (a0 << 4) | (b0 << 8) | (b0 << 12);
            const unsigned a1 = ((qb >> 4) & 3u) << 1, b1 = ((qb >> 6) & 3u) << 1;
            const unsigned sel1 = 0x1010u | a1 | (a1 << 4) | (b1 << 8) | (b1 << 12);
            if (c < 2) {
                h[c * 2 + 0] = __byte_perm(table01, table23, sel0);
                h[c * 2 + 1] = __byte_perm(table01, table23, sel1);
            } else {
                if (c == 2) out4[u * 2 + 0] = make_uint4(h[0], h[1], h[2], h[3]);
                h[(c - 2) * 2 + 0] = __byte_perm(table01, table23, sel0);
                h[(c - 2) * 2 + 1] = __byte_perm(table01, table23, sel1);
            }
        }
        out4[u * 2 + 1] = make_uint4(h[0], h[1], h[2], h[3]);
    }
}

__global__ void __launch_bounds__(64) dequant_iq4nl_dn_soa_f16_kernel(
    const uint8_t* __restrict__ dn_dev, int N, int n_ff, uint16_t* __restrict__ dst) {
    const int K_blk = n_ff / 32;
    const int r = (int) blockIdx.x * 64 + threadIdx.x;
    const int b = (int) blockIdx.y;
    if (r >= N) return;
    const uint4* Wq = reinterpret_cast<const uint4*>(dn_dev);
    const __half* Wd = reinterpret_cast<const __half*>(dn_dev + (size_t) 16 * K_blk * N);
    const uint4 q4 = __ldg(Wq + (size_t) b * N + r);
    const __half dh = __ldg(Wd + (size_t) b * N + r);
    const __half2 dh2 = __halves2half2(dh, dh);
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    constexpr uint32_t T0 = 0x3f2d1801u, T1 = 0x766a5d4fu, T2 = 0xa6998d81u, T3 = 0xf1d9c5b5u;
    const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
    uint2* out2 = reinterpret_cast<uint2*>(dst + (size_t) r * n_ff + b * 32);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const uint2 v = iq4nl_lut_reg(wv[u], T0, T1, T2, T3);
        const uint32_t a0 = __byte_perm(0x64006400u, v.x, 0x3514);
        const uint32_t a1 = __byte_perm(0x64006400u, v.x, 0x3716);
        const uint32_t b0 = __byte_perm(0x64006400u, v.y, 0x3514);
        const uint32_t b1 = __byte_perm(0x64006400u, v.y, 0x3716);
        const __half2 ra0 = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&a0), c1152), dh2);
        const __half2 ra1 = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&a1), c1152), dh2);
        const __half2 rb0 = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&b0), c1152), dh2);
        const __half2 rb1 = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&b1), c1152), dh2);
        out2[u] = make_uint2(*reinterpret_cast<const uint32_t*>(&ra0), *reinterpret_cast<const uint32_t*>(&ra1));
        out2[4 + u] = make_uint2(*reinterpret_cast<const uint32_t*>(&rb0), *reinterpret_cast<const uint32_t*>(&rb1));
    }
}

__global__ void __launch_bounds__(128) dequant_q4k_gu_soa_f16_kernel(
    const uint8_t* __restrict__ blob, int n_ff, int n_embd, uint16_t* __restrict__ dst) {
    const int N = 2 * n_ff;
    const int K_blk = n_embd / 256;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row = (int) blockIdx.x * 32 + lane;
    const int b = (int) blockIdx.y;
    if (row >= N) return;
    const uint4* Wqs = reinterpret_cast<const uint4*>(blob);
    const uint4* Wscdm = reinterpret_cast<const uint4*>(blob + (size_t) 128 * K_blk * N);
    const uint4 scdm = __ldg(Wscdm + (size_t) b * N + row);
    const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&scdm.w));
    const uint32_t s0 = scdm.x, s1 = scdm.y, s2 = scdm.z;
    const uint32_t sc_w[2] = {s0 & 0x3f3f3f3fu, (s2 & 0x0f0f0f0fu) | ((s0 >> 2) & 0x30303030u)};
    const uint32_t m_w[2]  = {s1 & 0x3f3f3f3fu, ((s2 >> 4) & 0x0f0f0f0fu) | ((s1 >> 2) & 0x30303030u)};
    const int c = warp; // 4 warps = 4 chunks of 64 values
    const int sb0 = c * 2, sb1 = c * 2 + 1;
    const float d0 = dm.x * (float) ((sc_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
    const float m0 = dm.y * (float) ((m_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
    const float d1 = dm.x * (float) ((sc_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
    const float m1 = dm.y * (float) ((m_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
    const uint4 qa = __ldg(Wqs + ((size_t) b * 8 + c * 2 + 0) * N + row);
    const uint4 qb = __ldg(Wqs + ((size_t) b * 8 + c * 2 + 1) * N + row);
    const uint32_t wv[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
    const int dst_row = (row < n_ff) ? (2 * row) : (2 * (row - n_ff) + 1);
    uint16_t* out = dst + (size_t) dst_row * n_embd + b * 256 + c * 64;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const uint32_t w = wv[u];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t by = (w >> (k * 8)) & 0xffu;
            out[u * 4 + k]      = __half_as_ushort(__float2half_rn(d0 * (float) (by & 0xfu) - m0));
            out[32 + u * 4 + k] = __half_as_ushort(__float2half_rn(d1 * (float) (by >> 4) - m1));
        }
    }
}

__global__ void __launch_bounds__(64) dequant_q51_dn_soa_f16_kernel(
    const uint8_t* __restrict__ dn_dev, int N, int n_ff, uint16_t* __restrict__ dst) {
    const int K_blk = n_ff / 32;
    const int r = (int) blockIdx.x * 64 + threadIdx.x;
    const int b = (int) blockIdx.y;
    if (r >= N) return;
    const uint4* Wqs = reinterpret_cast<const uint4*>(dn_dev);
    const uint2* Wdmqh = reinterpret_cast<const uint2*>(dn_dev + (size_t) 16 * K_blk * N);
    const uint4 q4 = __ldg(Wqs + (size_t) b * N + r);
    const uint2 dmqh = __ldg(Wdmqh + (size_t) b * N + r);
    const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&dmqh.x));
    const uint32_t qh = dmqh.y;
    const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
    uint16_t* out = dst + (size_t) r * n_ff + b * 32;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const uint32_t w = wv[u];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int j = u * 4 + k;
            const uint32_t by = (w >> (k * 8)) & 0xffu;
            const int v0 = (int) ((by & 0xfu) | (((qh >> (k * 8 + u)) & 1u) << 4));
            const int v1 = (int) ((by >> 4) | (((qh >> (k * 8 + 4 + u)) & 1u) << 4));
            out[j]      = __half_as_ushort(__float2half_rn((float) v0 * dm.x + dm.y));
            out[16 + j] = __half_as_ushort(__float2half_rn((float) v1 * dm.x + dm.y));
        }
    }
}

__global__ void __launch_bounds__(64) dequant_q80_dn_soa_f16_kernel(
    const uint8_t* __restrict__ dn_dev, int N, int n_ff, uint16_t* __restrict__ dst) {
    const int K_blk = n_ff / 32;
    const int r = (int) blockIdx.x * 64 + threadIdx.x;
    const int b = (int) blockIdx.y;
    if (r >= N) return;
    const uint4* Wqs = reinterpret_cast<const uint4*>(dn_dev);
    const __half* Wd = reinterpret_cast<const __half*>(dn_dev + (size_t) 32 * K_blk * N);
    const float d = __half2float(__ldg(Wd + (size_t) b * N + r));
    uint16_t* out = dst + (size_t) r * n_ff + b * 32;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
        const uint4 qv = __ldg(Wqs + ((size_t) b * 2 + hf) * N + r);
        const uint8_t* q8 = reinterpret_cast<const uint8_t*>(&qv);
#pragma unroll
        for (int k = 0; k < 16; ++k) out[hf * 16 + k] = __half_as_ushort(__float2half_rn((float) ((int) q8[k] - 128) * d));
    }
}

bool iq_gu_soa_eligible(const NativeExpertLayout& L) noexcept { return iq_gu_soa_ok(L); }
bool iq_dn_soa_eligible(const NativeExpertLayout& L) noexcept { return iq_dn_soa_ok(L); }
bool iq_expert_soa_eligible(const NativeExpertLayout& L) noexcept {
    return iq_gu_soa_ok(L) || iq_dn_soa_ok(L);
}

void iq_dequant_gu_f16_soa(int gu_type, const void* blob_dev, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    const dim3 grid((unsigned) ((2 * n_ff + 31) / 32), (unsigned) (n_embd / 256));
    if (gu_type == 12) {
        dequant_q4k_gu_soa_f16_kernel<<<grid, 128, 0, s>>>((const uint8_t*) blob_dev, (int) n_ff, (int) n_embd, dst);
    } else if (gu_type == 16) {
        dequant_iq2xxs_gu_soa_f16_kernel<<<grid, 128, 0, s>>>((const uint8_t*) blob_dev, (int) n_ff, (int) n_embd, dst);
    } else if (gu_type == 22) {
        dequant_iq2s_gu_soa_f16_kernel<<<grid, 128, 0, s>>>((const uint8_t*) blob_dev, (int) n_ff, (int) n_embd, dst);
    } else {
        std::fprintf(stderr, "iq_dequant_gu_f16_soa: unsupported type %d\n", gu_type);
        std::exit(1);
    }
    check("iq_dequant_gu_f16_soa");
}

void iq_dequant_dn_f16_soa(int d_type, const void* dn_dev, int64_t n_embd, int64_t n_ff, uint16_t* dst, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    if (d_type == 42) {
        const dim3 grid((unsigned) ((n_embd + 63) / 64), (unsigned) (n_ff / 64));
        dequant_q20_dn_soa_f16_kernel<<<grid, 64, 0, s>>>((const uint8_t*) dn_dev, (int) n_embd, (int) n_ff, dst);
    } else if (d_type == 20) {
        const dim3 grid((unsigned) ((n_embd + 63) / 64), (unsigned) (n_ff / 32));
        dequant_iq4nl_dn_soa_f16_kernel<<<grid, 64, 0, s>>>((const uint8_t*) dn_dev, (int) n_embd, (int) n_ff, dst);
    } else if (d_type == 7) {
        const dim3 grid((unsigned) ((n_embd + 63) / 64), (unsigned) (n_ff / 32));
        dequant_q51_dn_soa_f16_kernel<<<grid, 64, 0, s>>>((const uint8_t*) dn_dev, (int) n_embd, (int) n_ff, dst);
    } else if (d_type == 8) {
        const dim3 grid((unsigned) ((n_embd + 63) / 64), (unsigned) (n_ff / 32));
        dequant_q80_dn_soa_f16_kernel<<<grid, 64, 0, s>>>((const uint8_t*) dn_dev, (int) n_embd, (int) n_ff, dst);
    } else {
        std::fprintf(stderr, "iq_dequant_dn_f16_soa: unsupported type %d\n", d_type);
        std::exit(1);
    }
    check("iq_dequant_dn_f16_soa");
}

// =================================================================================================
// GP100 Fused Grouped SoA HFMA2 Prefill MoE: 1 launch for ALL 512 experts!
// Uses GP100 32-bit bank-split SoA shared-memory codebooks (sg_lo[] + sg_hi[], 2x LDS.32, ZERO bank conflicts,
// ZERO stack frames, ZERO register spills, 256 threads/block staging 2 tokens in parallel -> HALF norm work!).
// =================================================================================================
__global__ void __launch_bounds__(256, 2) p100_prefill_iq2xxs_gu_soa_kernel(
    const unsigned long long* __restrict__ exp_ptr, const int32_t* __restrict__ exp_off,
    int n_ff, int n_embd, const uint16_t* __restrict__ xs_f16, float* __restrict__ gu_out) {
    const int g = (int) blockIdx.y;
    const int o0 = exp_off[g], o1 = exp_off[g + 1];
    if (o0 >= o1) return;

    const int tid = threadIdx.x;
    const int N = 2 * n_ff;
    const int K_blk = n_embd / 256;
    const int row = (int) blockIdx.x * 256 + tid;
    const bool valid = row < N;

    const uint8_t* blob = (const uint8_t*) exp_ptr[g];
    const uint4* Wq = reinterpret_cast<const uint4*>(blob);
    const __half* Wd = reinterpret_cast<const __half*>(blob + (size_t) 64 * K_blk * N);

    __shared__ uint32_t sg_lo[256], sg_hi[256];       // 2048 B (32-bit bank-split LDS.32!)
    __shared__ __align__(16) __half2 sxh[8][128];     // 4096 B (TM=8 + LDS.128 broadcast!)
    __shared__ float sxd[8][8];                       //  256 B -> total 6,400 B!

    {
        const uint2 gp = __ldg(reinterpret_cast<const uint2*>(iq2xxs_grid) + tid);
        sg_lo[tid] = gp.x;
        sg_hi[tid] = gp.y;
    }
    __syncthreads();
    const int st_slot = tid >> 7;  // 0 or 1: 256 threads stage 2 tokens simultaneously!
    const int st_col = tid & 127;

    for (int eb = o0; eb < o1; eb += 8) {
        const int nt = (eb + 8 <= o1) ? 8 : (o1 - eb);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const int np = (nt + 1) >> 1;

        for (int b = 0; b < K_blk; ++b) {
            for (int p = 0; p < np; ++p) {
                const int t = p * 2 + st_slot;
                if (t < nt) {
                    const __half2* xrow = reinterpret_cast<const __half2*>(xs_f16 + (size_t) (eb + t) * n_embd + b * 256);
                    const __half2 v = __ldg(xrow + st_col);
                    const float2 vf = __half22float2(v);
                    float am = fmaxf(fabsf(vf.x), fabsf(vf.y));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
                    const float inv = (am > 1e-12f) ? __fdividef(1.0f, am) : 0.0f;
                    sxh[t][st_col] = __floats2half2_rn(vf.x * inv, vf.y * inv);
                    if ((st_col & 15) == 0) sxd[t][st_col >> 4] = am;
                }
            }
            __syncthreads();

            if (valid) {
                const uint4* Wqb = Wq + (size_t) b * 4 * N + row;
                if (nt > 4) {
                    float bsum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    for (int q = 0; q < 4; ++q) {
                        const uint4 wv = __ldg(Wqb + (size_t) q * N);
                        const uint32_t idx_a[2] = {wv.x, wv.z};
                        const uint32_t aux_a[2] = {wv.y, wv.w};
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int sb = q * 2 + h;
                            const uint32_t idx32 = idx_a[h];
                            const uint32_t aux32 = aux_a[h];
                            const float ls = (float) ((aux32 >> 27) | 1u);
                            __half2 d[8];
#pragma unroll
                            for (int t = 0; t < 8; ++t) d[t] = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                            for (int k0 = 0; k0 < 4; ++k0) {
                                const uint32_t gidx = (idx32 >> (k0 * 8)) & 0xffu;
                                const uint32_t gx = sg_lo[gidx], gy = sg_hi[gidx];
                                const uint32_t s7 = (aux32 >> (7 * k0)) & 0x7fu;
                                const uint32_t sg = s7 | ((__popc(s7) & 1u) << 7);
                                const __half2 w0 = iq_decode_h2(gx, 0x5150, ((sg & 1u) << 15) | ((sg & 2u) << 30));
                                const __half2 w1 = iq_decode_h2(gx, 0x5352, ((sg & 4u) << 13) | ((sg & 8u) << 28));
                                const __half2 w2 = iq_decode_h2(gy, 0x5150, ((sg & 16u) << 11) | ((sg & 32u) << 26));
                                const __half2 w3 = iq_decode_h2(gy, 0x5352, ((sg & 64u) << 9) | ((sg & 128u) << 24));
                                const int xoff = sb * 16 + k0 * 4;
#pragma unroll
                                for (int t = 0; t < 8; ++t) {
                                    const uint4 xv = *reinterpret_cast<const uint4*>(&sxh[t][xoff]);
                                    __half2 dt = d[t];
                                    dt = __hfma2(w0, *reinterpret_cast<const __half2*>(&xv.x), dt);
                                    dt = __hfma2(w1, *reinterpret_cast<const __half2*>(&xv.y), dt);
                                    dt = __hfma2(w2, *reinterpret_cast<const __half2*>(&xv.z), dt);
                                    dt = __hfma2(w3, *reinterpret_cast<const __half2*>(&xv.w), dt);
                                    d[t] = dt;
                                }
                            }
#pragma unroll
                            for (int t = 0; t < 8; ++t)
                                bsum[t] = fmaf((__low2float(d[t]) + __high2float(d[t])) * ls, sxd[t][sb], bsum[t]);
                        }
                    }
                    const float db = __half2float(__ldg(Wd + (size_t) b * N + row)) * 0.125f;
#pragma unroll
                    for (int t = 0; t < 8; ++t) acc[t] = fmaf(db, bsum[t], acc[t]);
                } else {
                    float bsum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int q = 0; q < 4; ++q) {
                        const uint4 wv = __ldg(Wqb + (size_t) q * N);
                        const uint32_t idx_a[2] = {wv.x, wv.z};
                        const uint32_t aux_a[2] = {wv.y, wv.w};
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int sb = q * 2 + h;
                            const uint32_t idx32 = idx_a[h];
                            const uint32_t aux32 = aux_a[h];
                            const float ls = (float) ((aux32 >> 27) | 1u);
                            __half2 d[4];
#pragma unroll
                            for (int t = 0; t < 4; ++t) d[t] = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                            for (int k0 = 0; k0 < 4; ++k0) {
                                const uint32_t gidx = (idx32 >> (k0 * 8)) & 0xffu;
                                const uint32_t gx = sg_lo[gidx], gy = sg_hi[gidx];
                                const uint32_t s7 = (aux32 >> (7 * k0)) & 0x7fu;
                                const uint32_t sg = s7 | ((__popc(s7) & 1u) << 7);
                                const __half2 w0 = iq_decode_h2(gx, 0x5150, ((sg & 1u) << 15) | ((sg & 2u) << 30));
                                const __half2 w1 = iq_decode_h2(gx, 0x5352, ((sg & 4u) << 13) | ((sg & 8u) << 28));
                                const __half2 w2 = iq_decode_h2(gy, 0x5150, ((sg & 16u) << 11) | ((sg & 32u) << 26));
                                const __half2 w3 = iq_decode_h2(gy, 0x5352, ((sg & 64u) << 9) | ((sg & 128u) << 24));
                                const int xoff = sb * 16 + k0 * 4;
#pragma unroll
                                for (int t = 0; t < 4; ++t) {
                                    const uint4 xv = *reinterpret_cast<const uint4*>(&sxh[t][xoff]);
                                    __half2 dt = d[t];
                                    dt = __hfma2(w0, *reinterpret_cast<const __half2*>(&xv.x), dt);
                                    dt = __hfma2(w1, *reinterpret_cast<const __half2*>(&xv.y), dt);
                                    dt = __hfma2(w2, *reinterpret_cast<const __half2*>(&xv.z), dt);
                                    dt = __hfma2(w3, *reinterpret_cast<const __half2*>(&xv.w), dt);
                                    d[t] = dt;
                                }
                            }
#pragma unroll
                            for (int t = 0; t < 4; ++t)
                                bsum[t] = fmaf((__low2float(d[t]) + __high2float(d[t])) * ls, sxd[t][sb], bsum[t]);
                        }
                    }
                    const float db = __half2float(__ldg(Wd + (size_t) b * N + row)) * 0.125f;
#pragma unroll
                    for (int t = 0; t < 4; ++t) acc[t] = fmaf(db, bsum[t], acc[t]);
                }
            }
            __syncthreads();
        }
        if (valid) {
            for (int t = 0; t < nt; ++t) gu_out[(size_t) (eb + t) * N + row] = acc[t];
        }
    }
}

__global__ void __launch_bounds__(256, 2) p100_prefill_iq2s_gu_soa_kernel(
    const unsigned long long* __restrict__ exp_ptr, const int32_t* __restrict__ exp_off,
    int n_ff, int n_embd, const uint16_t* __restrict__ xs_f16, float* __restrict__ gu_out) {
    const int g = (int) blockIdx.y;
    const int o0 = exp_off[g], o1 = exp_off[g + 1];
    if (o0 >= o1) return;

    const int tid = threadIdx.x;
    const int N = 2 * n_ff;
    const int K_blk = n_embd / 256;
    const int row = (int) blockIdx.x * 256 + tid;
    const bool valid = row < N;

    const uint8_t* blob = (const uint8_t*) exp_ptr[g];
    const uint4* Wqs = reinterpret_cast<const uint4*>(blob);
    const uint4* Wqhsc = reinterpret_cast<const uint4*>(blob + (size_t) 64 * K_blk * N);
    const __half* Wd = reinterpret_cast<const __half*>(blob + (size_t) 80 * K_blk * N);

    __shared__ uint32_t sg_lo[1024], sg_hi[1024];     // 8192 B (32-bit bank-split LDS.32!)
    __shared__ __align__(16) __half2 sxh[8][128];     // 4096 B (TM=8 + LDS.128 broadcast!)
    __shared__ float sxd[8][8];                       //  256 B -> total 12,544 B (+23.6% vs TM=4!)

    for (int i = tid; i < 1024; i += 256) {
        const uint2 gp = __ldg(reinterpret_cast<const uint2*>(iq2s_grid) + i);
        sg_lo[i] = gp.x;
        sg_hi[i] = gp.y;
    }
    __syncthreads();
    const int st_slot = tid >> 7;
    const int st_col = tid & 127;

    for (int eb = o0; eb < o1; eb += 8) {
        const int nt = (eb + 8 <= o1) ? 8 : (o1 - eb);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const int np = (nt + 1) >> 1;

        for (int b = 0; b < K_blk; ++b) {
            for (int p = 0; p < np; ++p) {
                const int t = p * 2 + st_slot;
                if (t < nt) {
                    const __half2* xrow = reinterpret_cast<const __half2*>(xs_f16 + (size_t) (eb + t) * n_embd + b * 256);
                    const __half2 v = __ldg(xrow + st_col);
                    const float2 vf = __half22float2(v);
                    float am = fmaxf(fabsf(vf.x), fabsf(vf.y));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
                    const float inv = (am > 1e-12f) ? __fdividef(1.0f, am) : 0.0f;
                    sxh[t][st_col] = __floats2half2_rn(vf.x * inv, vf.y * inv);
                    if ((st_col & 15) == 0) sxd[t][st_col >> 4] = am;
                }
            }
            __syncthreads();

            if (valid) {
                const uint4 qhsc = __ldg(Wqhsc + (size_t) b * N + row);
                const uint4* Wb = Wqs + (size_t) b * 4 * N + row;
                if (nt > 4) {
                    float bsum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    auto do_sb8 = [&](int sb, uint32_t g32, uint32_t s32, uint32_t qh, uint32_t sc) {
                        const float ls0 = (float) ((sc & 0x0fu) * 2u + 1u);
                        const float ls1 = (float) ((sc >> 4) * 2u + 1u);
#pragma unroll
                        for (int hs = 0; hs < 2; ++hs) {
                            __half2 d[8];
#pragma unroll
                            for (int t = 0; t < 8; ++t) d[t] = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                            for (int kk = 0; kk < 2; ++kk) {
                                const int k0 = hs * 2 + kk;
                                const uint32_t gidx = ((g32 >> (k0 * 8)) & 0xffu) | ((qh << (8 - 2 * k0)) & 0x300u);
                                const uint32_t gx = sg_lo[gidx], gy = sg_hi[gidx];
                                const uint32_t sg = (s32 >> (k0 * 8)) & 0xffu;
                                const __half2 w0 = iq_decode_h2(gx, 0x5150, ((sg & 1u) << 15) | ((sg & 2u) << 30));
                                const __half2 w1 = iq_decode_h2(gx, 0x5352, ((sg & 4u) << 13) | ((sg & 8u) << 28));
                                const __half2 w2 = iq_decode_h2(gy, 0x5150, ((sg & 16u) << 11) | ((sg & 32u) << 26));
                                const __half2 w3 = iq_decode_h2(gy, 0x5352, ((sg & 64u) << 9) | ((sg & 128u) << 24));
                                const int xoff = sb * 16 + k0 * 4;
#pragma unroll
                                for (int t = 0; t < 8; ++t) {
                                    const uint4 xv = *reinterpret_cast<const uint4*>(&sxh[t][xoff]);
                                    __half2 dt = d[t];
                                    dt = __hfma2(w0, *reinterpret_cast<const __half2*>(&xv.x), dt);
                                    dt = __hfma2(w1, *reinterpret_cast<const __half2*>(&xv.y), dt);
                                    dt = __hfma2(w2, *reinterpret_cast<const __half2*>(&xv.z), dt);
                                    dt = __hfma2(w3, *reinterpret_cast<const __half2*>(&xv.w), dt);
                                    d[t] = dt;
                                }
                            }
                            const float ls = hs ? ls1 : ls0;
#pragma unroll
                            for (int t = 0; t < 8; ++t)
                                bsum[t] = fmaf((__low2float(d[t]) + __high2float(d[t])) * ls, sxd[t][sb], bsum[t]);
                        }
                    };
                    uint4 gw = __ldg(Wb), sw = __ldg(Wb + 2 * N);
                    do_sb8(0, gw.x, sw.x, qhsc.x & 0xffu, qhsc.z & 0xffu);
                    do_sb8(1, gw.y, sw.y, (qhsc.x >> 8) & 0xffu, (qhsc.z >> 8) & 0xffu);
                    do_sb8(2, gw.z, sw.z, (qhsc.x >> 16) & 0xffu, (qhsc.z >> 16) & 0xffu);
                    do_sb8(3, gw.w, sw.w, (qhsc.x >> 24) & 0xffu, (qhsc.z >> 24) & 0xffu);
                    gw = __ldg(Wb + N); sw = __ldg(Wb + 3 * N);
                    do_sb8(4, gw.x, sw.x, qhsc.y & 0xffu, qhsc.w & 0xffu);
                    do_sb8(5, gw.y, sw.y, (qhsc.y >> 8) & 0xffu, (qhsc.w >> 8) & 0xffu);
                    do_sb8(6, gw.z, sw.z, (qhsc.y >> 16) & 0xffu, (qhsc.w >> 16) & 0xffu);
                    do_sb8(7, gw.w, sw.w, (qhsc.y >> 24) & 0xffu, (qhsc.w >> 24) & 0xffu);
                    const float db = __half2float(__ldg(Wd + (size_t) b * N + row)) * 0.125f;
#pragma unroll
                    for (int t = 0; t < 8; ++t) acc[t] = fmaf(db, bsum[t], acc[t]);
                } else {
                    float bsum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    auto do_sb4 = [&](int sb, uint32_t g32, uint32_t s32, uint32_t qh, uint32_t sc) {
                        const float ls0 = (float) ((sc & 0x0fu) * 2u + 1u);
                        const float ls1 = (float) ((sc >> 4) * 2u + 1u);
#pragma unroll
                        for (int hs = 0; hs < 2; ++hs) {
                            __half2 d[4];
#pragma unroll
                            for (int t = 0; t < 4; ++t) d[t] = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                            for (int kk = 0; kk < 2; ++kk) {
                                const int k0 = hs * 2 + kk;
                                const uint32_t gidx = ((g32 >> (k0 * 8)) & 0xffu) | ((qh << (8 - 2 * k0)) & 0x300u);
                                const uint32_t gx = sg_lo[gidx], gy = sg_hi[gidx];
                                const uint32_t sg = (s32 >> (k0 * 8)) & 0xffu;
                                const __half2 w0 = iq_decode_h2(gx, 0x5150, ((sg & 1u) << 15) | ((sg & 2u) << 30));
                                const __half2 w1 = iq_decode_h2(gx, 0x5352, ((sg & 4u) << 13) | ((sg & 8u) << 28));
                                const __half2 w2 = iq_decode_h2(gy, 0x5150, ((sg & 16u) << 11) | ((sg & 32u) << 26));
                                const __half2 w3 = iq_decode_h2(gy, 0x5352, ((sg & 64u) << 9) | ((sg & 128u) << 24));
                                const int xoff = sb * 16 + k0 * 4;
#pragma unroll
                                for (int t = 0; t < 4; ++t) {
                                    const uint4 xv = *reinterpret_cast<const uint4*>(&sxh[t][xoff]);
                                    __half2 dt = d[t];
                                    dt = __hfma2(w0, *reinterpret_cast<const __half2*>(&xv.x), dt);
                                    dt = __hfma2(w1, *reinterpret_cast<const __half2*>(&xv.y), dt);
                                    dt = __hfma2(w2, *reinterpret_cast<const __half2*>(&xv.z), dt);
                                    dt = __hfma2(w3, *reinterpret_cast<const __half2*>(&xv.w), dt);
                                    d[t] = dt;
                                }
                            }
                            const float ls = hs ? ls1 : ls0;
#pragma unroll
                            for (int t = 0; t < 4; ++t)
                                bsum[t] = fmaf((__low2float(d[t]) + __high2float(d[t])) * ls, sxd[t][sb], bsum[t]);
                        }
                    };
                    uint4 gw = __ldg(Wb), sw = __ldg(Wb + 2 * N);
                    do_sb4(0, gw.x, sw.x, qhsc.x & 0xffu, qhsc.z & 0xffu);
                    do_sb4(1, gw.y, sw.y, (qhsc.x >> 8) & 0xffu, (qhsc.z >> 8) & 0xffu);
                    do_sb4(2, gw.z, sw.z, (qhsc.x >> 16) & 0xffu, (qhsc.z >> 16) & 0xffu);
                    do_sb4(3, gw.w, sw.w, (qhsc.x >> 24) & 0xffu, (qhsc.z >> 24) & 0xffu);
                    gw = __ldg(Wb + N); sw = __ldg(Wb + 3 * N);
                    do_sb4(4, gw.x, sw.x, qhsc.y & 0xffu, qhsc.w & 0xffu);
                    do_sb4(5, gw.y, sw.y, (qhsc.y >> 8) & 0xffu, (qhsc.w >> 8) & 0xffu);
                    do_sb4(6, gw.z, sw.z, (qhsc.y >> 16) & 0xffu, (qhsc.w >> 16) & 0xffu);
                    do_sb4(7, gw.w, sw.w, (qhsc.y >> 24) & 0xffu, (qhsc.w >> 24) & 0xffu);
                    const float db = __half2float(__ldg(Wd + (size_t) b * N + row)) * 0.125f;
#pragma unroll
                    for (int t = 0; t < 4; ++t) acc[t] = fmaf(db, bsum[t], acc[t]);
                }
            }
            __syncthreads();
        }
        if (valid) {
            for (int t = 0; t < nt; ++t) gu_out[(size_t) (eb + t) * N + row] = acc[t];
        }
    }
}

__global__ void __launch_bounds__(128, 8) p100_prefill_q20_dn_soa_kernel(
    const unsigned long long* __restrict__ exp_ptr, const int32_t* __restrict__ exp_off,
    size_t down_off, int N, int n_ff, const uint16_t* __restrict__ hh_f16, float* __restrict__ dm_out) {
    const int g = (int) blockIdx.y;
    const int o0 = exp_off[g], o1 = exp_off[g + 1];
    if (o0 >= o1) return;

    const int tid = threadIdx.x;
    const int K_blk = n_ff / 64;   // 10 for n_ff=640
    const int r = (int) blockIdx.x * 128 + tid;
    const bool valid = r < N;

    const uint8_t* dn_dev = (const uint8_t*) exp_ptr[g] + down_off;
    const uint4* Wq = reinterpret_cast<const uint4*>(dn_dev);
    const __half* Wd = reinterpret_cast<const __half*>(dn_dev + (size_t) 16 * K_blk * N);

    __shared__ __half2 sxh[4][320];  // 5120 B
    __shared__ float sxd[4][20];     //  320 B -> total 5,440 B (8+ blk/SM!)
    constexpr uint32_t tbl01 = 0x0000BC00u;  // {-1.0h, 0.0h}
    constexpr uint32_t tbl23 = 0x40003C00u;  // {+1.0h, +2.0h}

    for (int eb = o0; eb < o1; eb += 4) {
        const int nt = (eb + 4 <= o1) ? 4 : (o1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 10) {
            const int nb = (b0 + 10 <= K_blk) ? 10 : (K_blk - b0);
            const int nh2 = nb * 32;
            for (int t = 0; t < nt; ++t) {
                const __half2* xrow = reinterpret_cast<const __half2*>(hh_f16 + (size_t) (eb + t) * n_ff + b0 * 64);
                for (int base = 0; base < nh2; base += 128) {
                    const int idx = base + tid;
                    if (idx < nh2) {
                        const __half2 v = __ldg(xrow + idx);
                        const float2 vf = __half22float2(v);
                        float am = fmaxf(fabsf(vf.x), fabsf(vf.y));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
                        const float inv = (am > 1e-12f) ? __fdividef(1.0f, am) : 0.0f;
                        sxh[t][idx] = __floats2half2_rn(vf.x * inv, vf.y * inv);
                        if ((idx & 15) == 0) sxd[t][idx >> 4] = am;
                    }
                }
            }
            for (int t = nt; t < 4; ++t) {
                for (int base = 0; base < nh2; base += 128) {
                    const int idx = base + tid;
                    if (idx < nh2) {
                        sxh[t][idx] = __floats2half2_rn(0.0f, 0.0f);
                        if ((idx & 15) == 0) sxd[t][idx >> 4] = 0.0f;
                    }
                }
            }
            __syncthreads();

            if (valid) {
                for (int db_k = 0; db_k < nb; ++db_k) {
                    const int b = b0 + db_k;
                    const uint4 q4 = __ldg(Wq + (size_t) b * N + r);
                    const float db = __half2float(__ldg(Wd + (size_t) b * N + r));
                    const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                    __half2 s0 = __floats2half2_rn(0.0f, 0.0f), s1 = __floats2half2_rn(0.0f, 0.0f);
                    __half2 s2 = __floats2half2_rn(0.0f, 0.0f), s3 = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const uint32_t w = wv[u];
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const uint32_t qb = (w >> (c * 8)) & 0xffu;
                            const unsigned a0 = (qb & 3u) << 1, b0v = ((qb >> 2) & 3u) << 1;
                            const unsigned a1 = ((qb >> 4) & 3u) << 1, b1v = ((qb >> 6) & 3u) << 1;
                            const uint32_t p0 = __byte_perm(tbl01, tbl23, 0x1010u | a0 | (a0 << 4) | (b0v << 8) | (b0v << 12));
                            const uint32_t p1 = __byte_perm(tbl01, tbl23, 0x1010u | a1 | (a1 << 4) | (b1v << 8) | (b1v << 12));
                            const __half2 h0 = *reinterpret_cast<const __half2*>(&p0);
                            const __half2 h1 = *reinterpret_cast<const __half2*>(&p1);
                            const int xoff = db_k * 32 + u * 8 + c * 2;
                            s0 = __hfma2(h0, sxh[0][xoff + 0], s0); s0 = __hfma2(h1, sxh[0][xoff + 1], s0);
                            s1 = __hfma2(h0, sxh[1][xoff + 0], s1); s1 = __hfma2(h1, sxh[1][xoff + 1], s1);
                            s2 = __hfma2(h0, sxh[2][xoff + 0], s2); s2 = __hfma2(h1, sxh[2][xoff + 1], s2);
                            s3 = __hfma2(h0, sxh[3][xoff + 0], s3); s3 = __hfma2(h1, sxh[3][xoff + 1], s3);
                        }
                    }
                    acc0 = fmaf(db * sxd[0][db_k * 2], __low2float(s0) + __high2float(s0), acc0);
                    acc1 = fmaf(db * sxd[1][db_k * 2], __low2float(s1) + __high2float(s1), acc1);
                    acc2 = fmaf(db * sxd[2][db_k * 2], __low2float(s2) + __high2float(s2), acc2);
                    acc3 = fmaf(db * sxd[3][db_k * 2], __low2float(s3) + __high2float(s3), acc3);
                    s0 = __floats2half2_rn(0.0f, 0.0f); s1 = __floats2half2_rn(0.0f, 0.0f);
                    s2 = __floats2half2_rn(0.0f, 0.0f); s3 = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                    for (int u = 2; u < 4; ++u) {
                        const uint32_t w = wv[u];
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const uint32_t qb = (w >> (c * 8)) & 0xffu;
                            const unsigned a0 = (qb & 3u) << 1, b0v = ((qb >> 2) & 3u) << 1;
                            const unsigned a1 = ((qb >> 4) & 3u) << 1, b1v = ((qb >> 6) & 3u) << 1;
                            const uint32_t p0 = __byte_perm(tbl01, tbl23, 0x1010u | a0 | (a0 << 4) | (b0v << 8) | (b0v << 12));
                            const uint32_t p1 = __byte_perm(tbl01, tbl23, 0x1010u | a1 | (a1 << 4) | (b1v << 8) | (b1v << 12));
                            const __half2 h0 = *reinterpret_cast<const __half2*>(&p0);
                            const __half2 h1 = *reinterpret_cast<const __half2*>(&p1);
                            const int xoff = db_k * 32 + u * 8 + c * 2;
                            s0 = __hfma2(h0, sxh[0][xoff + 0], s0); s0 = __hfma2(h1, sxh[0][xoff + 1], s0);
                            s1 = __hfma2(h0, sxh[1][xoff + 0], s1); s1 = __hfma2(h1, sxh[1][xoff + 1], s1);
                            s2 = __hfma2(h0, sxh[2][xoff + 0], s2); s2 = __hfma2(h1, sxh[2][xoff + 1], s2);
                            s3 = __hfma2(h0, sxh[3][xoff + 0], s3); s3 = __hfma2(h1, sxh[3][xoff + 1], s3);
                        }
                    }
                    acc0 = fmaf(db * sxd[0][db_k * 2 + 1], __low2float(s0) + __high2float(s0), acc0);
                    acc1 = fmaf(db * sxd[1][db_k * 2 + 1], __low2float(s1) + __high2float(s1), acc1);
                    acc2 = fmaf(db * sxd[2][db_k * 2 + 1], __low2float(s2) + __high2float(s2), acc2);
                    acc3 = fmaf(db * sxd[3][db_k * 2 + 1], __low2float(s3) + __high2float(s3), acc3);
                }
            }
            if (b0 + 10 < K_blk) __syncthreads();
        }
        if (valid) {
            dm_out[(size_t) (eb + 0) * N + r] = acc0;
            if (nt > 1) dm_out[(size_t) (eb + 1) * N + r] = acc1;
            if (nt > 2) dm_out[(size_t) (eb + 2) * N + r] = acc2;
            if (nt > 3) dm_out[(size_t) (eb + 3) * N + r] = acc3;
        }
        if (eb + 4 < o1) __syncthreads();
    }
}

__global__ void __launch_bounds__(128, 8) p100_prefill_iq4nl_dn_soa_kernel(
    const unsigned long long* __restrict__ exp_ptr, const int32_t* __restrict__ exp_off,
    size_t down_off, int N, int n_ff, const uint16_t* __restrict__ hh_f16, float* __restrict__ dm_out) {
    const int g = (int) blockIdx.y;
    const int o0 = exp_off[g], o1 = exp_off[g + 1];
    if (o0 >= o1) return;

    const int tid = threadIdx.x;
    const int K_blk = n_ff / 32;   // 20 for n_ff=640
    const int r = (int) blockIdx.x * 128 + tid;
    const bool valid = r < N;

    const uint8_t* dn_dev = (const uint8_t*) exp_ptr[g] + down_off;
    const uint4* Wq = reinterpret_cast<const uint4*>(dn_dev);
    const __half* Wd = reinterpret_cast<const __half*>(dn_dev + (size_t) 16 * K_blk * N);

    __shared__ __half2 sxh[4][320];  // 5120 B
    __shared__ float sxd[4][20];     //  320 B -> total 5,440 B (8+ blk/SM!)
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    constexpr uint32_t T0 = 0x3f2d1801u, T1 = 0x766a5d4fu, T2 = 0xa6998d81u, T3 = 0xf1d9c5b5u;

    for (int eb = o0; eb < o1; eb += 4) {
        const int nt = (eb + 4 <= o1) ? 4 : (o1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = (b0 + 20 <= K_blk) ? 20 : (K_blk - b0);
            const int nh2 = nb * 16;
            for (int t = 0; t < nt; ++t) {
                const __half2* xrow = reinterpret_cast<const __half2*>(hh_f16 + (size_t) (eb + t) * n_ff + b0 * 32);
                for (int base = 0; base < nh2; base += 128) {
                    const int idx = base + tid;
                    if (idx < nh2) {
                        const __half2 v = __ldg(xrow + idx);
                        const float2 vf = __half22float2(v);
                        float am = fmaxf(fabsf(vf.x), fabsf(vf.y));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
                        const float inv = (am > 1e-12f) ? __fdividef(1.0f, am) : 0.0f;
                        sxh[t][idx] = __floats2half2_rn(vf.x * inv, vf.y * inv);
                        if ((idx & 15) == 0) sxd[t][idx >> 4] = am;
                    }
                }
            }
            for (int t = nt; t < 4; ++t) {
                for (int base = 0; base < nh2; base += 128) {
                    const int idx = base + tid;
                    if (idx < nh2) {
                        sxh[t][idx] = __floats2half2_rn(0.0f, 0.0f);
                        if ((idx & 15) == 0) sxd[t][idx >> 4] = 0.0f;
                    }
                }
            }
            __syncthreads();

            if (valid) {
                for (int db_k = 0; db_k < nb; ++db_k) {
                    const int b = b0 + db_k;
                    const uint4 q4 = __ldg(Wq + (size_t) b * N + r);
                    const float db = __half2float(__ldg(Wd + (size_t) b * N + r));
                    const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                    __half2 s0 = __floats2half2_rn(0.0f, 0.0f), s1 = __floats2half2_rn(0.0f, 0.0f);
                    __half2 s2 = __floats2half2_rn(0.0f, 0.0f), s3 = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const uint2 v = iq4nl_lut_reg(wv[u], T0, T1, T2, T3);
                        const uint32_t a0 = __byte_perm(0x64006400u, v.x, 0x3514);
                        const uint32_t a1 = __byte_perm(0x64006400u, v.x, 0x3716);
                        const uint32_t b0v = __byte_perm(0x64006400u, v.y, 0x3514);
                        const uint32_t b1v = __byte_perm(0x64006400u, v.y, 0x3716);
                        const __half2 va0 = __hsub2(*reinterpret_cast<const __half2*>(&a0), c1152);
                        const __half2 va1 = __hsub2(*reinterpret_cast<const __half2*>(&a1), c1152);
                        const __half2 vb0 = __hsub2(*reinterpret_cast<const __half2*>(&b0v), c1152);
                        const __half2 vb1 = __hsub2(*reinterpret_cast<const __half2*>(&b1v), c1152);
                        const int xbase = db_k * 16;
                        s0 = __hfma2(va0, sxh[0][xbase + u * 2 + 0], s0); s0 = __hfma2(va1, sxh[0][xbase + u * 2 + 1], s0);
                        s0 = __hfma2(vb0, sxh[0][xbase + 8 + u * 2 + 0], s0); s0 = __hfma2(vb1, sxh[0][xbase + 8 + u * 2 + 1], s0);
                        s1 = __hfma2(va0, sxh[1][xbase + u * 2 + 0], s1); s1 = __hfma2(va1, sxh[1][xbase + u * 2 + 1], s1);
                        s1 = __hfma2(vb0, sxh[1][xbase + 8 + u * 2 + 0], s1); s1 = __hfma2(vb1, sxh[1][xbase + 8 + u * 2 + 1], s1);
                        s2 = __hfma2(va0, sxh[2][xbase + u * 2 + 0], s2); s2 = __hfma2(va1, sxh[2][xbase + u * 2 + 1], s2);
                        s2 = __hfma2(vb0, sxh[2][xbase + 8 + u * 2 + 0], s2); s2 = __hfma2(vb1, sxh[2][xbase + 8 + u * 2 + 1], s2);
                        s3 = __hfma2(va0, sxh[3][xbase + u * 2 + 0], s3); s3 = __hfma2(va1, sxh[3][xbase + u * 2 + 1], s3);
                        s3 = __hfma2(vb0, sxh[3][xbase + 8 + u * 2 + 0], s3); s3 = __hfma2(vb1, sxh[3][xbase + 8 + u * 2 + 1], s3);
                    }
                    acc0 = fmaf(db * sxd[0][db_k], __low2float(s0) + __high2float(s0), acc0);
                    acc1 = fmaf(db * sxd[1][db_k], __low2float(s1) + __high2float(s1), acc1);
                    acc2 = fmaf(db * sxd[2][db_k], __low2float(s2) + __high2float(s2), acc2);
                    acc3 = fmaf(db * sxd[3][db_k], __low2float(s3) + __high2float(s3), acc3);
                }
            }
            if (b0 + 20 < K_blk) __syncthreads();
        }
        if (valid) {
            dm_out[(size_t) (eb + 0) * N + r] = acc0;
            if (nt > 1) dm_out[(size_t) (eb + 1) * N + r] = acc1;
            if (nt > 2) dm_out[(size_t) (eb + 2) * N + r] = acc2;
            if (nt > 3) dm_out[(size_t) (eb + 3) * N + r] = acc3;
        }
        if (eb + 4 < o1) __syncthreads();
    }
}

// Fused grouped Q5_0 down projection for prefill (Layer 4 on Stage 0: eliminates 512 dequants + 512 cuBLAS calls = -1.33s on 3.8K prefill!).
__global__ void __launch_bounds__(128, 6) p100_prefill_q50_dn_kernel(
    const unsigned long long* __restrict__ exp_ptr, const int32_t* __restrict__ exp_off,
    size_t down_off, int N, int n_ff, const uint16_t* __restrict__ hh_f16, float* __restrict__ dm_out) {
    const int g = (int) blockIdx.y;
    const int o0 = exp_off[g], o1 = exp_off[g + 1];
    if (o0 >= o1) return;

    const int tid = threadIdx.x;
    const int K_blk = n_ff / 32;   // 20 for n_ff=640
    const int r = (int) blockIdx.x * 128 + tid;
    const bool valid = r < N;

    const block_q5_0* row_blk = valid
        ? (reinterpret_cast<const block_q5_0*>((const uint8_t*) exp_ptr[g] + down_off) + (size_t) r * K_blk)
        : nullptr;

    __shared__ __half2 sxh[4][320];  // 5120 B
    __shared__ float sxd[4][20];     //  320 B -> total 5,440 B
    const __half2 c1040 = __floats2half2_rn(1040.0f, 1040.0f);

    for (int eb = o0; eb < o1; eb += 4) {
        const int nt = (eb + 4 <= o1) ? 4 : (o1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = (b0 + 20 <= K_blk) ? 20 : (K_blk - b0);
            const int nh2 = nb * 16;
            for (int t = 0; t < nt; ++t) {
                const __half2* xrow = reinterpret_cast<const __half2*>(hh_f16 + (size_t) (eb + t) * n_ff + b0 * 32);
                for (int base = 0; base < nh2; base += 128) {
                    const int idx = base + tid;
                    if (idx < nh2) {
                        const __half2 v = __ldg(xrow + idx);
                        const float2 vf = __half22float2(v);
                        float am = fmaxf(fabsf(vf.x), fabsf(vf.y));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
                        const float inv = (am > 1e-12f) ? __fdividef(1.0f, am) : 0.0f;
                        sxh[t][idx] = __floats2half2_rn(vf.x * inv, vf.y * inv);
                        if ((idx & 15) == 0) sxd[t][idx >> 4] = am;
                    }
                }
            }
            for (int t = nt; t < 4; ++t) {
                for (int base = 0; base < nh2; base += 128) {
                    const int idx = base + tid;
                    if (idx < nh2) {
                        sxh[t][idx] = __floats2half2_rn(0.0f, 0.0f);
                        if ((idx & 15) == 0) sxd[t][idx >> 4] = 0.0f;
                    }
                }
            }
            __syncthreads();

            if (valid) {
                for (int db_k = 0; db_k < nb; ++db_k) {
                    const int b = b0 + db_k;
                    const uint16_t* bp = reinterpret_cast<const uint16_t*>(row_blk + b);
                    const float db = __half2float(__ushort_as_half(__ldg(bp + 0)));
                    const uint32_t qh = (uint32_t) __ldg(bp + 1) | ((uint32_t) __ldg(bp + 2) << 16);
                    __half2 s0 = __floats2half2_rn(0.0f, 0.0f), s1 = s0, s2 = s0, s3 = s0;
                    const int xbase = db_k * 16;
#pragma unroll
                    for (int p = 0; p < 8; ++p) {
                        const uint32_t q16 = (uint32_t) __ldg(bp + 3 + p);
                        const uint32_t lo0 = (q16 & 0x0fu) | (((qh >> (2 * p)) & 1u) << 4);
                        const uint32_t lo1 = ((q16 >> 8) & 0x0fu) | (((qh >> (2 * p + 1)) & 1u) << 4);
                        const uint32_t hi0 = ((q16 >> 4) & 0x0fu) | (((qh >> (16 + 2 * p)) & 1u) << 4);
                        const uint32_t hi1 = ((q16 >> 12) & 0x0fu) | (((qh >> (16 + 2 * p + 1)) & 1u) << 4);
                        const uint32_t hl = (0x6400u | lo0) | ((0x6400u | lo1) << 16);
                        const uint32_t hh = (0x6400u | hi0) | ((0x6400u | hi1) << 16);
                        const __half2 vl = __hsub2(*reinterpret_cast<const __half2*>(&hl), c1040);
                        const __half2 vh = __hsub2(*reinterpret_cast<const __half2*>(&hh), c1040);
                        s0 = __hfma2(vl, sxh[0][xbase + p], s0); s0 = __hfma2(vh, sxh[0][xbase + 8 + p], s0);
                        s1 = __hfma2(vl, sxh[1][xbase + p], s1); s1 = __hfma2(vh, sxh[1][xbase + 8 + p], s1);
                        s2 = __hfma2(vl, sxh[2][xbase + p], s2); s2 = __hfma2(vh, sxh[2][xbase + 8 + p], s2);
                        s3 = __hfma2(vl, sxh[3][xbase + p], s3); s3 = __hfma2(vh, sxh[3][xbase + 8 + p], s3);
                    }
                    acc0 = fmaf(db * sxd[0][db_k], __low2float(s0) + __high2float(s0), acc0);
                    acc1 = fmaf(db * sxd[1][db_k], __low2float(s1) + __high2float(s1), acc1);
                    acc2 = fmaf(db * sxd[2][db_k], __low2float(s2) + __high2float(s2), acc2);
                    acc3 = fmaf(db * sxd[3][db_k], __low2float(s3) + __high2float(s3), acc3);
                }
            }
            if (b0 + 20 < K_blk) __syncthreads();
        }
        if (valid) {
            dm_out[(size_t) (eb + 0) * N + r] = acc0;
            if (nt > 1) dm_out[(size_t) (eb + 1) * N + r] = acc1;
            if (nt > 2) dm_out[(size_t) (eb + 2) * N + r] = acc2;
            if (nt > 3) dm_out[(size_t) (eb + 3) * N + r] = acc3;
        }
        if (eb + 4 < o1) __syncthreads();
    }
}

// GP100 Fused Grouped SoA HFMA2 Prefill for Q4_K Gate+Up (TM=4 tokens/pass, 256 threads/block).
__global__ void __launch_bounds__(256, 2) p100_prefill_q4k_gu_soa_kernel(
    const unsigned long long* __restrict__ exp_ptr, const int32_t* __restrict__ exp_off,
    int n_ff, int n_embd, const uint16_t* __restrict__ xs_f16, float* __restrict__ gu_out) {
    const int g = (int) blockIdx.y;
    const int o0 = exp_off[g], o1 = exp_off[g + 1];
    if (o0 >= o1) return;

    const int tid = threadIdx.x;
    const int N = 2 * n_ff;
    const int K_blk = n_embd / 256;
    const int row = (int) blockIdx.x * 256 + tid;
    const bool valid = row < N;

    const uint8_t* blob = (const uint8_t*) exp_ptr[g];
    const uint4* Wqs = reinterpret_cast<const uint4*>(blob);
    const uint4* Wscdm = reinterpret_cast<const uint4*>(blob + (size_t) 128 * K_blk * N);

    __shared__ __half2 sxh[4][128];
    __shared__ float sxd[4][8];
    __shared__ float sxs[4][8];
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    const int st_slot = tid >> 7;
    const int st_col = tid & 127;

    for (int eb = o0; eb < o1; eb += 4) {
        const int nt = (eb + 4 <= o1) ? 4 : (o1 - eb);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int b = 0; b < K_blk; ++b) {
            for (int p = 0; p < 2; ++p) {
                const int t = p * 2 + st_slot;
                if (t < nt) {
                    const __half2* xrow = reinterpret_cast<const __half2*>(xs_f16 + (size_t) (eb + t) * n_embd + b * 256);
                    const float2 vf = __half22float2(__ldg(xrow + st_col));
                    float am = fmaxf(fabsf(vf.x), fabsf(vf.y));
                    float sm = vf.x + vf.y;
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8)); sm += __shfl_xor_sync(0xffffffffu, sm, 8);
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4)); sm += __shfl_xor_sync(0xffffffffu, sm, 4);
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2)); sm += __shfl_xor_sync(0xffffffffu, sm, 2);
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1)); sm += __shfl_xor_sync(0xffffffffu, sm, 1);
                    const float inv = (am > 1e-12f) ? __fdividef(1.0f, am) : 0.0f;
                    sxh[t][st_col] = __floats2half2_rn(vf.x * inv, vf.y * inv);
                    if ((st_col & 15) == 0) { sxd[t][st_col >> 4] = am; sxs[t][st_col >> 4] = sm; }
                } else {
                    sxh[t][st_col] = __floats2half2_rn(0.0f, 0.0f);
                    if ((st_col & 15) == 0) { sxd[t][st_col >> 4] = 0.0f; sxs[t][st_col >> 4] = 0.0f; }
                }
            }
            __syncthreads();
            if (valid) {
                const uint4 scdm = __ldg(Wscdm + (size_t) b * N + row);
                const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&scdm.w));
                const uint32_t s0 = scdm.x, s1 = scdm.y, s2 = scdm.z;
                const uint32_t sc_w[2] = {s0 & 0x3f3f3f3fu, (s2 & 0x0f0f0f0fu) | ((s0 >> 2) & 0x30303030u)};
                const uint32_t m_w[2]  = {s1 & 0x3f3f3f3fu, ((s2 >> 4) & 0x0f0f0f0fu) | ((s1 >> 2) & 0x30303030u)};
                float sumd[4] = {0.0f, 0.0f, 0.0f, 0.0f}, summ[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4 qa = __ldg(Wqs + ((size_t) b * 8 + c * 2 + 0) * N + row);
                    const uint4 qb = __ldg(Wqs + ((size_t) b * 8 + c * 2 + 1) * N + row);
                    const uint32_t wv[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
                    const int sb0 = c * 2, sb1 = c * 2 + 1;
                    const float sc0 = (float) ((sc_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                    const float sc1 = (float) ((sc_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                    const float mn0 = (float) ((m_w[sb0 >> 2] >> ((sb0 & 3) * 8)) & 0xffu);
                    const float mn1 = (float) ((m_w[sb1 >> 2] >> ((sb1 & 3) * 8)) & 0xffu);
                    __half2 pa[4], pb[4];
#pragma unroll
                    for (int t = 0; t < 4; ++t) { pa[t] = __float2half2_rn(0.0f); pb[t] = __float2half2_rn(0.0f); }
#pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        const uint32_t w = wv[u];
                        const uint32_t lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu;
                        const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                        const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                        const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                        const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                        const __half2 wla = __hsub2(*reinterpret_cast<const __half2*>(&la), c1024);
                        const __half2 wlb = __hsub2(*reinterpret_cast<const __half2*>(&lb), c1024);
                        const __half2 wha = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                        const __half2 whb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            pa[t] = __hfma2(wla, sxh[t][sb0 * 16 + u * 2 + 0], pa[t]);
                            pa[t] = __hfma2(wlb, sxh[t][sb0 * 16 + u * 2 + 1], pa[t]);
                            pb[t] = __hfma2(wha, sxh[t][sb1 * 16 + u * 2 + 0], pb[t]);
                            pb[t] = __hfma2(whb, sxh[t][sb1 * 16 + u * 2 + 1], pb[t]);
                        }
                    }
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const float2 fa = __half22float2(pa[t]), fb = __half22float2(pb[t]);
                        sumd[t] = fmaf(sc0 * sxd[t][sb0], fa.x + fa.y, sumd[t]);
                        sumd[t] = fmaf(sc1 * sxd[t][sb1], fb.x + fb.y, sumd[t]);
                        summ[t] = fmaf(mn0, sxs[t][sb0], summ[t]);
                        summ[t] = fmaf(mn1, sxs[t][sb1], summ[t]);
                    }
                }
#pragma unroll
                for (int t = 0; t < 4; ++t) acc[t] = fmaf(dm.x, sumd[t], fmaf(-dm.y, summ[t], acc[t]));
            }
            __syncthreads();
        }
        if (valid) {
            for (int t = 0; t < nt; ++t) gu_out[(size_t) (eb + t) * N + row] = acc[t];
        }
    }
}

// GP100 Fused Grouped SoA HFMA2 Prefill for Q5_1 Down (TM=4, SWAR 0x00204081u).
__global__ void __launch_bounds__(128, 6) p100_prefill_q51_dn_soa_kernel(
    const unsigned long long* __restrict__ exp_ptr, const int32_t* __restrict__ exp_off,
    size_t down_off, int N, int n_ff, const uint16_t* __restrict__ hh_f16, float* __restrict__ dm_out) {
    const int g = (int) blockIdx.y;
    const int o0 = exp_off[g], o1 = exp_off[g + 1];
    if (o0 >= o1) return;

    const int tid = threadIdx.x;
    const int K_blk = n_ff / 32;
    const int r = (int) blockIdx.x * 128 + tid;
    const bool valid = r < N;

    const uint8_t* dn_dev = (const uint8_t*) exp_ptr[g] + down_off;
    const uint4* Wqs = reinterpret_cast<const uint4*>(dn_dev);
    const uint2* Wdmqh = reinterpret_cast<const uint2*>(dn_dev + (size_t) 16 * K_blk * N);

    __shared__ __half2 sxh[4][320];
    __shared__ float sxd[4][20];
    __shared__ float sxs[4][20];
    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);

    for (int eb = o0; eb < o1; eb += 4) {
        const int nt = (eb + 4 <= o1) ? 4 : (o1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = (b0 + 20 <= K_blk) ? 20 : (K_blk - b0);
            const int nh2 = nb * 16;
            for (int t = 0; t < nt; ++t) {
                const __half2* xrow = reinterpret_cast<const __half2*>(hh_f16 + (size_t) (eb + t) * n_ff + b0 * 32);
                for (int base = 0; base < nh2; base += 128) {
                    const int idx = base + tid;
                    if (idx < nh2) {
                        const float2 vf = __half22float2(__ldg(xrow + idx));
                        float am = fmaxf(fabsf(vf.x), fabsf(vf.y));
                        float sm = vf.x + vf.y;
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8)); sm += __shfl_xor_sync(0xffffffffu, sm, 8);
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4)); sm += __shfl_xor_sync(0xffffffffu, sm, 4);
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2)); sm += __shfl_xor_sync(0xffffffffu, sm, 2);
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1)); sm += __shfl_xor_sync(0xffffffffu, sm, 1);
                        const float inv = (am > 1e-12f) ? __fdividef(1.0f, am) : 0.0f;
                        sxh[t][idx] = __floats2half2_rn(vf.x * inv, vf.y * inv);
                        if ((idx & 15) == 0) { sxd[t][idx >> 4] = am; sxs[t][idx >> 4] = sm; }
                    }
                }
            }
            for (int t = nt; t < 4; ++t) {
                for (int base = 0; base < nh2; base += 128) {
                    const int idx = base + tid;
                    if (idx < nh2) {
                        sxh[t][idx] = __floats2half2_rn(0.0f, 0.0f);
                        if ((idx & 15) == 0) { sxd[t][idx >> 4] = 0.0f; sxs[t][idx >> 4] = 0.0f; }
                    }
                }
            }
            __syncthreads();
            if (valid) {
                for (int db_k = 0; db_k < nb; ++db_k) {
                    const int b = b0 + db_k;
                    const uint4 q4 = __ldg(Wqs + (size_t) b * N + r);
                    const uint2 dmqh = __ldg(Wdmqh + (size_t) b * N + r);
                    const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&dmqh.x));
                    const uint32_t qh = dmqh.y;
                    const uint32_t wv[4] = {q4.x, q4.y, q4.z, q4.w};
                    __half2 s0 = __float2half2_rn(0.0f), s1 = s0, s2 = s0, s3 = s0;
                    const int xbase = db_k * 16;
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const uint32_t w = wv[u];
                        const uint32_t lo = (w & 0x0f0f0f0fu) | ((qh << (4 - u)) & 0x10101010u);
                        const uint32_t hi = ((w >> 4) & 0x0f0f0f0fu) | ((qh >> u) & 0x10101010u);
                        const uint32_t la = __byte_perm(lo, 0x64006400u, 0x5150);
                        const uint32_t lb = __byte_perm(lo, 0x64006400u, 0x5352);
                        const uint32_t ha = __byte_perm(hi, 0x64006400u, 0x5150);
                        const uint32_t hb = __byte_perm(hi, 0x64006400u, 0x5352);
                        const __half2 wla = __hsub2(*reinterpret_cast<const __half2*>(&la), c1024);
                        const __half2 wlb = __hsub2(*reinterpret_cast<const __half2*>(&lb), c1024);
                        const __half2 wha = __hsub2(*reinterpret_cast<const __half2*>(&ha), c1024);
                        const __half2 whb = __hsub2(*reinterpret_cast<const __half2*>(&hb), c1024);
                        s0 = __hfma2(wla, sxh[0][xbase + u * 2 + 0], s0); s0 = __hfma2(wlb, sxh[0][xbase + u * 2 + 1], s0);
                        s0 = __hfma2(wha, sxh[0][xbase + 8 + u * 2 + 0], s0); s0 = __hfma2(whb, sxh[0][xbase + 8 + u * 2 + 1], s0);
                        s1 = __hfma2(wla, sxh[1][xbase + u * 2 + 0], s1); s1 = __hfma2(wlb, sxh[1][xbase + u * 2 + 1], s1);
                        s1 = __hfma2(wha, sxh[1][xbase + 8 + u * 2 + 0], s1); s1 = __hfma2(whb, sxh[1][xbase + 8 + u * 2 + 1], s1);
                        s2 = __hfma2(wla, sxh[2][xbase + u * 2 + 0], s2); s2 = __hfma2(wlb, sxh[2][xbase + u * 2 + 1], s2);
                        s2 = __hfma2(wha, sxh[2][xbase + 8 + u * 2 + 0], s2); s2 = __hfma2(whb, sxh[2][xbase + 8 + u * 2 + 1], s2);
                        s3 = __hfma2(wla, sxh[3][xbase + u * 2 + 0], s3); s3 = __hfma2(wlb, sxh[3][xbase + u * 2 + 1], s3);
                        s3 = __hfma2(wha, sxh[3][xbase + 8 + u * 2 + 0], s3); s3 = __hfma2(whb, sxh[3][xbase + 8 + u * 2 + 1], s3);
                    }
                    acc0 = fmaf(dm.y, sxs[0][db_k], fmaf(dm.x * sxd[0][db_k], __low2float(s0) + __high2float(s0), acc0));
                    acc1 = fmaf(dm.y, sxs[1][db_k], fmaf(dm.x * sxd[1][db_k], __low2float(s1) + __high2float(s1), acc1));
                    acc2 = fmaf(dm.y, sxs[2][db_k], fmaf(dm.x * sxd[2][db_k], __low2float(s2) + __high2float(s2), acc2));
                    acc3 = fmaf(dm.y, sxs[3][db_k], fmaf(dm.x * sxd[3][db_k], __low2float(s3) + __high2float(s3), acc3));
                }
            }
            if (b0 + 20 < K_blk) __syncthreads();
        }
        if (valid) {
            dm_out[(size_t) (eb + 0) * N + r] = acc0;
            if (nt > 1) dm_out[(size_t) (eb + 1) * N + r] = acc1;
            if (nt > 2) dm_out[(size_t) (eb + 2) * N + r] = acc2;
            if (nt > 3) dm_out[(size_t) (eb + 3) * N + r] = acc3;
        }
        if (eb + 4 < o1) __syncthreads();
    }
}

// GP100 Fused Grouped SoA HFMA2 Prefill for Q8_0 Down (TM=4).
__global__ void __launch_bounds__(128, 8) p100_prefill_q80_dn_soa_kernel(
    const unsigned long long* __restrict__ exp_ptr, const int32_t* __restrict__ exp_off,
    size_t down_off, int N, int n_ff, const uint16_t* __restrict__ hh_f16, float* __restrict__ dm_out) {
    const int g = (int) blockIdx.y;
    const int o0 = exp_off[g], o1 = exp_off[g + 1];
    if (o0 >= o1) return;

    const int tid = threadIdx.x;
    const int K_blk = n_ff / 32;
    const int r = (int) blockIdx.x * 128 + tid;
    const bool valid = r < N;

    const uint8_t* dn_dev = (const uint8_t*) exp_ptr[g] + down_off;
    const uint4* Wqs = reinterpret_cast<const uint4*>(dn_dev);
    const __half* Wd = reinterpret_cast<const __half*>(dn_dev + (size_t) 32 * K_blk * N);

    __shared__ __half2 sxh[4][320];
    __shared__ float sxd[4][20];
    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);

    for (int eb = o0; eb < o1; eb += 4) {
        const int nt = (eb + 4 <= o1) ? 4 : (o1 - eb);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int b0 = 0; b0 < K_blk; b0 += 20) {
            const int nb = (b0 + 20 <= K_blk) ? 20 : (K_blk - b0);
            const int nh2 = nb * 16;
            for (int t = 0; t < nt; ++t) {
                const __half2* xrow = reinterpret_cast<const __half2*>(hh_f16 + (size_t) (eb + t) * n_ff + b0 * 32);
                for (int base = 0; base < nh2; base += 128) {
                    const int idx = base + tid;
                    if (idx < nh2) {
                        const float2 vf = __half22float2(__ldg(xrow + idx));
                        float am = fmaxf(fabsf(vf.x), fabsf(vf.y));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
                        const float inv = (am > 1e-12f) ? __fdividef(1.0f, am) : 0.0f;
                        sxh[t][idx] = __floats2half2_rn(vf.x * inv, vf.y * inv);
                        if ((idx & 15) == 0) sxd[t][idx >> 4] = am;
                    }
                }
            }
            for (int t = nt; t < 4; ++t) {
                for (int base = 0; base < nh2; base += 128) {
                    const int idx = base + tid;
                    if (idx < nh2) {
                        sxh[t][idx] = __floats2half2_rn(0.0f, 0.0f);
                        if ((idx & 15) == 0) sxd[t][idx >> 4] = 0.0f;
                    }
                }
            }
            __syncthreads();
            if (valid) {
                for (int db_k = 0; db_k < nb; ++db_k) {
                    const int b = b0 + db_k;
                    const float db = __half2float(__ldg(Wd + (size_t) b * N + r));
                    __half2 s0 = __float2half2_rn(0.0f), s1 = s0, s2 = s0, s3 = s0;
                    const int xbase = db_k * 16;
#pragma unroll
                    for (int hf = 0; hf < 2; ++hf) {
                        const uint4 qv = __ldg(Wqs + ((size_t) b * 2 + hf) * N + r);
                        const uint32_t ws[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const uint32_t vx = ws[u];
                            const uint32_t a = __byte_perm(vx, 0x64006400u, 0x5150);
                            const uint32_t bv = __byte_perm(vx, 0x64006400u, 0x5352);
                            const __half2 wa = __hsub2(*reinterpret_cast<const __half2*>(&a), c1152);
                            const __half2 wb = __hsub2(*reinterpret_cast<const __half2*>(&bv), c1152);
                            const int xo = xbase + hf * 8 + u * 2;
                            s0 = __hfma2(wa, sxh[0][xo + 0], s0); s0 = __hfma2(wb, sxh[0][xo + 1], s0);
                            s1 = __hfma2(wa, sxh[1][xo + 0], s1); s1 = __hfma2(wb, sxh[1][xo + 1], s1);
                            s2 = __hfma2(wa, sxh[2][xo + 0], s2); s2 = __hfma2(wb, sxh[2][xo + 1], s2);
                            s3 = __hfma2(wa, sxh[3][xo + 0], s3); s3 = __hfma2(wb, sxh[3][xo + 1], s3);
                        }
                    }
                    acc0 = fmaf(db * sxd[0][db_k], __low2float(s0) + __high2float(s0), acc0);
                    acc1 = fmaf(db * sxd[1][db_k], __low2float(s1) + __high2float(s1), acc1);
                    acc2 = fmaf(db * sxd[2][db_k], __low2float(s2) + __high2float(s2), acc2);
                    acc3 = fmaf(db * sxd[3][db_k], __low2float(s3) + __high2float(s3), acc3);
                }
            }
            if (b0 + 20 < K_blk) __syncthreads();
        }
        if (valid) {
            dm_out[(size_t) (eb + 0) * N + r] = acc0;
            if (nt > 1) dm_out[(size_t) (eb + 1) * N + r] = acc1;
            if (nt > 2) dm_out[(size_t) (eb + 2) * N + r] = acc2;
            if (nt > 3) dm_out[(size_t) (eb + 3) * N + r] = acc3;
        }
        if (eb + 4 < o1) __syncthreads();
    }
}

void iq_prefill_gu_soa_grouped(int gu_type, const unsigned long long* exp_ptr, const int32_t* exp_off, int n_exp,
                               int64_t n_ff, int64_t n_embd, const uint16_t* xs_f16, float* gu_out, void* stream) {
    if (n_exp <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    const dim3 grid((unsigned) ((2 * n_ff + 255) / 256), (unsigned) n_exp);
    if (gu_type == 12) {
        p100_prefill_q4k_gu_soa_kernel<<<grid, 256, 0, s>>>(exp_ptr, exp_off, (int) n_ff, (int) n_embd, xs_f16, gu_out);
    } else if (gu_type == 16) {
        p100_prefill_iq2xxs_gu_soa_kernel<<<grid, 256, 0, s>>>(exp_ptr, exp_off, (int) n_ff, (int) n_embd, xs_f16, gu_out);
    } else if (gu_type == 22) {
        p100_prefill_iq2s_gu_soa_kernel<<<grid, 256, 0, s>>>(exp_ptr, exp_off, (int) n_ff, (int) n_embd, xs_f16, gu_out);
    } else {
        std::fprintf(stderr, "iq_prefill_gu_soa_grouped: unsupported type %d\n", gu_type);
        std::exit(1);
    }
    check("iq_prefill_gu_soa_grouped");
}

void iq_prefill_dn_soa_grouped(int d_type, const unsigned long long* exp_ptr, const int32_t* exp_off, int n_exp,
                               int64_t n_embd, int64_t n_ff, const uint16_t* hh_f16, float* dm_out, void* stream) {
    if (n_exp <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    const dim3 grid((unsigned) ((n_embd + 127) / 128), (unsigned) n_exp);
    // exp_ptr already points to the down matrix start (blob_dev + f.down_off) when down_off=0!
    if (d_type == 42) {
        p100_prefill_q20_dn_soa_kernel<<<grid, 128, 0, s>>>(exp_ptr, exp_off, 0, (int) n_embd, (int) n_ff, hh_f16, dm_out);
    } else if (d_type == 20) {
        p100_prefill_iq4nl_dn_soa_kernel<<<grid, 128, 0, s>>>(exp_ptr, exp_off, 0, (int) n_embd, (int) n_ff, hh_f16, dm_out);
    } else if (d_type == 6) {
        p100_prefill_q50_dn_kernel<<<grid, 128, 0, s>>>(exp_ptr, exp_off, 0, (int) n_embd, (int) n_ff, hh_f16, dm_out);
    } else if (d_type == 7) {
        p100_prefill_q51_dn_soa_kernel<<<grid, 128, 0, s>>>(exp_ptr, exp_off, 0, (int) n_embd, (int) n_ff, hh_f16, dm_out);
    } else if (d_type == 8) {
        p100_prefill_q80_dn_soa_kernel<<<grid, 128, 0, s>>>(exp_ptr, exp_off, 0, (int) n_embd, (int) n_ff, hh_f16, dm_out);
    } else {
        std::fprintf(stderr, "iq_prefill_dn_soa_grouped: unsupported type %d\n", d_type);
        std::exit(1);
    }
    check("iq_prefill_dn_soa_grouped");
}

// ========================================================================================================
// GP100 128x64 HFMA2 Tiled Grouped Prefill MoE Kernels (VRAM SoA & RAM-staged AoS GGUF, ZERO cuBLAS!)
// ========================================================================================================
namespace {

constexpr int PT_BM = 128, PT_BN = 64, PT_BK = 16, PT_WPAD = PT_BM + 4, PT_NTHR = 256;

__device__ __forceinline__ __half2 pt_h2_u32(uint32_t u) { return *reinterpret_cast<const __half2*>(&u); }
__device__ __forceinline__ uint32_t pt_u32_h2(__half2 h) { return *reinterpret_cast<const uint32_t*>(&h); }

template <int BUF>
__device__ __forceinline__ void pt_fma_stage_h(const uint4 (*Ws)[PT_BK / 2][PT_WPAD / 4],
                                               const uint4 (*Xs)[PT_BK / 2][PT_BN / 4],
                                               int tx, int ty, __half2 (&acc)[8][4]) {
#pragma unroll
    for (int kp = 0; kp < PT_BK / 2; ++kp) {
        const uint4 wa = Ws[BUF][kp][tx];
        const uint4 wb = Ws[BUF][kp][PT_BM / 8 + tx];
        const uint4 xv = Xs[BUF][kp][ty ^ ((kp >> 1) << 1)];
        const uint32_t wu[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
        const uint32_t xu[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = __hfma2(pt_h2_u32(wu[i]), pt_h2_u32(xu[j]), acc[i][j]);
    }
}

__device__ __forceinline__ void pt_decode_q4k_8(uint2 q8, bool hi_nib, __half2 d_sc_h2, __half2 neg_d_mn_h2,
                                                __half2 c1024, uint32_t (&out)[4]) {
    const uint32_t w0 = hi_nib ? ((q8.x >> 4) & 0x0f0f0f0fu) : (q8.x & 0x0f0f0f0fu);
    const uint32_t w1 = hi_nib ? ((q8.y >> 4) & 0x0f0f0f0fu) : (q8.y & 0x0f0f0f0fu);
    const uint32_t p0 = __byte_perm(w0, 0x64006400u, 0x5150);
    const uint32_t p1 = __byte_perm(w0, 0x64006400u, 0x5352);
    const uint32_t p2 = __byte_perm(w1, 0x64006400u, 0x5150);
    const uint32_t p3 = __byte_perm(w1, 0x64006400u, 0x5352);
    out[0] = pt_u32_h2(__hfma2(__hsub2(pt_h2_u32(p0), c1024), d_sc_h2, neg_d_mn_h2));
    out[1] = pt_u32_h2(__hfma2(__hsub2(pt_h2_u32(p1), c1024), d_sc_h2, neg_d_mn_h2));
    out[2] = pt_u32_h2(__hfma2(__hsub2(pt_h2_u32(p2), c1024), d_sc_h2, neg_d_mn_h2));
    out[3] = pt_u32_h2(__hfma2(__hsub2(pt_h2_u32(p3), c1024), d_sc_h2, neg_d_mn_h2));
}

template <bool IS_SOA>
__global__ void __launch_bounds__(PT_NTHR, 2) p100_prefill_q4k_gu_tiled_kernel(
    const unsigned long long* __restrict__ exp_ptr, const IqPrefillTile* __restrict__ tiles,
    int n_ff, int n_embd, size_t up_off, const uint16_t* __restrict__ xs_f16, float* __restrict__ gu_out) {
    const IqPrefillTile td = tiles[blockIdx.y];
    const int ntok = td.ntok;
    if (ntok <= 0) return;
    const int o0 = td.tok0;
    const int mt = (int) blockIdx.x;
    const int N = 2 * n_ff;
    const int K_blk = n_embd / 256;

    __shared__ __align__(16) uint32_t Ws[2][PT_BK / 2][PT_WPAD];
    __shared__ __align__(16) uint32_t Xs[2][PT_BK / 2][PT_BN];

    const int tid = threadIdx.x;
    const int wrow = tid >> 1, hh = tid & 1;
    const int row = min(mt * PT_BM + wrow, N - 1);
    const int xr = tid >> 2, xc = tid & 3;
    const bool xok = xr < ntok;
    const __half* xp_base = reinterpret_cast<const __half*>(xs_f16 + (size_t) (o0 + min(xr, ntok - 1)) * n_embd + 4 * xc);
    const int xs = xr ^ (xc << 3);
    const int tx = tid & 15, ty = tid >> 4;
    const bool active = (tid >> 5) * 8 < ntok;

    const uint8_t* blob = (const uint8_t*) exp_ptr[td.exp];
    const uint4* Wqs = IS_SOA ? reinterpret_cast<const uint4*>(blob) : nullptr;
    const uint4* Wscdm = IS_SOA ? reinterpret_cast<const uint4*>(blob + (size_t) 128 * K_blk * N) : nullptr;
    const uint8_t* aos_row = IS_SOA ? nullptr : ((row < n_ff ? blob : (blob + up_off)) +
                                                 (size_t) (row < n_ff ? row : (row - n_ff)) * (size_t) K_blk * 144u);

    float acc32[8][4];
    __half2 acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc32[i][j] = 0.0f;
            acc[i][j] = __float2half2_rn(0.0f);
        }

    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    const uint2 zero2 = make_uint2(0u, 0u);
    const auto Wv = reinterpret_cast<const uint4 (*)[PT_BK / 2][PT_WPAD / 4]>(&Ws[0][0][0]);
    const auto Xv = reinterpret_cast<const uint4 (*)[PT_BK / 2][PT_BN / 4]>(&Xs[0][0][0]);

    const auto put_x = [&](int buf, int k_off) {
        const uint2 v = xok ? __ldg(reinterpret_cast<const uint2*>(xp_base + k_off)) : zero2;
        Xs[buf][2 * xc + 0][xs] = v.x;
        Xs[buf][2 * xc + 1][xs] = v.y;
    };
    const auto put_w = [&](int buf, const uint32_t (&wv)[4]) {
        Ws[buf][4 * hh + 0][wrow] = wv[0];
        Ws[buf][4 * hh + 1][wrow] = wv[1];
        Ws[buf][4 * hh + 2][wrow] = wv[2];
        Ws[buf][4 * hh + 3][wrow] = wv[3];
    };
    const auto load_q8 = [&](int b, int u) -> uint2 {
        if (IS_SOA) {
            const uint2* p = reinterpret_cast<const uint2*>(Wqs + ((size_t) b * 8 + u) * N + row);
            return __ldg(p + hh);
        } else {
            const uint2* p = reinterpret_cast<const uint2*>(aos_row + (size_t) b * 144u + 16u + (size_t) u * 16u);
            return __ldg(p + hh);
        }
    };

    for (int b = 0; b < K_blk; ++b) {
        uint32_t s0, s1, s2, dm_u;
        if (IS_SOA) {
            const uint4 scdm = __ldg(Wscdm + (size_t) b * N + row);
            s0 = scdm.x; s1 = scdm.y; s2 = scdm.z; dm_u = scdm.w;
        } else {
            const uint4 hdr = __ldg(reinterpret_cast<const uint4*>(aos_row + (size_t) b * 144u));
            dm_u = hdr.x; s0 = hdr.y; s1 = hdr.z; s2 = hdr.w;
        }
        const float2 dm = __half22float2(pt_h2_u32(dm_u));
        const uint32_t sc_w[2] = {s0 & 0x3f3f3f3fu, (s2 & 0x0f0f0f0fu) | ((s0 >> 2) & 0x30303030u)};
        const uint32_t m_w[2]  = {s1 & 0x3f3f3f3fu, ((s2 >> 4) & 0x0f0f0f0fu) | ((s1 >> 2) & 0x30303030u)};

        // Preload chunk 0, stage 0 into buf 0
        uint2 q0 = load_q8(b, 0);
        uint2 q1 = load_q8(b, 1);
        auto get_sm = [&](int sb, __half2& dsc, __half2& ndm) {
            const float sc = (float) ((sc_w[sb >> 2] >> ((sb & 3) * 8)) & 0xffu);
            const float mn = (float) ((m_w[sb >> 2] >> ((sb & 3) * 8)) & 0xffu);
            dsc = __float2half2_rn(dm.x * sc);
            ndm = __float2half2_rn(-dm.y * mn);
        };
        __half2 dsc0, ndm0, dsc1, ndm1;
        get_sm(0, dsc0, ndm0);
        get_sm(1, dsc1, ndm1);
        {
            uint32_t wv[4];
            pt_decode_q4k_8(q0, false, dsc0, ndm0, c1024, wv);
            put_w(0, wv);
            put_x(0, b * 256 + 0);
        }
        __syncthreads();

#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int kb = b * 256 + c * 64;
            uint32_t wv[4];
            // Stage 1 -> buf 1 while FMA buf 0 (Stage 0)
            pt_decode_q4k_8(q1, false, dsc0, ndm0, c1024, wv);
            put_w(1, wv);
            put_x(1, kb + 16);
            if (active) pt_fma_stage_h<0>(Wv, Xv, tx, ty, acc);
            __syncthreads();

            // Stage 2 -> buf 0 while FMA buf 1 (Stage 1)
            pt_decode_q4k_8(q0, true, dsc1, ndm1, c1024, wv);
            put_w(0, wv);
            put_x(0, kb + 32);
            if (active) pt_fma_stage_h<1>(Wv, Xv, tx, ty, acc);
            __syncthreads();

            // Stage 3 -> buf 1 while FMA buf 0 (Stage 2)
            pt_decode_q4k_8(q1, true, dsc1, ndm1, c1024, wv);
            put_w(1, wv);
            put_x(1, kb + 48);
            if (active) pt_fma_stage_h<0>(Wv, Xv, tx, ty, acc);
            __syncthreads();

            // Next chunk's Stage 0 -> buf 0 while FMA buf 1 (Stage 3)
            if (c + 1 < 4) {
                q0 = load_q8(b, (c + 1) * 2 + 0);
                q1 = load_q8(b, (c + 1) * 2 + 1);
                get_sm((c + 1) * 2 + 0, dsc0, ndm0);
                get_sm((c + 1) * 2 + 1, dsc1, ndm1);
                pt_decode_q4k_8(q0, false, dsc0, ndm0, c1024, wv);
                put_w(0, wv);
                put_x(0, kb + 64);
            }
            if (active) pt_fma_stage_h<1>(Wv, Xv, tx, ty, acc);
            __syncthreads();
        }
        if (active) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc32[i][j] += __low2float(acc[i][j]) + __high2float(acc[i][j]);
                    acc[i][j] = __float2half2_rn(0.0f);
                }
        }
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = ty * 4 + j;
        if (t >= ntok) continue;
        float* d = gu_out + (size_t) (o0 + t) * N + mt * PT_BM;
        const int c0 = mt * PT_BM + tx * 4, c1 = mt * PT_BM + PT_BM / 2 + tx * 4;
        if (c0 + 3 < N) *(float4*) (d + tx * 4) = make_float4(acc32[0][j], acc32[1][j], acc32[2][j], acc32[3][j]);
        if (c1 + 3 < N) *(float4*) (d + PT_BM / 2 + tx * 4) = make_float4(acc32[4][j], acc32[5][j], acc32[6][j], acc32[7][j]);
    }
}

template <bool IS_SOA>
__device__ __forceinline__ void pt_decode_q51_8(uint2 qs2, uint32_t qh, int hh, bool hi_half,
                                                __half2 d_h2, __half2 m_h2, __half2 c1024, uint32_t (&out)[4]) {
    uint32_t hb0, hb1;
    if constexpr (IS_SOA) {
        const int u0 = hh * 2;
        if (hi_half) {
            hb0 = (qh >> u0) & 0x10101010u;
            hb1 = (qh >> (u0 + 1)) & 0x10101010u;
        } else {
            hb0 = (qh << (4 - u0)) & 0x10101010u;
            hb1 = (qh << (3 - u0)) & 0x10101010u;
        }
    } else {
        const int sh0 = (hi_half ? 16 : 0) + hh * 8;
        hb0 = ((((qh >> (sh0 + 0)) & 0xfu) * 0x00204081u) & 0x01010101u) << 4;
        hb1 = ((((qh >> (sh0 + 4)) & 0xfu) * 0x00204081u) & 0x01010101u) << 4;
    }
    const uint32_t w0 = (hi_half ? ((qs2.x >> 4) & 0x0f0f0f0fu) : (qs2.x & 0x0f0f0f0fu)) | hb0;
    const uint32_t w1 = (hi_half ? ((qs2.y >> 4) & 0x0f0f0f0fu) : (qs2.y & 0x0f0f0f0fu)) | hb1;
    const uint32_t p0 = __byte_perm(w0, 0x64006400u, 0x5150);
    const uint32_t p1 = __byte_perm(w0, 0x64006400u, 0x5352);
    const uint32_t p2 = __byte_perm(w1, 0x64006400u, 0x5150);
    const uint32_t p3 = __byte_perm(w1, 0x64006400u, 0x5352);
    out[0] = pt_u32_h2(__hfma2(__hsub2(pt_h2_u32(p0), c1024), d_h2, m_h2));
    out[1] = pt_u32_h2(__hfma2(__hsub2(pt_h2_u32(p1), c1024), d_h2, m_h2));
    out[2] = pt_u32_h2(__hfma2(__hsub2(pt_h2_u32(p2), c1024), d_h2, m_h2));
    out[3] = pt_u32_h2(__hfma2(__hsub2(pt_h2_u32(p3), c1024), d_h2, m_h2));
}

template <bool IS_SOA>
__global__ void __launch_bounds__(PT_NTHR, 2) p100_prefill_q51_dn_tiled_kernel(
    const unsigned long long* __restrict__ exp_ptr, const IqPrefillTile* __restrict__ tiles,
    int N, int n_ff, const uint16_t* __restrict__ hh_f16, float* __restrict__ dm_out) {
    const IqPrefillTile td = tiles[blockIdx.y];
    const int ntok = td.ntok;
    if (ntok <= 0) return;
    const int o0 = td.tok0;
    const int mt = (int) blockIdx.x;
    const int K_blk = n_ff / 32;

    __shared__ __align__(16) uint32_t Ws[2][PT_BK / 2][PT_WPAD];
    __shared__ __align__(16) uint32_t Xs[2][PT_BK / 2][PT_BN];

    const int tid = threadIdx.x;
    const int wrow = tid >> 1, hh = tid & 1;
    const int row = min(mt * PT_BM + wrow, N - 1);
    const int xr = tid >> 2, xc = tid & 3;
    const bool xok = xr < ntok;
    const __half* xp_base = reinterpret_cast<const __half*>(hh_f16 + (size_t) (o0 + min(xr, ntok - 1)) * n_ff + 4 * xc);
    const int xs = xr ^ (xc << 3);
    const int tx = tid & 15, ty = tid >> 4;
    const bool active = (tid >> 5) * 8 < ntok;

    const uint8_t* dn_dev = (const uint8_t*) exp_ptr[td.exp];
    const uint4* Wqs = IS_SOA ? reinterpret_cast<const uint4*>(dn_dev) : nullptr;
    const uint2* Wdmqh = IS_SOA ? reinterpret_cast<const uint2*>(dn_dev + (size_t) 16 * K_blk * N) : nullptr;
    const uint8_t* aos_row = IS_SOA ? nullptr : (dn_dev + (size_t) row * (size_t) K_blk * 24u);

    float acc32[8][4];
    __half2 acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc32[i][j] = 0.0f;
            acc[i][j] = __float2half2_rn(0.0f);
        }

    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    const uint2 zero2 = make_uint2(0u, 0u);
    const auto Wv = reinterpret_cast<const uint4 (*)[PT_BK / 2][PT_WPAD / 4]>(&Ws[0][0][0]);
    const auto Xv = reinterpret_cast<const uint4 (*)[PT_BK / 2][PT_BN / 4]>(&Xs[0][0][0]);

    const auto put_x = [&](int buf, int k_off) {
        const uint2 v = xok ? __ldg(reinterpret_cast<const uint2*>(xp_base + k_off)) : zero2;
        Xs[buf][2 * xc + 0][xs] = v.x;
        Xs[buf][2 * xc + 1][xs] = v.y;
    };
    const auto put_w = [&](int buf, const uint32_t (&wv)[4]) {
        Ws[buf][4 * hh + 0][wrow] = wv[0];
        Ws[buf][4 * hh + 1][wrow] = wv[1];
        Ws[buf][4 * hh + 2][wrow] = wv[2];
        Ws[buf][4 * hh + 3][wrow] = wv[3];
    };
    struct Raw51 { uint2 qs2; uint32_t qh; __half2 d_h2; __half2 m_h2; };
    const auto load_blk = [&](int b) -> Raw51 {
        Raw51 r;
        uint2 dmqh;
        if (IS_SOA) {
            r.qs2 = __ldg(reinterpret_cast<const uint2*>(Wqs + (size_t) b * N + row) + hh);
            dmqh = __ldg(Wdmqh + (size_t) b * N + row);
        } else {
            const uint8_t* bp = aos_row + (size_t) b * 24u;
            dmqh = __ldg(reinterpret_cast<const uint2*>(bp));
            r.qs2 = __ldg(reinterpret_cast<const uint2*>(bp + 8u) + hh);
        }
        const float2 dm = __half22float2(pt_h2_u32(dmqh.x));
        r.qh = dmqh.y;
        r.d_h2 = __float2half2_rn(dm.x);
        r.m_h2 = __float2half2_rn(dm.y);
        return r;
    };
    const auto flush = [&]() {
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc32[i][j] += __low2float(acc[i][j]) + __high2float(acc[i][j]);
                acc[i][j] = __float2half2_rn(0.0f);
            }
    };

    Raw51 cur = load_blk(0);
    {
        uint32_t wv[4];
        pt_decode_q51_8<IS_SOA>(cur.qs2, cur.qh, hh, false, cur.d_h2, cur.m_h2, c1024, wv);
        put_w(0, wv);
        put_x(0, 0);
    }
    __syncthreads();

    for (int b = 0; b < K_blk; ++b) {
        const bool more = b + 1 < K_blk;
        uint32_t wv[4];
        // Stage 1 (hi half) -> buf 1 while FMA buf 0 (lo half)
        pt_decode_q51_8<IS_SOA>(cur.qs2, cur.qh, hh, true, cur.d_h2, cur.m_h2, c1024, wv);
        put_w(1, wv);
        put_x(1, b * 32 + 16);
        Raw51 nxt;
        if (more) nxt = load_blk(b + 1);
        if (active) pt_fma_stage_h<0>(Wv, Xv, tx, ty, acc);
        __syncthreads();

        // Next block Stage 0 -> buf 0 while FMA buf 1 (hi half)
        if (more) {
            pt_decode_q51_8<IS_SOA>(nxt.qs2, nxt.qh, hh, false, nxt.d_h2, nxt.m_h2, c1024, wv);
            put_w(0, wv);
            put_x(0, (b + 1) * 32);
            cur = nxt;
        }
        if (active) pt_fma_stage_h<1>(Wv, Xv, tx, ty, acc);
        __syncthreads();
        if (active && (b & 7) == 7) flush();
    }
    if (active && (K_blk & 7) != 0) flush();

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = ty * 4 + j;
        if (t >= ntok) continue;
        float* d = dm_out + (size_t) (o0 + t) * N + mt * PT_BM;
        const int c0 = mt * PT_BM + tx * 4, c1 = mt * PT_BM + PT_BM / 2 + tx * 4;
        if (c0 + 3 < N) *(float4*) (d + tx * 4) = make_float4(acc32[0][j], acc32[1][j], acc32[2][j], acc32[3][j]);
        if (c1 + 3 < N) *(float4*) (d + PT_BM / 2 + tx * 4) = make_float4(acc32[4][j], acc32[5][j], acc32[6][j], acc32[7][j]);
    }
}

template <bool IS_SOA>
__device__ __forceinline__ void pt_decode_q80_8(uint2 q8, __half2 d_h2, __half2 c1152, uint32_t (&out)[4]) {
    const uint32_t v0 = IS_SOA ? q8.x : (q8.x ^ 0x80808080u);
    const uint32_t v1 = IS_SOA ? q8.y : (q8.y ^ 0x80808080u);
    const uint32_t p0 = __byte_perm(v0, 0x64006400u, 0x5150);
    const uint32_t p1 = __byte_perm(v0, 0x64006400u, 0x5352);
    const uint32_t p2 = __byte_perm(v1, 0x64006400u, 0x5150);
    const uint32_t p3 = __byte_perm(v1, 0x64006400u, 0x5352);
    out[0] = pt_u32_h2(__hmul2(__hsub2(pt_h2_u32(p0), c1152), d_h2));
    out[1] = pt_u32_h2(__hmul2(__hsub2(pt_h2_u32(p1), c1152), d_h2));
    out[2] = pt_u32_h2(__hmul2(__hsub2(pt_h2_u32(p2), c1152), d_h2));
    out[3] = pt_u32_h2(__hmul2(__hsub2(pt_h2_u32(p3), c1152), d_h2));
}

__device__ __forceinline__ uint2 pt_load_u2_unaligned(const uint8_t* p) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    const uint32_t a = (uint32_t) __ldg(h + 0) | ((uint32_t) __ldg(h + 1) << 16);
    const uint32_t b = (uint32_t) __ldg(h + 2) | ((uint32_t) __ldg(h + 3) << 16);
    return make_uint2(a, b);
}

template <bool IS_SOA>
__global__ void __launch_bounds__(PT_NTHR, 2) p100_prefill_q80_dn_tiled_kernel(
    const unsigned long long* __restrict__ exp_ptr, const IqPrefillTile* __restrict__ tiles,
    int N, int n_ff, const uint16_t* __restrict__ hh_f16, float* __restrict__ dm_out) {
    const IqPrefillTile td = tiles[blockIdx.y];
    const int ntok = td.ntok;
    if (ntok <= 0) return;
    const int o0 = td.tok0;
    const int mt = (int) blockIdx.x;
    const int K_blk = n_ff / 32;

    __shared__ __align__(16) uint32_t Ws[2][PT_BK / 2][PT_WPAD];
    __shared__ __align__(16) uint32_t Xs[2][PT_BK / 2][PT_BN];

    const int tid = threadIdx.x;
    const int wrow = tid >> 1, hh = tid & 1;
    const int row = min(mt * PT_BM + wrow, N - 1);
    const int xr = tid >> 2, xc = tid & 3;
    const bool xok = xr < ntok;
    const __half* xp_base = reinterpret_cast<const __half*>(hh_f16 + (size_t) (o0 + min(xr, ntok - 1)) * n_ff + 4 * xc);
    const int xs = xr ^ (xc << 3);
    const int tx = tid & 15, ty = tid >> 4;
    const bool active = (tid >> 5) * 8 < ntok;

    const uint8_t* dn_dev = (const uint8_t*) exp_ptr[td.exp];
    const uint4* Wqs = IS_SOA ? reinterpret_cast<const uint4*>(dn_dev) : nullptr;
    const __half* Wd = IS_SOA ? reinterpret_cast<const __half*>(dn_dev + (size_t) 32 * K_blk * N) : nullptr;
    const uint8_t* aos_row = IS_SOA ? nullptr : (dn_dev + (size_t) row * (size_t) K_blk * 34u);

    float acc32[8][4];
    __half2 acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc32[i][j] = 0.0f;
            acc[i][j] = __float2half2_rn(0.0f);
        }

    const __half2 c1152 = __floats2half2_rn(1152.0f, 1152.0f);
    const uint2 zero2 = make_uint2(0u, 0u);
    const auto Wv = reinterpret_cast<const uint4 (*)[PT_BK / 2][PT_WPAD / 4]>(&Ws[0][0][0]);
    const auto Xv = reinterpret_cast<const uint4 (*)[PT_BK / 2][PT_BN / 4]>(&Xs[0][0][0]);

    const auto put_x = [&](int buf, int k_off) {
        const uint2 v = xok ? __ldg(reinterpret_cast<const uint2*>(xp_base + k_off)) : zero2;
        Xs[buf][2 * xc + 0][xs] = v.x;
        Xs[buf][2 * xc + 1][xs] = v.y;
    };
    const auto put_w = [&](int buf, const uint32_t (&wv)[4]) {
        Ws[buf][4 * hh + 0][wrow] = wv[0];
        Ws[buf][4 * hh + 1][wrow] = wv[1];
        Ws[buf][4 * hh + 2][wrow] = wv[2];
        Ws[buf][4 * hh + 3][wrow] = wv[3];
    };
    struct Raw80 { uint2 q0, q1; __half2 d_h2; };
    const auto load_blk = [&](int b) -> Raw80 {
        Raw80 r;
        if (IS_SOA) {
            r.q0 = __ldg(reinterpret_cast<const uint2*>(Wqs + ((size_t) b * 2 + 0) * N + row) + hh);
            r.q1 = __ldg(reinterpret_cast<const uint2*>(Wqs + ((size_t) b * 2 + 1) * N + row) + hh);
            const __half dh = __ldg(Wd + (size_t) b * N + row);
            r.d_h2 = __half2half2(dh);
        } else {
            const uint8_t* bp = aos_row + (size_t) b * 34u;
            const __half dh = __ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(bp)));
            r.d_h2 = __half2half2(dh);
            r.q0 = pt_load_u2_unaligned(bp + 2u + hh * 8u);
            r.q1 = pt_load_u2_unaligned(bp + 18u + hh * 8u);
        }
        return r;
    };
    const auto flush = [&]() {
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc32[i][j] += __low2float(acc[i][j]) + __high2float(acc[i][j]);
                acc[i][j] = __float2half2_rn(0.0f);
            }
    };

    Raw80 cur = load_blk(0);
    {
        uint32_t wv[4];
        pt_decode_q80_8<IS_SOA>(cur.q0, cur.d_h2, c1152, wv);
        put_w(0, wv);
        put_x(0, 0);
    }
    __syncthreads();

    for (int b = 0; b < K_blk; ++b) {
        const bool more = b + 1 < K_blk;
        uint32_t wv[4];
        pt_decode_q80_8<IS_SOA>(cur.q1, cur.d_h2, c1152, wv);
        put_w(1, wv);
        put_x(1, b * 32 + 16);
        Raw80 nxt;
        if (more) nxt = load_blk(b + 1);
        if (active) pt_fma_stage_h<0>(Wv, Xv, tx, ty, acc);
        __syncthreads();

        if (more) {
            pt_decode_q80_8<IS_SOA>(nxt.q0, nxt.d_h2, c1152, wv);
            put_w(0, wv);
            put_x(0, (b + 1) * 32);
            cur = nxt;
        }
        if (active) pt_fma_stage_h<1>(Wv, Xv, tx, ty, acc);
        __syncthreads();
        if (active && (b & 7) == 7) flush();
    }
    if (active && (K_blk & 7) != 0) flush();

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = ty * 4 + j;
        if (t >= ntok) continue;
        float* d = dm_out + (size_t) (o0 + t) * N + mt * PT_BM;
        const int c0 = mt * PT_BM + tx * 4, c1 = mt * PT_BM + PT_BM / 2 + tx * 4;
        if (c0 + 3 < N) *(float4*) (d + tx * 4) = make_float4(acc32[0][j], acc32[1][j], acc32[2][j], acc32[3][j]);
        if (c1 + 3 < N) *(float4*) (d + PT_BM / 2 + tx * 4) = make_float4(acc32[4][j], acc32[5][j], acc32[6][j], acc32[7][j]);
    }
}

__device__ __forceinline__ void pt_decode_q5k_8(uint2 q8, uint2 qh8, bool hi_nib, int qh_bit,
                                                __half2 d_sc_h2, __half2 neg_d_mn_h2, __half2 c1024, uint32_t (&out)[4]) {
    const uint32_t hb0 = ((qh8.x >> qh_bit) & 0x01010101u) << 4;
    const uint32_t hb1 = ((qh8.y >> qh_bit) & 0x01010101u) << 4;
    const uint32_t w0 = (hi_nib ? ((q8.x >> 4) & 0x0f0f0f0fu) : (q8.x & 0x0f0f0f0fu)) | hb0;
    const uint32_t w1 = (hi_nib ? ((q8.y >> 4) & 0x0f0f0f0fu) : (q8.y & 0x0f0f0f0fu)) | hb1;
    const uint32_t p0 = __byte_perm(w0, 0x64006400u, 0x5150);
    const uint32_t p1 = __byte_perm(w0, 0x64006400u, 0x5352);
    const uint32_t p2 = __byte_perm(w1, 0x64006400u, 0x5150);
    const uint32_t p3 = __byte_perm(w1, 0x64006400u, 0x5352);
    out[0] = pt_u32_h2(__hfma2(__hsub2(pt_h2_u32(p0), c1024), d_sc_h2, neg_d_mn_h2));
    out[1] = pt_u32_h2(__hfma2(__hsub2(pt_h2_u32(p1), c1024), d_sc_h2, neg_d_mn_h2));
    out[2] = pt_u32_h2(__hfma2(__hsub2(pt_h2_u32(p2), c1024), d_sc_h2, neg_d_mn_h2));
    out[3] = pt_u32_h2(__hfma2(__hsub2(pt_h2_u32(p3), c1024), d_sc_h2, neg_d_mn_h2));
}

__global__ void __launch_bounds__(PT_NTHR, 2) p100_prefill_q5k_gu_tiled_kernel(
    const unsigned long long* __restrict__ exp_ptr, const IqPrefillTile* __restrict__ tiles,
    int n_ff, int n_embd, size_t up_off, const uint16_t* __restrict__ xs_f16, float* __restrict__ gu_out) {
    const IqPrefillTile td = tiles[blockIdx.y];
    const int ntok = td.ntok;
    if (ntok <= 0) return;
    const int o0 = td.tok0;
    const int mt = (int) blockIdx.x;
    const int N = 2 * n_ff;
    const int K_blk = n_embd / 256;

    __shared__ __align__(16) uint32_t Ws[2][PT_BK / 2][PT_WPAD];
    __shared__ __align__(16) uint32_t Xs[2][PT_BK / 2][PT_BN];

    const int tid = threadIdx.x;
    const int wrow = tid >> 1, hh = tid & 1;
    const int row = min(mt * PT_BM + wrow, N - 1);
    const int xr = tid >> 2, xc = tid & 3;
    const bool xok = xr < ntok;
    const __half* xp_base = reinterpret_cast<const __half*>(xs_f16 + (size_t) (o0 + min(xr, ntok - 1)) * n_embd + 4 * xc);
    const int xs = xr ^ (xc << 3);
    const int tx = tid & 15, ty = tid >> 4;
    const bool active = (tid >> 5) * 8 < ntok;

    const uint8_t* blob = (const uint8_t*) exp_ptr[td.exp];
    const uint8_t* aos_row = (row < n_ff ? blob : (blob + up_off)) +
                             (size_t) (row < n_ff ? row : (row - n_ff)) * (size_t) K_blk * 176u;

    float acc32[8][4];
    __half2 acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc32[i][j] = 0.0f;
            acc[i][j] = __float2half2_rn(0.0f);
        }

    const __half2 c1024 = __floats2half2_rn(1024.0f, 1024.0f);
    const uint2 zero2 = make_uint2(0u, 0u);
    const auto Wv = reinterpret_cast<const uint4 (*)[PT_BK / 2][PT_WPAD / 4]>(&Ws[0][0][0]);
    const auto Xv = reinterpret_cast<const uint4 (*)[PT_BK / 2][PT_BN / 4]>(&Xs[0][0][0]);

    const auto put_x = [&](int buf, int k_off) {
        const uint2 v = xok ? __ldg(reinterpret_cast<const uint2*>(xp_base + k_off)) : zero2;
        Xs[buf][2 * xc + 0][xs] = v.x;
        Xs[buf][2 * xc + 1][xs] = v.y;
    };
    const auto put_w = [&](int buf, const uint32_t (&wv)[4]) {
        Ws[buf][4 * hh + 0][wrow] = wv[0];
        Ws[buf][4 * hh + 1][wrow] = wv[1];
        Ws[buf][4 * hh + 2][wrow] = wv[2];
        Ws[buf][4 * hh + 3][wrow] = wv[3];
    };

    for (int b = 0; b < K_blk; ++b) {
        const uint8_t* bp = aos_row + (size_t) b * 176u;
        const uint4 hdr = __ldg(reinterpret_cast<const uint4*>(bp));
        const float2 dm = __half22float2(pt_h2_u32(hdr.x));
        const uint32_t s0 = hdr.y, s1 = hdr.z, s2 = hdr.w;
        const uint32_t sc_w[2] = {s0 & 0x3f3f3f3fu, (s2 & 0x0f0f0f0fu) | ((s0 >> 2) & 0x30303030u)};
        const uint32_t m_w[2]  = {s1 & 0x3f3f3f3fu, ((s2 >> 4) & 0x0f0f0f0fu) | ((s1 >> 2) & 0x30303030u)};
        const uint2 qh0 = __ldg(reinterpret_cast<const uint2*>(bp + 16u) + hh);
        const uint2 qh1 = __ldg(reinterpret_cast<const uint2*>(bp + 32u) + hh);
        const auto load_q8 = [&](int u) -> uint2 {
            return __ldg(reinterpret_cast<const uint2*>(bp + 48u + (size_t) u * 16u) + hh);
        };
        auto get_sm = [&](int sb, __half2& dsc, __half2& ndm) {
            const float sc = (float) ((sc_w[sb >> 2] >> ((sb & 3) * 8)) & 0xffu);
            const float mn = (float) ((m_w[sb >> 2] >> ((sb & 3) * 8)) & 0xffu);
            dsc = __float2half2_rn(dm.x * sc);
            ndm = __float2half2_rn(-dm.y * mn);
        };

        uint2 q0 = load_q8(0);
        uint2 q1 = load_q8(1);
        __half2 dsc0, ndm0, dsc1, ndm1;
        get_sm(0, dsc0, ndm0);
        get_sm(1, dsc1, ndm1);
        {
            uint32_t wv[4];
            pt_decode_q5k_8(q0, qh0, false, 0, dsc0, ndm0, c1024, wv);
            put_w(0, wv);
            put_x(0, b * 256 + 0);
        }
        __syncthreads();

#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int kb = b * 256 + c * 64;
            uint32_t wv[4];
            pt_decode_q5k_8(q1, qh1, false, 2 * c, dsc0, ndm0, c1024, wv);
            put_w(1, wv);
            put_x(1, kb + 16);
            if (active) pt_fma_stage_h<0>(Wv, Xv, tx, ty, acc);
            __syncthreads();

            pt_decode_q5k_8(q0, qh0, true, 2 * c + 1, dsc1, ndm1, c1024, wv);
            put_w(0, wv);
            put_x(0, kb + 32);
            if (active) pt_fma_stage_h<1>(Wv, Xv, tx, ty, acc);
            __syncthreads();

            pt_decode_q5k_8(q1, qh1, true, 2 * c + 1, dsc1, ndm1, c1024, wv);
            put_w(1, wv);
            put_x(1, kb + 48);
            if (active) pt_fma_stage_h<0>(Wv, Xv, tx, ty, acc);
            __syncthreads();

            if (c + 1 < 4) {
                q0 = load_q8((c + 1) * 2 + 0);
                q1 = load_q8((c + 1) * 2 + 1);
                get_sm((c + 1) * 2 + 0, dsc0, ndm0);
                get_sm((c + 1) * 2 + 1, dsc1, ndm1);
                pt_decode_q5k_8(q0, qh0, false, 2 * (c + 1), dsc0, ndm0, c1024, wv);
                put_w(0, wv);
                put_x(0, kb + 64);
            }
            if (active) pt_fma_stage_h<1>(Wv, Xv, tx, ty, acc);
            __syncthreads();
        }
        if (active) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc32[i][j] += __low2float(acc[i][j]) + __high2float(acc[i][j]);
                    acc[i][j] = __float2half2_rn(0.0f);
                }
        }
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = ty * 4 + j;
        if (t >= ntok) continue;
        float* d = gu_out + (size_t) (o0 + t) * N + mt * PT_BM;
        const int c0 = mt * PT_BM + tx * 4, c1 = mt * PT_BM + PT_BM / 2 + tx * 4;
        if (c0 + 3 < N) *(float4*) (d + tx * 4) = make_float4(acc32[0][j], acc32[1][j], acc32[2][j], acc32[3][j]);
        if (c1 + 3 < N) *(float4*) (d + PT_BM / 2 + tx * 4) = make_float4(acc32[4][j], acc32[5][j], acc32[6][j], acc32[7][j]);
    }
}

}  // namespace

bool iq_prefill_gu_tiled_supported(int gu_type) noexcept { return gu_type == 12 || gu_type == 13; }
bool iq_prefill_dn_tiled_supported(int d_type) noexcept { return d_type == 7 || d_type == 8; }

void iq_prefill_gu_tiled_grouped(int gu_type, bool is_soa, const unsigned long long* exp_ptr,
                                 const IqPrefillTile* tiles, int n_tiles, int64_t n_ff, int64_t n_embd, size_t up_off,
                                 const uint16_t* xs_f16, float* gu_out, void* stream) {
    if (n_tiles <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    const dim3 grid((unsigned) ((2 * n_ff + PT_BM - 1) / PT_BM), (unsigned) n_tiles);
    if (gu_type == 12) {
        if (is_soa)
            p100_prefill_q4k_gu_tiled_kernel<true><<<grid, PT_NTHR, 0, s>>>(exp_ptr, tiles, (int) n_ff, (int) n_embd,
                                                                            up_off, xs_f16, gu_out);
        else
            p100_prefill_q4k_gu_tiled_kernel<false><<<grid, PT_NTHR, 0, s>>>(exp_ptr, tiles, (int) n_ff, (int) n_embd,
                                                                             up_off, xs_f16, gu_out);
    } else if (gu_type == 13) {
        p100_prefill_q5k_gu_tiled_kernel<<<grid, PT_NTHR, 0, s>>>(exp_ptr, tiles, (int) n_ff, (int) n_embd,
                                                                  up_off, xs_f16, gu_out);
    } else {
        std::fprintf(stderr, "iq_prefill_gu_tiled_grouped: unsupported type %d\n", gu_type);
        std::exit(1);
    }
    check("iq_prefill_gu_tiled_grouped");
}

void iq_prefill_dn_tiled_grouped(int d_type, bool is_soa, const unsigned long long* exp_ptr,
                                 const IqPrefillTile* tiles, int n_tiles, int64_t n_embd, int64_t n_ff,
                                 const uint16_t* hh_f16, float* dm_out, void* stream) {
    if (n_tiles <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    const dim3 grid((unsigned) ((n_embd + PT_BM - 1) / PT_BM), (unsigned) n_tiles);
    if (d_type == 7) {
        if (is_soa)
            p100_prefill_q51_dn_tiled_kernel<true><<<grid, PT_NTHR, 0, s>>>(exp_ptr, tiles, (int) n_embd, (int) n_ff,
                                                                            hh_f16, dm_out);
        else
            p100_prefill_q51_dn_tiled_kernel<false><<<grid, PT_NTHR, 0, s>>>(exp_ptr, tiles, (int) n_embd, (int) n_ff,
                                                                             hh_f16, dm_out);
    } else if (d_type == 8) {
        if (is_soa)
            p100_prefill_q80_dn_tiled_kernel<true><<<grid, PT_NTHR, 0, s>>>(exp_ptr, tiles, (int) n_embd, (int) n_ff,
                                                                            hh_f16, dm_out);
        else
            p100_prefill_q80_dn_tiled_kernel<false><<<grid, PT_NTHR, 0, s>>>(exp_ptr, tiles, (int) n_embd, (int) n_ff,
                                                                             hh_f16, dm_out);
    } else {
        std::fprintf(stderr, "iq_prefill_dn_tiled_grouped: unsupported type %d\n", d_type);
        std::exit(1);
    }
    check("iq_prefill_dn_tiled_grouped");
}

void iq_repack_expert_blob_soa(const NativeExpertLayout& L, uint8_t* blob_dev, uint8_t* scratch_dev, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    int dev = 0;
    cudaGetDevice(&dev);
    static bool g_iq2s_inited[64] = {};
    if (dev >= 0 && dev < 64 && !g_iq2s_inited[dev]) {
        init_iq2s_h2_kernel<<<4, 256, 0, s>>>();
        g_iq2s_inited[dev] = true;
    }
    if (iq_gu_soa_ok(L)) {
        const int N_gu = 2 * (int) L.n_ff;
        const int K_blk = (int) (L.n_embd / 256);
        const int tot = N_gu * K_blk;
        const size_t gu_bytes = L.down_off;
        if (L.gu_type == 12) {
            repack_q4k_soa_kernel<<<(unsigned) ((tot + 255) / 256), 256, 0, s>>>(blob_dev, scratch_dev, N_gu, K_blk);
        } else if (L.gu_type == 16) {
            repack_iq2xxs_soa_kernel<<<(unsigned) ((tot + 255) / 256), 256, 0, s>>>(blob_dev, scratch_dev, N_gu, K_blk);
        } else if (L.gu_type == 22) {
            repack_iq2s_soa_kernel<<<(unsigned) ((tot + 255) / 256), 256, 0, s>>>(blob_dev, scratch_dev, N_gu, K_blk);
        }
        cudaMemcpyAsync(blob_dev, scratch_dev, gu_bytes, cudaMemcpyDeviceToDevice, s);
    }
    if (iq_dn_soa_ok(L)) {
        const int N_dn = (int) L.n_embd;
        const int K_blk = (int) (L.n_ff / (L.d_type == 42 ? 64 : 32));
        const int tot = N_dn * K_blk;
        const size_t dn_bytes = L.bytes - L.down_off;
        if (L.d_type == 7) {
            repack_q51_soa_kernel<<<(unsigned) ((tot + 255) / 256), 256, 0, s>>>(
                blob_dev + L.down_off, scratch_dev + L.down_off, N_dn, K_blk);
        } else if (L.d_type == 8) {
            repack_q80_soa_kernel<<<(unsigned) ((tot + 255) / 256), 256, 0, s>>>(
                blob_dev + L.down_off, scratch_dev + L.down_off, N_dn, K_blk);
        } else {
            repack_18b_soa_kernel<<<(unsigned) ((tot + 255) / 256), 256, 0, s>>>(
                blob_dev + L.down_off, scratch_dev + L.down_off, N_dn, K_blk);
        }
        cudaMemcpyAsync(blob_dev + L.down_off, scratch_dev + L.down_off, dn_bytes, cudaMemcpyDeviceToDevice, s);
    }
}

bool native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           float swiglu_limit, bool soa) {
    // glm-dsa B3.5: a type with no kernel below (the two switches' cases) is refused before any launch and reported by
    // the caller; it stopped the process (std::exit) when the K-quants were missing
    const auto has_kernel = [](int t) { return (t >= 10 && t <= 14) || (t >= 16 && t <= 23) || t == 29 || t == 42; };
    if (!has_kernel(L.gu_type) || !(has_kernel(L.d_type) || L.d_type == 6 || L.d_type == 7 || L.d_type == 8)) {
        std::fprintf(stderr, "native_expert_grouped: no kernel for gate/up type %d, down type %d\n", L.gu_type, L.d_type);
        return false;
    }
    if (cap_groups <= 0 || cap_entries <= 0) return true;
    cudaStream_t s = (cudaStream_t) stream;
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const dim3 ggu((unsigned) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS), (unsigned) cap_groups);
    if (L.u_type >= 0 && L.u_type != L.gu_type) {
        // a mixed layer: the gate rows by their type, the up rows by theirs (two launches, the same products)
        const dim3 gh((unsigned) ((L.n_ff + GU_ROWS - 1) / GU_ROWS), (unsigned) cap_groups);
        if (!has_kernel(L.u_type) || L.gu_type == 42 || L.u_type == 42 ||
            !launch_gu_half<false>(L.gu_type, gh, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate) ||
            !launch_gu_half<true>(L.u_type, gh, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, up)) {
            std::fprintf(stderr, "native_expert_grouped: no kernel for gate type %d with up type %d\n", L.gu_type,
                         L.u_type);
            return false;
        }
    } else switch (L.gu_type) {
        case 16:
            if (soa && iq_gu_soa_ok(L) && (L.n_ff & 31) == 0) {
                const dim3 gp((unsigned) ((2 * L.n_ff + 63) / 64), (unsigned) cap_groups);
                p100_iq2xxs_gu_soa_bsk_kernel<<<gp, 320, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            } else if (L.n_embd == 2560 && ((2 * L.n_ff) & 31) == 0 && iq_p100_hfma2_on()) {
                const dim3 gp((unsigned) ((2 * L.n_ff) / 32), (unsigned) cap_groups);
                p100_iq2xxs_gu_hfma2_kernel<<<gp, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            } else {
                native_gu_kernel<16><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            }
            break;
        case 17: native_gu_kernel<17><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 18: native_gu_kernel<18><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 21: native_gu_kernel<21><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 22:
            if (soa && iq_gu_soa_ok(L) && (L.n_ff & 31) == 0) {
                const dim3 gp((unsigned) ((2 * L.n_ff + 63) / 64), (unsigned) cap_groups);
                p100_iq2s_gu_soa_bsk_kernel<<<gp, 320, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            } else if (L.n_embd == 2560 && ((2 * L.n_ff) & 31) == 0 && iq_p100_hfma2_on()) {
                const dim3 gp((unsigned) ((2 * L.n_ff) / 32), (unsigned) cap_groups);
                p100_iq2s_gu_hfma2_kernel<<<gp, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            } else {
                native_gu_kernel<22><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            }
            break;
        case 23: native_gu_kernel<23><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 29: native_gu_kernel<29><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 19: native_gu_kernel<19><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 42: native_gu_kernel<42><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 10: native_gu_kernel<10><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 11: native_gu_kernel<11><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 12:
            if (soa && iq_gu_soa_ok(L) && (L.n_ff & 31) == 0 && iq_p100_hfma2_on()) {
                static const bool use_bsk2 = [] {
                    const char* v = std::getenv("MAYA_Q4K_GU_BSK2");
                    return v == nullptr || std::atoi(v) != 0;
                }();
                const dim3 gp((unsigned) ((2 * L.n_ff + 63) / 64), (unsigned) cap_groups);
                if (use_bsk2 && cap_entries <= 80) {
                    if (L.n_embd == 2560 && L.n_ff == 640)
                        p100_q4k_gu_soa_bsk2_kernel<true><<<gp, 320, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
                    else
                        p100_q4k_gu_soa_bsk2_kernel<false><<<gp, 320, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
                } else {
                    p100_q4k_gu_soa_bsk_kernel<<<gp, 320, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
                }
            } else if ((L.n_embd & 255) == 0 && ((2 * L.n_ff) & 31) == 0 && iq_p100_hfma2_on()) {
                const dim3 gp((unsigned) ((2 * L.n_ff + 31) / 32), (unsigned) cap_groups);
                p100_q4k_gu_hfma2_kernel<<<gp, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            } else {
                native_gu_kernel<12><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            }
            break;
        case 13:
            if ((L.n_embd & 255) == 0 && ((2 * L.n_ff) & 31) == 0 && iq_p100_hfma2_on()) {
                const dim3 gp((unsigned) ((2 * L.n_ff + 31) / 32), (unsigned) cap_groups);
                p100_q5k_gu_hfma2_kernel<<<gp, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            } else {
                native_gu_kernel<13><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            }
            break;
        case 14: native_gu_kernel<14><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 20: native_gu_kernel<20><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); return false;
    }
    check("native_expert_grouped/gu");
    // Grid sized to cap (CUDA-graph stable); fused swiglu+q8_1 bound to grp_start[*n_groups] (TOPK/empty idle).
    const long long nh = (long long) cap_entries * L.n_ff;
    if ((L.n_ff & 31) == 0) {
        swiglu_quant_q8_1_bounded_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(
            gate, up, hq, swiglu_limit, nh, n_groups, grp_start, (int) L.n_ff);
    } else {
        swiglu_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, swiglu_limit, nh, n_groups,
                                                                             grp_start, (int) L.n_ff);
        quantize_q8_1_bounded_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(h, hq, nh, n_groups, grp_start,
                                                                                   (int) L.n_ff);
    }
    const dim3 gd((unsigned) ((L.n_embd + 7) / 8), (unsigned) cap_groups);
    switch (L.d_type) {
        case 6:
            if (L.n_ff >= 32 && (L.n_ff & 31) == 0 && iq_p100_hfma2_on()) {
                const dim3 gd4((unsigned) ((L.n_embd + 63) / 64), (unsigned) cap_groups);
                p100_q50_down_kernel<<<gd4, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            } else {
                native_down_kernel<6><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            }
            break;
        case 7:
            if (soa && iq_dn_soa_ok(L) && iq_p100_hfma2_on()) {
                const dim3 gd4((unsigned) ((L.n_embd + 63) / 64), (unsigned) cap_groups);
                if (L.n_embd == 2560 && L.n_ff == 640)
                    p100_q51_down_soa_sk5_kernel<true><<<gd4, 320, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
                else if (L.n_ff >= 160)
                    p100_q51_down_soa_sk5_kernel<false><<<gd4, 320, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
                else
                    p100_q51_down_soa_kernel<<<gd4, 64, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            } else if (L.n_ff >= 32 && (L.n_ff & 31) == 0 && iq_p100_hfma2_on()) {
                const dim3 gd4((unsigned) ((L.n_embd + 63) / 64), (unsigned) cap_groups);
                p100_q51_down_hfma2_kernel<<<gd4, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            } else {
                native_down_kernel<7><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            }
            break;
        case 8:
            if (soa && iq_dn_soa_ok(L) && iq_p100_hfma2_on()) {
                const dim3 gd4((unsigned) ((L.n_embd + 63) / 64), (unsigned) cap_groups);
                if (L.n_embd == 2560 && L.n_ff == 640)
                    p100_q80_down_soa_sk5_kernel<true><<<gd4, 320, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
                else if (L.n_ff >= 160)
                    p100_q80_down_soa_sk5_kernel<false><<<gd4, 320, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
                else
                    p100_q80_down_soa_kernel<<<gd4, 64, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            } else if (L.n_ff >= 32 && (L.n_ff & 31) == 0 && iq_p100_hfma2_on()) {
                const dim3 gd4((unsigned) ((L.n_embd + 63) / 64), (unsigned) cap_groups);
                p100_q80_down_hfma2_kernel<<<gd4, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            } else {
                native_down_kernel<8><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            }
            break;
        case 18: native_down_kernel<18><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 20:
            if (soa && iq_dn_soa_ok(L)) {
                const dim3 gd4((unsigned) ((L.n_embd + 63) / 64), (unsigned) cap_groups);
                if (L.n_ff >= 160)
                    p100_iq4nl_down_soa_sk5_kernel<<<gd4, 320, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
                else
                    p100_iq4nl_down_soa_kernel<<<gd4, 64, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            } else if (L.n_ff == 640) {
                const dim3 gd4((unsigned) ((L.n_embd + 63) / 64), (unsigned) cap_groups);
                if (iq_p100_hfma2_on())
                    p100_iq4nl_down_hfma2_kernel<<<gd4, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
                else
                    p100_iq4nl_down_vmad_kernel<<<gd4, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            } else {
                native_down_kernel<20><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            }
            break;
        case 23: native_down_kernel<23><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 42:
            if (soa && iq_dn_soa_ok(L)) {
                const dim3 gd4((unsigned) ((L.n_embd + 63) / 64), (unsigned) cap_groups);
                p100_q20_down_soa_kernel<<<gd4, 64, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            } else if (L.n_ff == 640) {
                const dim3 gd4((unsigned) ((L.n_embd + 63) / 64), (unsigned) cap_groups);
                p100_q20_down_kernel<<<gd4, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            } else {
                native_down_kernel<42><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            }
            break;
        case 10: native_down_kernel<10><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 11: native_down_kernel<11><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 12: native_down_kernel<12><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 13: native_down_kernel<13><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 14: native_down_kernel<14><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 16: native_down_kernel<16><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 17: native_down_kernel<17><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 19: native_down_kernel<19><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 21: native_down_kernel<21><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 22: native_down_kernel<22><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 29: native_down_kernel<29><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); return false;
    }
    check("native_expert_grouped/down");
    return true;   // glm-dsa B3.5: launched
}
}  // namespace strata::kernels

// ---- Volta prompt experts (MAYA_PF_WMMA): a group's experts as one launch of FP16 tensor-core tiles, the weights
// dequantized into shared memory by dq_dispatch (the FP16 path's own formulas), the activations FP16, FP32 sums.
// Replaces MMQ's dp4a products on sm_70, and the FP16 path's dequantize-to-global + one cuBLAS call per expert.
#if !defined(__HIPCC__)
#include <mma.h>
namespace strata::kernels {
namespace {
constexpr int WG_BM = 64, WG_BN = 128, WG_THREADS = 256, WG_XK = 128, WG_MT = 4;
// The static codebook a TY carries (the kernel's `codebook`), which counts toward the block's shared memory.
__host__ __device__ constexpr size_t wg_codebook_words(int ty) {
    return (size_t) (ty == 22 ? 1024 : ty == 21 ? 256 : ty == 18 ? 128 : 2);
}
__host__ __device__ constexpr size_t gemm_iq_f16_grouped_smem(int wk, int xk = WG_XK) {
    return (size_t) WG_BN * (wk + 8) * 2 + (size_t) WG_BM * (xk + 8) * 2;
}
// A block covers 128 weight rows of one expert and up to MT * WG_BM token rows.
// Each weight chunk is decoded once into shared memory; activations use a BM x XK tile.
// Eight warps use 32 x 32 m-tiles and FP32 accumulators for MT m-tiles.
// DEC selects the decoder: 0 dq_dispatch (WK 256) or, for TY 18/21/22 at WK 128, dq_iq3_xxs_half / dq_iq3_s_half /
// dq_iq2_s_half, 1 Q2_0 (64 values), 2 IQ4_NL (32 values).  Two tile shapes: the wide one (MT = WG_MT = 4, 256 token
// rows, one block per SM) and the small one (MT = 2, 128 token rows, 43 KiB of weight tile at XK 64 or 39 KiB at
// XK 32, plus the static codebook under 48 KiB, so two blocks fit one V100 SM).  The launcher picks the shape from
// the launch's largest expert; both are exact and differ only in how many token rows share one weight decode.
// Optional ids gather input rows.  With a weight table (wsrc 1 = gate/up, 2 = down) expert e's matrix is the
// table's device pointer; otherwise experts are contiguous with expert_bytes stride.
template<int WK, int DEC, int TY = 0, int XK = WG_XK, int MT = WG_MT>
__global__ void __launch_bounds__(WG_THREADS, MT == 2 ? 2 : 1) gemm_iq_f16_grouped_kernel(
        int ty, const uint8_t* __restrict__ W, size_t expert_bytes, size_t row_bytes, int K,
        const __half* __restrict__ X, int ldx, const int32_t* __restrict__ bounds, const int32_t* __restrict__ ids,
        int ch_per_e, float* __restrict__ Y, int ldy, int wsrc, GroupedWeightTable tbl) {
    static_assert(gemm_iq_f16_grouped_smem(WK, XK) >= (size_t) WG_BM * (WG_BN + 4) * sizeof(float));
    static_assert(DEC != 2 || WK == 128, "IQ4_NL decodes one 128-value chunk per k0 step");
    static_assert(MT != 2 || (WK == 128 && (XK == 64 || (TY == 22 && XK == 32))),
                  "the two-m-tile tiles are the 128-value chunks; only IQ2_S's 8 KiB codebook needs XK 32");
    static_assert(MT != 2 || wg_codebook_words(TY) * sizeof(uint64_t) + gemm_iq_f16_grouped_smem(WK, XK) <= 48 * 1024,
                  "a small tile's static codebook and dynamic tiles must fit 48 KiB so two blocks fit one SM");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int LDW = WK + 8, LDX = XK + 8;
    extern __shared__ __align__(16) unsigned char wg_smem[];
    __half* sw = reinterpret_cast<__half*>(wg_smem);            // [BN][LDW]
    __half* sx = sw + WG_BN * LDW;                              // [BM][LDX]
    const int e = blockIdx.y / ch_per_e, ch = blockIdx.y % ch_per_e;
    const int c0 = bounds[e] + ch * MT * WG_BM, r1 = bounds[e + 1];
    if (c0 >= r1) return;
    const int crow = min(MT * WG_BM, r1 - c0);
    const int nmt = (crow + WG_BM - 1) / WG_BM;
    const int n0 = blockIdx.x * WG_BN;
    const uint8_t* we = wsrc == 0 ? W + (size_t) e * expert_bytes
                                  : (const uint8_t*) (wsrc == 1 ? tbl.gu[e] : tbl.dn[e]);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;                     // rows wm*32, cols wn*32 of an m-tile
    __shared__ uint64_t codebook[wg_codebook_words(TY)];
    auto* grid32 = reinterpret_cast<uint32_t*>(codebook);
    auto* grid8 = reinterpret_cast<int8_t*>(codebook);
    if constexpr (DEC == 0 && TY != 0) {
        if constexpr (TY == 22) {
            for (int i = tid; i < 1024; i += WG_THREADS) codebook[i] = iq2s_grid[i];
        } else if constexpr (TY == 21) {
            for (int i = tid; i < 512; i += WG_THREADS) grid32[i] = iq3s_grid[i];
        } else if constexpr (TY == 18) {
            for (int i = tid; i < 256; i += WG_THREADS) grid32[i] = iq3xxs_grid[i];
        } else if (tid < 16) grid8[tid] = kvalues_iq4nl[tid];
        __syncthreads();
    } else if constexpr (DEC == 2) {
        if (tid < 16) grid8[tid] = kvalues_iq4nl[tid];
        __syncthreads();
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT][2][2];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int b = 0; b < 2; ++b) wmma::fill_fragment(acc[t][a][b], 0.0f);
    // the activation tiles in order (k0, m-tile, half): the next one's XQ uint4 per thread are loaded into registers
    // while the current one is multiplied (the first before the weights' dequantization)
    constexpr int XQ = WG_BM * (XK / 8) / WG_THREADS;
    const int nx = WK / XK, per_k = nmt * nx, total = (K / WK) * per_k;
    uint4 xr[XQ];
    auto load_x = [&](int it) {
        const int kb = it / per_k, rem = it % per_k, t = rem / nx, xh = rem % nx;
        const int rows = min(WG_BM, crow - t * WG_BM);
#pragma unroll
        for (int q = 0; q < XQ; ++q) {
            const int i = tid + q * WG_THREADS, r = i / (XK / 8), c = (i % (XK / 8)) * 8;
            const int row = c0 + t * WG_BM + r;                  // the output row; ids gathers its input row
            xr[q] = r < rows ? *reinterpret_cast<const uint4*>(X + (size_t) (ids ? ids[row] : row) * ldx + kb * WK +
                                                              xh * XK + c)
                             : make_uint4(0, 0, 0, 0);
        }
    };
    load_x(0);
    int it = 0;
    for (int k0 = 0; k0 < K; k0 += WK) {
        if constexpr (DEC == 0) {
            if constexpr (WK == 128 && (TY == 18 || TY == 21 || TY == 22)) {
                // The small tile: a WK-128 chunk is one 128-value half of an IQ3_XXS / IQ3_S / IQ2_S superblock,
                // its sub-blocks ib = 4*kh .. 4*kh + 3 landing at columns 0 .. 127, sixteen lanes to a row and two
                // rows per warp, so a lane decodes the same eight values per half as the 256-value decoders below
                // give it per superblock.  The wide tile keeps those decoders.
                const int kh = (k0 % 256) / 128, sub = lane & 15, rr = lane >> 4;
                for (int r = 2 * warp + rr; r < WG_BN; r += 2 * (WG_THREADS / 32)) {
                    const uint8_t* wr = we + (size_t) (n0 + r) * row_bytes;
                    auto* out = sw + r * LDW;
                    if constexpr (TY == 18) dq_iq3_xxs_half(wr, k0 / 256, kh, sub, out, grid32);
                    else if constexpr (TY == 21) dq_iq3_s_half(wr, k0 / 256, kh, sub, out, grid32);
                    else dq_iq2_s_half(wr, k0 / 256, kh, sub, out, codebook);
                }
            } else {
                for (int r = warp; r < WG_BN; r += WG_THREADS / 32) {
                    const auto* wr = we + (size_t) (n0 + r) * row_bytes;
                    auto* out = sw + r * LDW;
                    if constexpr (TY == 22) dq_iq2_s(wr, k0 / 256, out, lane, codebook);
                    else if constexpr (TY == 21) dq_iq3_s(wr, k0 / 256, out, lane, grid32);
                    else if constexpr (TY == 18) dq_iq3_xxs(wr, k0 / 256, out, lane, grid32);
                    else if constexpr (TY == 23) dq_iq4_xs(wr, k0 / 256, out, lane, grid8);
                    else dq_dispatch<__half>(ty, wr, k0 / 256, out, lane);
                }
            }
        } else if constexpr (DEC == 1) {
            const int r = tid >> 1, blk = tid & 1;               // one thread per (row, 64-value block)
            const block_q2_0* b = reinterpret_cast<const block_q2_0*>(we + (size_t) (n0 + r) * row_bytes) + k0 / 64 + blk;
            // Codes select {-d, signed zero, d, 2*d}. Pack two exact FP16 weights with
            // a byte permutation instead of repeated integer-to-float conversions.
            const uint32_t d = __half_as_ushort(b->d);
            const uint32_t z = __half_as_ushort(__hmul(b->d, __float2half(0.0f)));
            const uint32_t twice = __half_as_ushort(__hadd(b->d, b->d));
            const uint32_t table01 = (d ^ 0x8000u) | (z << 16);
            const uint32_t table23 = d | (twice << 16);
            uint4 q4;
            memcpy(&q4, b->qs, 16);
            const uint8_t* qb = reinterpret_cast<const uint8_t*>(&q4);
            __half* o = sw + r * LDW + blk * 64;
#pragma unroll
            for (int c = 0; c < 16; ++c) {
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const unsigned q = qb[c] >> (4 * pair);
                    const unsigned a = (q & 3u) << 1, b = ((q >> 2) & 3u) << 1;
                    const unsigned selector = 0x1010u | a | (a << 4) | (b << 8) | (b << 12);
                    *reinterpret_cast<uint32_t*>(o + 4 * c + 2 * pair) = __byte_perm(table01, table23, selector);
                }
            }
        } else {
            const int r = tid >> 1, h = tid & 1;                 // one thread per (row, 64-value half of the chunk)
            const block_iq4_nl* b = reinterpret_cast<const block_iq4_nl*>(we + (size_t) (n0 + r) * row_bytes) + k0 / 32 + 2 * h;
            __half* o = sw + r * LDW + h * 64;
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const __half2 d = __halves2half2(b[blk].d, b[blk].d);
                const uint8_t* qs = b[blk].qs;
#pragma unroll
                for (int j = 0; j < 16; j += 2) {
                    uint16_t q;
                    memcpy(&q, qs + j, sizeof(q));
                    const __half2 lo = __floats2half2_rn((float) grid8[q & 0xf], (float) grid8[(q >> 8) & 0xf]);
                    const __half2 hi = __floats2half2_rn((float) grid8[(q >> 4) & 0xf], (float) grid8[q >> 12]);
                    *reinterpret_cast<__half2*>(o + 32 * blk + j) = __hmul2(d, lo);
                    *reinterpret_cast<__half2*>(o + 32 * blk + 16 + j) = __hmul2(d, hi);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t >= nmt) break;
#pragma unroll 1
            for (int xh = 0; xh < WK / XK; ++xh, ++it) {
                __syncthreads();                                 // the previous X tile is consumed (and W is written)
#pragma unroll
                for (int q = 0; q < XQ; ++q) {
                    const int i = tid + q * WG_THREADS, r = i / (XK / 8), c = (i % (XK / 8)) * 8;
                    *reinterpret_cast<uint4*>(sx + r * LDX + c) = xr[q];
                }
                __syncthreads();
                if (it + 1 < total) load_x(it + 1);
#pragma unroll
                for (int kk = 0; kk < XK; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        wmma::load_matrix_sync(a[f], sx + (wm * 32 + f * 16) * LDX + kk, LDX);
                        wmma::load_matrix_sync(b[f], sw + (wn * 32 + f * 16) * LDW + xh * XK + kk, LDW);
                    }
#pragma unroll
                    for (int fa = 0; fa < 2; ++fa)
#pragma unroll
                        for (int fb = 0; fb < 2; ++fb) wmma::mma_sync(acc[t][fa][fb], a[fa], b[fb], acc[t][fa][fb]);
                }
            }
        }
        __syncthreads();                                         // W is consumed before the next chunk
    }
    float* so = reinterpret_cast<float*>(wg_smem);               // [BM][BN + 4] per m-tile
    constexpr int LDO = WG_BN + 4;
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (t >= nmt) break;
        const int rows = min(WG_BM, crow - t * WG_BM);
#pragma unroll
        for (int fa = 0; fa < 2; ++fa)
#pragma unroll
            for (int fb = 0; fb < 2; ++fb)
                wmma::store_matrix_sync(so + (wm * 32 + fa * 16) * LDO + wn * 32 + fb * 16, acc[t][fa][fb], LDO, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < rows * (WG_BN / 4); i += WG_THREADS) {
            const int r = i / (WG_BN / 4), c = (i % (WG_BN / 4)) * 4;
            *reinterpret_cast<float4*>(Y + (size_t) (c0 + t * WG_BM + r) * ldy + n0 + c) = *reinterpret_cast<const float4*>(so + r * LDO + c);
        }
        __syncthreads();
    }
#endif
}

// The FP16 tensor cores need sm_70+, and the dynamic-shared-memory attribute is per device (a two-GPU process must
// raise it on each card): cache, per device, which instantiations the attribute was accepted for.  A 128 x 264-half
// weight tile is 83 KiB, past the 64 KiB a Turing card allows and within Volta's / Ampere's opt-in limit.  The bits:
// 1 queried, 2 the WK-256 gate/up kernels, 4 the WK-128 Q2_0 kernel, 8 the wide WK-128 IQ4_NL kernel,
// 16 the small IQ3_S tile, 32 the small IQ4_NL tile, 64 the small IQ3_XXS tile, 128 the small IQ2_S tile.  Each
// tile shape has its own bit: a device that refuses one attribute must keep that shape on the MMQ path rather than
// launch a kernel whose tile was never attributed.
unsigned gemm_iq_f16_grouped_caps(int dev) {
    static thread_local unsigned per_dev[64];   // per-thread: two prefill objects may first touch one device at once
    if (dev < 0 || dev >= 64) return 0;
    if (!(per_dev[dev] & 1u)) {
        unsigned caps = 1u;                      // queried
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= 7) {
            const bool gu = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(256)) == cudaSuccess;
            const bool g18 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, 0, 18>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(256)) == cudaSuccess;
            const bool g21 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, 0, 21>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(256)) == cudaSuccess;
            const bool g22 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, 0, 22>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(256)) == cudaSuccess;
            const bool g23 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, 0, 23>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(256)) == cudaSuccess;
            const bool d1 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 1, 0, 64, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(128, 64)) == cudaSuccess;
            const bool d2 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(128)) == cudaSuccess;
            const bool d21 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 0, 21, 64, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(128, 64)) == cudaSuccess;
            const bool d2s = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 2, 0, 64, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(128, 64)) == cudaSuccess;
            const bool d18 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 0, 18, 64, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(128, 64)) == cudaSuccess;
            const bool d22 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 0, 22, 32, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(128, 32)) == cudaSuccess;
            if (!gu || !g18 || !g21 || !g22 || !g23 || !d1 || !d2 || !d21 || !d2s || !d18 || !d22) (void) cudaGetLastError();
            caps |= ((gu && g18 && g21 && g22 && g23) ? 2u : 0u) | (d1 ? 4u : 0u) | (d2 ? 8u : 0u) |
                    (d21 ? 16u : 0u) | (d2s ? 32u : 0u) | (d18 ? 64u : 0u) | (d22 ? 128u : 0u);
        }
        per_dev[dev] = caps;
    }
    return per_dev[dev];
}
}  // namespace

bool gemm_iq_f16_grouped_supported(int dev) { return (gemm_iq_f16_grouped_caps(dev) & 2u) != 0; }

bool gemm_iq_f16_grouped(int ty, const void* W, size_t expert_bytes, int n_out, int K, const void* X, int ldx,
                         const int32_t* bounds, int n_experts, int max_rows, float* Y, int ldy, void* stream,
                         const int32_t* ids, const GroupedWeightTable* indirect, bool down) {
    if (n_out <= 0 || K <= 0 || n_out % WG_BN != 0 || n_experts <= 0 || max_rows <= 0) return false;
    if (indirect != nullptr && n_experts > indirect->n) return false;   // the table must cover the whole launch
    if (ldx < K || ldx % 8 != 0 || ldy < n_out || ldy % 4 != 0) return false;   // uint4 loads / float4 stores
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    // The IQ gate/up formats and IQ4_NL use 128-row tiles for short groups and 256-row tiles for wide groups.
    // The weight chunk is 128 values for IQ4_NL and Q2_0, and 128 (short) or 256 (wide) for IQ3_XXS / IQ3_S / IQ2_S.
    const int dec = ty == 42 ? 1 : ty == 20 ? 2 : 0;
    // A gate/up superblock is 256 values, so its K must split into whole 256-value chunks; the Q2_0 and IQ4_NL
    // kernels work on 64- and 32-value blocks, so a 640-column down product splits into whole 128-value chunks.
    if (dec == 0 ? (K % 256 != 0 || !is_iq(ty)) : K % 128 != 0) return false;
    const unsigned caps = gemm_iq_f16_grouped_caps(dev);
    // The small tile decodes a weight chunk once per 128 token rows but hides the decode behind two blocks per
    // SM, which wins while the group's largest expert holds at most one chunk; past that the wide tile's half as
    // frequent weight decode pays for its single block.  Both tiles are exact: only the chunking differs.
    const bool small = max_rows <= 2 * WG_BM && (dec == 2 || (dec == 0 && (ty == 18 || ty == 21 || ty == 22)));
    const unsigned need = small ? (dec == 2 ? 32u : ty == 22 ? 128u : ty == 18 ? 64u : 16u)
                                : dec == 0 ? 2u : dec == 1 ? 4u : 8u;
    if (!(caps & need)) return false;
    const size_t row_bytes = iq_row_bytes(ty, K);
    const int mt = small || dec == 1 ? 2 : WG_MT;
    const int chunks = (max_rows + mt * WG_BM - 1) / (mt * WG_BM);
    const dim3 grid((unsigned) (n_out / WG_BN), (unsigned) (n_experts * chunks));
    cudaStream_t s = (cudaStream_t) stream;
    const uint8_t* w = (const uint8_t*) W;
    const __half* x = (const __half*) X;
    const int wsrc = indirect == nullptr ? 0 : (down ? 2 : 1);
    const GroupedWeightTable tbl = indirect != nullptr ? *indirect : GroupedWeightTable{};
    if (dec == 0 && small && ty == 18)
        gemm_iq_f16_grouped_kernel<128, 0, 18, 64, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128, 64), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0 && small && ty == 21)
        gemm_iq_f16_grouped_kernel<128, 0, 21, 64, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128, 64), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0 && small && ty == 22)
        gemm_iq_f16_grouped_kernel<128, 0, 22, 32, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128, 32), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0 && ty == 18)
        gemm_iq_f16_grouped_kernel<256, 0, 18><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0 && ty == 21)
        gemm_iq_f16_grouped_kernel<256, 0, 21><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0 && ty == 22)
        gemm_iq_f16_grouped_kernel<256, 0, 22><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0 && ty == 23)
        gemm_iq_f16_grouped_kernel<256, 0, 23><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0)
        gemm_iq_f16_grouped_kernel<256, 0><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 1)
        gemm_iq_f16_grouped_kernel<128, 1, 0, 64, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128, 64), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (small)
        gemm_iq_f16_grouped_kernel<128, 2, 0, 64, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128, 64), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else
        gemm_iq_f16_grouped_kernel<128, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    return cudaGetLastError() == cudaSuccess;
}
}  // namespace strata::kernels
#else
namespace strata::kernels {
bool gemm_iq_f16_grouped(int, const void*, size_t, int, int, const void*, int, const int32_t*, int, int, float*, int, void*,
                         const int32_t*, const GroupedWeightTable*, bool) {
    return false;
}
bool gemm_iq_f16_grouped_supported(int) { return false; }
}  // namespace strata::kernels
#endif
