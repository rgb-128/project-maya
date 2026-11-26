// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

struct __align__(2) GrQ80Block {
    __half d;
    int8_t qs[32];
};
static_assert(sizeof(GrQ80Block) == 34, "Q8_0 block must be 34 bytes");

__device__ __forceinline__ float dot8_q8(const int8_t* __restrict__ q, const float4 x0, const float4 x1) {
    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(q);
    const uint32_t w0 = uint32_t(__ldg(q16 + 0)) | (uint32_t(__ldg(q16 + 1)) << 16);
    const uint32_t w1 = uint32_t(__ldg(q16 + 2)) | (uint32_t(__ldg(q16 + 3)) << 16);
    float s = float(int8_t(w0 & 0xffu)) * x0.x;
    s = fmaf(float(int8_t((w0 >> 8) & 0xffu)), x0.y, s);
    s = fmaf(float(int8_t((w0 >> 16) & 0xffu)), x0.z, s);
    s = fmaf(float(int8_t(w0 >> 24)), x0.w, s);
    s = fmaf(float(int8_t(w1 & 0xffu)), x1.x, s);
    s = fmaf(float(int8_t((w1 >> 8) & 0xffu)), x1.y, s);
    s = fmaf(float(int8_t((w1 >> 16) & 0xffu)), x1.z, s);
    s = fmaf(float(int8_t(w1 >> 24)), x1.w, s);
    return s;
}

__device__ __forceinline__ float dot32_q8(const GrQ80Block* __restrict__ b, const float* __restrict__ x) {
    const float d = __half2float(__ldg(&b->d));
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float s = dot8_q8(b->qs + 0, x4[0], x4[1]);
    s += dot8_q8(b->qs + 8, x4[2], x4[3]);
    s += dot8_q8(b->qs + 16, x4[4], x4[5]);
    s += dot8_q8(b->qs + 24, x4[6], x4[7]);
    return d * s;
}

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ __half2 gr_u32_to_h2(uint32_t x) { return *reinterpret_cast<const __half2*>(&x); }
__device__ __forceinline__ __half2 gr_u8_lo_h2(uint32_t u) {
    return __hsub2(gr_u32_to_h2(__byte_perm(u, 0x64006400u, 0x5150)), __float2half2_rn(1152.0f));
}
__device__ __forceinline__ __half2 gr_u8_hi_h2(uint32_t u) {
    return __hsub2(gr_u32_to_h2(__byte_perm(u, 0x64006400u, 0x7372)), __float2half2_rn(1152.0f));
}
__device__ __forceinline__ float dot16_u8_f32(uint4 qv, const float* x) {
    const __half2 w0 = gr_u8_lo_h2(qv.x), w1 = gr_u8_hi_h2(qv.x);
    const __half2 w2 = gr_u8_lo_h2(qv.y), w3 = gr_u8_hi_h2(qv.y);
    const __half2 w4 = gr_u8_lo_h2(qv.z), w5 = gr_u8_hi_h2(qv.z);
    const __half2 w6 = gr_u8_lo_h2(qv.w), w7 = gr_u8_hi_h2(qv.w);
    __half2 p = __hmul2(w0, __floats2half2_rn(x[0], x[1]));
    p = __hfma2(w1, __floats2half2_rn(x[2], x[3]), p);
    p = __hfma2(w2, __floats2half2_rn(x[4], x[5]), p);
    p = __hfma2(w3, __floats2half2_rn(x[6], x[7]), p);
    p = __hfma2(w4, __floats2half2_rn(x[8], x[9]), p);
    p = __hfma2(w5, __floats2half2_rn(x[10], x[11]), p);
    p = __hfma2(w6, __floats2half2_rn(x[12], x[13]), p);
    p = __hfma2(w7, __floats2half2_rn(x[14], x[15]), p);
    const float2 f = __half22float2(p);
    return f.x + f.y;
}

// 8 bf16 packed in a uint4 against 8 floats.
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = fmaf(__uint_as_float(v[j] << 16), x[2 * j], acc);
        acc = fmaf(__uint_as_float(v[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}

__global__ void __launch_bounds__(THREADS) gr_down_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float xn[D];
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        if (blockIdx.x == 0) a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    __syncthreads();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    float acc = 0.0f;
    if (!inject_block && a.w_down_q8 != nullptr) {
        if (a.w_down_q8_soa) {
            // K-Major Q8_0_SOA (108): Wq [D/16][LR], Wd [D/32][LR]
            const uint4* wq = reinterpret_cast<const uint4*>(a.w_down_q8);
            const __half* wd = reinterpret_cast<const __half*>(
                reinterpret_cast<const uint8_t*>(a.w_down_q8) + (size_t) LR * D);
#pragma unroll 2
            for (int b = lane; b < D / 32; b += 32) {
                const float d = __half2float(__ldg(wd + (size_t) b * LR + row));
                const float s = dot16_u8_f32(__ldg(wq + (size_t) (b * 2 + 0) * LR + row), xn + b * 32) +
                                dot16_u8_f32(__ldg(wq + (size_t) (b * 2 + 1) * LR + row), xn + b * 32 + 16);
                acc = fmaf(d, s, acc);
            }
        } else {
            const GrQ80Block* wb = reinterpret_cast<const GrQ80Block*>(a.w_down_q8) + (size_t) row * (D / 32);
#pragma unroll 2
            for (int b = lane; b < D / 32; b += 32) acc += dot32_q8(wb + b, xn + b * 32);
        }
    } else {
        const uint16_t* wrow = (inject_block ? a.w_inject : a.w_down) + (size_t) row * D;
        const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
#pragma unroll 4
        for (int j = lane; j < D / 8; j += 32) acc += dot8(__ldg(w4 + j), xn + j * 8);
    }
    acc = warp_sum(acc);
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + __expf(-x));
    }
}

__global__ void __launch_bounds__(THREADS) gr_up_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float lo[LR];
    __shared__ float g[HC][UP_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UP_COLS;
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    __syncthreads();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        float acc = 0.0f;
        if (a.w_up_q8 != nullptr) {
            if (a.w_up_q8_soa) {
                if (lane < LR / 32) {
                    const uint4* wq = reinterpret_cast<const uint4*>(a.w_up_q8);
                    const __half* wd = reinterpret_cast<const __half*>(
                        reinterpret_cast<const uint8_t*>(a.w_up_q8) + (size_t) D * LR);
                    const float d = __half2float(__ldg(wd + (size_t) lane * D + i));
                    const float s = dot16_u8_f32(__ldg(wq + (size_t) (lane * 2) * D + i), lo + lane * 32) +
                                    dot16_u8_f32(__ldg(wq + (size_t) (lane * 2 + 1) * D + i), lo + lane * 32 + 16);
                    acc = d * s;
                }
            } else {
                const GrQ80Block* wb = reinterpret_cast<const GrQ80Block*>(a.w_up_q8) + (size_t) i * (LR / 32);
                if (lane < LR / 32) acc = dot32_q8(wb + lane, lo + lane * 32);
            }
        } else {
            const uint4* w4 = reinterpret_cast<const uint4*>(a.w_up + (size_t) i * LR);
            acc = dot8(__ldg(w4 + lane), lo + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8(__ldg(w4 + 32 + lane), lo + (32 + lane) * 8);
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    __syncthreads();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};

// Step 1 of `gr_down_kernel`, one block per (stream, token) = 4*T blocks (full SM utilization, single global store).
__global__ void __launch_bounds__(THREADS) gr_norm_multi_kernel(GrMulti m) {
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    const int c = (int) blockIdx.x;
    const int tok = (int) blockIdx.y;
    const FusedGrArgs& a = m.a[tok];
    float* xnc = m.xn + (size_t) tok * D + (size_t) c * N;
    const float* Rc = a.R + (size_t) c * N;
    const float* wc = a.w_norm + (size_t) c * N;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss = 0.0f;
    float4 rg[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int d = (t + k * THREADS) * 4;
        if (d < N) {
            float4 r = *reinterpret_cast<const float4*>(Rc + d);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = *reinterpret_cast<const float4*>(wc + d);
            ss += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            rg[k] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    __syncthreads();
    const float sc = s_rs;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int d = (t + k * THREADS) * 4;
        if (d < N) {
            *reinterpret_cast<float4*>(xnc + d) =
                make_float4(rg[k].x * sc, rg[k].y * sc, rg[k].z * sc, rg[k].w * sc);
        }
    }
}

// Fast Split-K direct L1/L2-cached gr_down for T<=4: 320 blocks (1 blk/row) x 128 threads (4 warps/row = 23 warps/SM
// on 56-SM P100, hides cold HBM2 latency -> 358-425 GB/s!). Block 320..323 handles the 4 inject_out rows identically.
template<int NT>
__global__ void __launch_bounds__(128) gr_down_direct_multi_kernel(GrMulti m) {
    __shared__ float part[NT][4];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int T = m.T;
    const bool inject_block = (int) blockIdx.x >= LR;
    const int row = inject_block ? ((int) blockIdx.x - LR) : (int) blockIdx.x;
    if (inject_block && (m.a[0].w_inject == nullptr || row >= HC)) return;
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) row * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (int j = tid; j < D / 8; j += 128) {
        const uint4 wv = __ldg(w4 + j);
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k >= T) break;
            const float4 xa = __ldg(reinterpret_cast<const float4*>(m.xn + (size_t) k * D + j * 8));
            const float4 xb = __ldg(reinterpret_cast<const float4*>(m.xn + (size_t) k * D + j * 8 + 4));
            float a = acc[k];
            a = fmaf(__uint_as_float(wv.x << 16), xa.x, a); a = fmaf(__uint_as_float(wv.x & 0xffff0000u), xa.y, a);
            a = fmaf(__uint_as_float(wv.y << 16), xa.z, a); a = fmaf(__uint_as_float(wv.y & 0xffff0000u), xa.w, a);
            a = fmaf(__uint_as_float(wv.z << 16), xb.x, a); a = fmaf(__uint_as_float(wv.z & 0xffff0000u), xb.y, a);
            a = fmaf(__uint_as_float(wv.w << 16), xb.z, a); a = fmaf(__uint_as_float(wv.w & 0xffff0000u), xb.w, a);
            acc[k] = a;
        }
    }
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        if (k >= T) break;
        const float s = warp_sum(acc[k]);
        if (lane == 0) part[k][warp] = s;
    }
    __syncthreads();
    if (tid < T) {
        const float s = part[tid][0] + part[tid][1] + part[tid][2] + part[tid][3];
        if (inject_block) {
            m.a[tid].inject_out[row] = s;
        } else {
            const float x = s / (float) HC;
            m.a[tid].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

#if defined(__HIPCC__)
constexpr int TILE = 1280;             // eight-token tile fits gfx1100's 64 KiB LDS limit
#else
constexpr int TILE = 2560;             // xn floats per token staged at a time: 320 chunks of 8, 10 per lane
#endif
constexpr int TQ = TILE / 8 / 32;      // uint4 weight chunks per lane per tile

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's 10 weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
__global__ void __launch_bounds__(THREADS) gr_down_multi_kernel(GrMulti m) {
    extern __shared__ __align__(16) float tile[];   // [T][TILE]
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
    for (int base = 0; base < D; base += TILE) {
        uint4 wv[TQ];
        if (active) {
#pragma unroll
            for (int q = 0; q < TQ; ++q) wv[q] = __ldg(w4 + base / 8 + lane + 32 * q);
        }
        __syncthreads();                                   // the previous tile is consumed
        const float4* src4 = reinterpret_cast<const float4*>(m.xn);
        float4* tile4 = reinterpret_cast<float4*>(tile);
        for (int i = t; i < T * (TILE / 4); i += THREADS) {
            const int k = i / (TILE / 4), off = i - k * (TILE / 4);
            tile4[i] = src4[((size_t) k * D + base) / 4 + off];
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k)
                if (k < T) acc[k] += dot8(wv[q], tile + k * TILE + j * 8);
        }
    }
    if (!active) return;
    float s[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
__global__ void __launch_bounds__(THREADS) gr_up_multi_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[kFusedGrMaxT][LR];
    __shared__ float g[kFusedGrMaxT][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        const uint4 wa = __ldg(w4 + lane);
        const uint4 wb = lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0);
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float acc = dot8(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
}

// Fast Split-K Q8_0 down kernel: 1.88x less HBM2 traffic (3.48 MB vs 6.55 MB BF16), exact to original GGUF Q8_0!
// 128 threads/block (4 warps), 4 lanes/block = 10 exact Q8_0 blocks/group (0% idle threads).
template<int NT>
__global__ void __launch_bounds__(128) gr_down_q8_0_multi_kernel(GrMulti m) {
    const int row = blockIdx.x;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const bool inject_block = row >= LR;
    const int inj_row = row - LR;
    if (inject_block && (m.a[0].w_inject == nullptr || inj_row >= HC)) return;

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

    if (inject_block) {
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_inject + (size_t) inj_row * D);
        for (int j = t; j < D / 8; j += 128) {
            const uint4 wv = __ldg(w4 + j);
            const uint32_t v0 = wv.x, v1 = wv.y, v2 = wv.z, v3 = wv.w;
            const float f0 = __uint_as_float(v0 << 16), f1 = __uint_as_float(v0 & 0xffff0000u);
            const float f2 = __uint_as_float(v1 << 16), f3 = __uint_as_float(v1 & 0xffff0000u);
            const float f4 = __uint_as_float(v2 << 16), f5 = __uint_as_float(v2 & 0xffff0000u);
            const float f6 = __uint_as_float(v3 << 16), f7 = __uint_as_float(v3 & 0xffff0000u);
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (k >= m.T) break;
                const float4* xp = reinterpret_cast<const float4*>(m.xn + (size_t) k * D + j * 8);
                const float4 xa = __ldg(xp), xb = __ldg(xp + 1);
                acc[k] = fmaf(f0, xa.x, fmaf(f1, xa.y, fmaf(f2, xa.z, fmaf(f3, xa.w,
                         fmaf(f4, xb.x, fmaf(f5, xb.y, fmaf(f6, xb.z, fmaf(f7, xb.w, acc[k]))))))));
            }
        }
    } else {
        const GrQ80Block* wb = reinterpret_cast<const GrQ80Block*>(m.a[0].w_down_q8) + (size_t) row * (D / 32);
        const int sub = t & 3;
#pragma unroll 2
        for (int b = t >> 2; b < D / 32; b += 32) {
            const GrQ80Block* blk = wb + b;
            const float d = __half2float(__ldg(&blk->d));
            const uint16_t* q16 = reinterpret_cast<const uint16_t*>(blk->qs + sub * 8);
            const uint32_t w0 = uint32_t(__ldg(q16 + 0)) | (uint32_t(__ldg(q16 + 1)) << 16);
            const uint32_t w1 = uint32_t(__ldg(q16 + 2)) | (uint32_t(__ldg(q16 + 3)) << 16);
            const float q0 = float(int8_t(w0 & 0xffu)), q1 = float(int8_t((w0 >> 8) & 0xffu));
            const float q2 = float(int8_t((w0 >> 16) & 0xffu)), q3 = float(int8_t(w0 >> 24));
            const float q4 = float(int8_t(w1 & 0xffu)), q5 = float(int8_t((w1 >> 8) & 0xffu));
            const float q6 = float(int8_t((w1 >> 16) & 0xffu)), q7 = float(int8_t(w1 >> 24));
            const int xoff = b * 32 + sub * 8;
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (k >= m.T) break;
                const float4* xp = reinterpret_cast<const float4*>(m.xn + (size_t) k * D + xoff);
                const float4 xa = __ldg(xp), xb = __ldg(xp + 1);
                float s = q0 * xa.x;
                s = fmaf(q1, xa.y, s); s = fmaf(q2, xa.z, s); s = fmaf(q3, xa.w, s);
                s = fmaf(q4, xb.x, s); s = fmaf(q5, xb.y, s); s = fmaf(q6, xb.z, s); s = fmaf(q7, xb.w, s);
                acc[k] = fmaf(d, s, acc[k]);
            }
        }
    }

    __shared__ float wsum[4][NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        const float v = warp_sum(acc[k]);
        if (lane == 0) wsum[warp][k] = v;
    }
    __syncthreads();
    if (t < m.T) {
        const float s = wsum[0][t] + wsum[1][t] + wsum[2][t] + wsum[3][t];
        if (inject_block) {
            m.a[t].inject_out[inj_row] = s;
        } else {
            const float x = s * (1.0f / (float) HC);
            m.a[t].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// Fast 2-lane Q8_0 up kernel: 1.88x less HBM2 traffic (3.48 MB vs 6.55 MB), 5 exact blocks/lane (0% idle!), 1 shuffle!
// 320 blocks x 8 cols = 2560; 64 threads/block = 32 rows (4 streams x 8 cols).
template<int NT>
__global__ void __launch_bounds__(64) gr_up_q8_0_multi_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[NT][LR];
    __shared__ float g[NT][HC][8];
    const int t = threadIdx.x;
    const int T = m.T;
    const int d0 = blockIdx.x * 8;
    for (int i = t; i < T * LR; i += 64) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();

    const int r = t >> 1;      // 0..31 -> 4 streams x 8 cols
    const int sub = t & 1;     // 0..1 -> 5 Q8_0 blocks each
    const int c = r >> 3, dd = r & 7;
    const int idx = c * N + d0 + dd;
    const GrQ80Block* wb = reinterpret_cast<const GrQ80Block*>(m.a[0].w_up_q8) + (size_t) idx * (LR / 32);

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int b = sub + j * 2;
        const GrQ80Block* blk = wb + b;
        const float d = __half2float(__ldg(&blk->d));
        const uint16_t* q16 = reinterpret_cast<const uint16_t*>(blk->qs);
        // Load all 32 int8s of this Q8_0 block once for all NT tokens!
        uint32_t wv[8];
#pragma unroll
        for (int u = 0; u < 8; ++u)
            wv[u] = uint32_t(__ldg(q16 + u * 2)) | (uint32_t(__ldg(q16 + u * 2 + 1)) << 16);
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k >= T) break;
            const float4* xp = reinterpret_cast<const float4*>(lo[k] + b * 32);
            float s = 0.0f;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const float4 xv = xp[u];
                const uint32_t wu = wv[u];
                s = fmaf(float(int8_t(wu & 0xffu)), xv.x, s);
                s = fmaf(float(int8_t((wu >> 8) & 0xffu)), xv.y, s);
                s = fmaf(float(int8_t((wu >> 16) & 0xffu)), xv.z, s);
                s = fmaf(float(int8_t(wu >> 24)), xv.w, s);
            }
            acc[k] = fmaf(d, s, acc[k]);
        }
    }

#pragma unroll
    for (int k = 0; k < NT; ++k) {
        if (k >= T) break;
        const float tot = acc[k] + __shfl_xor_sync(0xffffffffu, acc[k], 1);
        if (sub == 0) {
            const FusedGrArgs& a = m.a[k];
            float rv = a.R[idx];
            if (a.apply) {
                rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] * (1.0f / (float) HC)), rv);
                a.R_out[idx] = rv;
            }
            const float x = rv * a.w_norm[idx] * a.rs[c];
            g[k][c][dd] = x * sigmoidf_(tot);
        }
    }
    __syncthreads();
    for (int i = t; i < T * 8; i += 64) {
        const int k = i >> 3, col = i & 7;
        m.a[k].mixed[d0 + col] = (g[k][0][col] + g[k][1][col] + g[k][2][col] + g[k][3][col]) * (1.0f / (float) HC);
    }
}

// Fast 8-lane BF16 up kernel (0% idle lanes: 40 uint4 / 8 lanes = 5 exact!).
// 640 blocks x 4 cols = 2560; 128 threads/block = 16 rows (4 streams x 4 cols).
template<int NT>
__global__ void __launch_bounds__(128) gr_up_bf16_fast_multi_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[NT][LR];
    __shared__ float g[NT][HC][4];
    const int t = threadIdx.x;
    const int T = m.T;
    const int d0 = blockIdx.x * 4;
    for (int i = t; i < T * LR; i += 128) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();

    const int r = t >> 3;      // 0..15 -> 4 streams x 4 cols
    const int sub = t & 7;     // 0..7 -> 5 uint4 each
    const int c = r >> 2, dd = r & 3;
    const int idx = c * N + d0 + dd;
    const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) idx * LR);

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int off = sub + j * 8;
        const uint4 wv = __ldg(w4 + off);
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k >= T) break;
            acc[k] += dot8(wv, lo[k] + off * 8);
        }
    }

#pragma unroll
    for (int k = 0; k < NT; ++k) {
        if (k >= T) break;
        float tot = acc[k];
        tot += __shfl_xor_sync(0xffffffffu, tot, 1);
        tot += __shfl_xor_sync(0xffffffffu, tot, 2);
        tot += __shfl_xor_sync(0xffffffffu, tot, 4);
        if (sub == 0) {
            const FusedGrArgs& a = m.a[k];
            float rv = a.R[idx];
            if (a.apply) {
                rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] * (1.0f / (float) HC)), rv);
                a.R_out[idx] = rv;
            }
            const float x = rv * a.w_norm[idx] * a.rs[c];
            g[k][c][dd] = x * sigmoidf_(tot);
        }
    }
    __syncthreads();
    if (t < T * 4) {
        const int k = t >> 2, col = t & 3;
        m.a[k].mixed[d0 + col] = (g[k][0][col] + g[k][1][col] + g[k][2][col] + g[k][3][col]) * (1.0f / (float) HC);
    }
}

// Norm kernel that stores xn directly as __half2[T][D/2] for K-Major gr_down (half traffic, zero F2F!).
// When WRITE_R is true, also commits updated R_out in the same vectorized float4 pass so gr_up skips R/bo_prev/inj_prev!
// Wave 6: Also fuses the 4-row BF16 w_inject dot product right from registers (ZERO extra xn read!) and stages
// the 4 stream partials into a.lo[c*4 + 0..3] for deterministic reduction in gr_down_q8_soa_km_reduce_kernel!
template<bool WRITE_R>
__global__ void __launch_bounds__(THREADS) gr_norm_h2_multi_kernel(GrMulti m) {
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    __shared__ float inj_part[WARPS][HC];
    const int c = (int) blockIdx.x;
    const int tok = (int) blockIdx.y;
    const FusedGrArgs& a = m.a[tok];
    __half2* xnc = reinterpret_cast<__half2*>(m.xn) + (size_t) tok * (D / 2) + (size_t) c * (N / 2);
    const float* Rc = a.R + (size_t) c * N;
    float* Roc = a.R_out + (size_t) c * N;
    const float* wc = a.w_norm + (size_t) c * N;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss = 0.0f;
    float4 rg[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int d = (t + k * THREADS) * 4;
        if (d < N) {
            float4 r = *reinterpret_cast<const float4*>(Rc + d);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
                if constexpr (WRITE_R) *reinterpret_cast<float4*>(Roc + d) = r;
            }
            const float4 g = *reinterpret_cast<const float4*>(wc + d);
            ss += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            rg[k] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    __syncthreads();
    const float sc = s_rs;
    const bool do_inj = (a.w_inject != nullptr);
    const uint16_t* wi_c = do_inj ? (a.w_inject + (size_t) c * N) : nullptr;
    float inj0 = 0.0f, inj1 = 0.0f, inj2 = 0.0f, inj3 = 0.0f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int d = (t + k * THREADS) * 4;
        if (d < N) {
            const float x0 = rg[k].x * sc, x1 = rg[k].y * sc, x2 = rg[k].z * sc, x3 = rg[k].w * sc;
            const __half2 h0 = __floats2half2_rn(x0, x1);
            const __half2 h1 = __floats2half2_rn(x2, x3);
            *reinterpret_cast<uint2*>(xnc + (d >> 1)) =
                make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
            if (do_inj) {
                auto dot4_bf = [&](const uint16_t* row_c) -> float {
                    const uint2 w2 = __ldg(reinterpret_cast<const uint2*>(row_c + d));
                    float acc = __uint_as_float(w2.x << 16) * x0;
                    acc = fmaf(__uint_as_float(w2.x & 0xffff0000u), x1, acc);
                    acc = fmaf(__uint_as_float(w2.y << 16), x2, acc);
                    acc = fmaf(__uint_as_float(w2.y & 0xffff0000u), x3, acc);
                    return acc;
                };
                inj0 += dot4_bf(wi_c + (size_t) 0 * D);
                inj1 += dot4_bf(wi_c + (size_t) 1 * D);
                inj2 += dot4_bf(wi_c + (size_t) 2 * D);
                inj3 += dot4_bf(wi_c + (size_t) 3 * D);
            }
        }
    }
    if (do_inj) {
        const float v0 = warp_sum(inj0), v1 = warp_sum(inj1), v2 = warp_sum(inj2), v3 = warp_sum(inj3);
        if (lane == 0) {
            inj_part[warp][0] = v0; inj_part[warp][1] = v1;
            inj_part[warp][2] = v2; inj_part[warp][3] = v3;
        }
        __syncthreads();
        if (t < HC) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) s += inj_part[w][t];
            a.lo[c * HC + t] = s;
        }
    }
}

// Wave 6 GP100 K-Major Q8_0_SOA (108) hc_down Split-K16 (dim3(10, 16) = 160 blocks x 320 threads):
// 32 contiguous rows/block (512B = 4 full 128B L2 lines/warp!), 10 inner K-groups x 2 Q8_0 blocks,
// 10.24 KB union SMEM even at T=8 (4 blocks/SM!), LDS.128 broadcast, ZERO inject_row branch, STACK:0!
// Writes 16*T*320 partials into unused upper half of m.xn (m.xn + T * 5120 -> 0 extra VRAM!).
template<int NT>
__global__ void __launch_bounds__(320) gr_down_q8_soa_km_splitk_kernel(GrMulti m) {
    union GrDownKmSmem {
        __align__(16) __half2 sxh[NT][320];
        float sp[10][NT][32];
    };
    __shared__ GrDownKmSmem smem;

    const int tid = threadIdx.x;
    const int r_loc = tid & 31;
    const int kg = tid >> 5;
    const int row = (int) blockIdx.x * 32 + r_loc;
    const int kp = (int) blockIdx.y;
    const int b_chunk = kp * 20;

    const __half2* xh2 = reinterpret_cast<const __half2*>(m.xn);
    for (int i = tid; i < NT * 80; i += 320) {
        const int k = i / 80;
        const int j = i - k * 80;
        reinterpret_cast<uint4*>(smem.sxh[k])[j] =
            __ldg(reinterpret_cast<const uint4*>(xh2 + (size_t) k * (D / 2) + b_chunk * 16) + j);
    }
    __syncthreads();

    const uint4* wq = reinterpret_cast<const uint4*>(m.a[0].w_down_q8);
    const __half* wd = reinterpret_cast<const __half*>(
        reinterpret_cast<const uint8_t*>(m.a[0].w_down_q8) + (size_t) LR * D);

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

#pragma unroll
    for (int it = 0; it < 2; ++it) {
        const int db = kg + it * 10;
        const int b = b_chunk + db;
        const float d = __half2float(__ldg(wd + (size_t) b * LR + row));
        __half2 p[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) p[k] = __float2half2_rn(0.0f);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 qv = __ldg(wq + (size_t) (b * 2 + h) * LR + row);
            const int xb = db * 16 + h * 8;
            const __half2 w0 = gr_u8_lo_h2(qv.x), w1 = gr_u8_hi_h2(qv.x);
            const __half2 w2 = gr_u8_lo_h2(qv.y), w3 = gr_u8_hi_h2(qv.y);
            const __half2 w4 = gr_u8_lo_h2(qv.z), w5 = gr_u8_hi_h2(qv.z);
            const __half2 w6 = gr_u8_lo_h2(qv.w), w7 = gr_u8_hi_h2(qv.w);
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const uint4 xa = *reinterpret_cast<const uint4*>(&smem.sxh[k][xb + 0]);
                const uint4 xb4 = *reinterpret_cast<const uint4*>(&smem.sxh[k][xb + 4]);
                __half2 pk = p[k];
                pk = __hfma2(w0, gr_u32_to_h2(xa.x), pk); pk = __hfma2(w1, gr_u32_to_h2(xa.y), pk);
                pk = __hfma2(w2, gr_u32_to_h2(xa.z), pk); pk = __hfma2(w3, gr_u32_to_h2(xa.w), pk);
                pk = __hfma2(w4, gr_u32_to_h2(xb4.x), pk); pk = __hfma2(w5, gr_u32_to_h2(xb4.y), pk);
                pk = __hfma2(w6, gr_u32_to_h2(xb4.z), pk); pk = __hfma2(w7, gr_u32_to_h2(xb4.w), pk);
                p[k] = pk;
            }
        }
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const float2 f = __half22float2(p[k]);
            acc[k] = fmaf(d, f.x + f.y, acc[k]);
        }
    }

    __syncthreads();
#pragma unroll
    for (int k = 0; k < NT; ++k) smem.sp[kg][k][r_loc] = acc[k];
    __syncthreads();

    if (kg < NT) {
        const float s =
            smem.sp[0][kg][r_loc] + smem.sp[1][kg][r_loc] + smem.sp[2][kg][r_loc] + smem.sp[3][kg][r_loc] +
            smem.sp[4][kg][r_loc] + smem.sp[5][kg][r_loc] + smem.sp[6][kg][r_loc] + smem.sp[7][kg][r_loc] +
            smem.sp[8][kg][r_loc] + smem.sp[9][kg][r_loc];
        float* partials = m.xn + (size_t) NT * (D / 2);
        partials[((size_t) kp * NT + kg) * LR + row] = s;
    }
}

// Reduces the 16 Split-K partials -> a.lo[0..319] AND finishes the 4-stream inject_out[0..3] sum in blockIdx.x==0!
__global__ void __launch_bounds__(32) gr_down_q8_soa_km_reduce_kernel(GrMulti m) {
    const int tid = threadIdx.x;
    const int tok = (int) blockIdx.y;
    const int row = (int) blockIdx.x * 32 + tid;
    const FusedGrArgs& a = m.a[tok];
    if (blockIdx.x == 0 && a.w_inject != nullptr) {
        if (tid < HC) {
            a.inject_out[tid] = a.lo[0 * HC + tid] + a.lo[1 * HC + tid] + a.lo[2 * HC + tid] + a.lo[3 * HC + tid];
        }
        __syncwarp();
    }
    const size_t stride = (size_t) m.T * LR;
    const float* p = m.xn + (size_t) m.T * (D / 2) + (size_t) tok * LR + row;
    float s = 0.0f;
#pragma unroll
    for (int kp = 0; kp < 16; ++kp) s += p[(size_t) kp * stride];
    const float x = s * (1.0f / (float) HC);
    a.lo[row] = x / (1.0f + __expf(-x));
}

// GP100 Row-Major SoA Q8_0 hc_down: 100% coalesced LDG.E.128 + HFMA2 -> ~11.9 us (2.0x faster than Build 4!).
template<int NT>
__global__ void __launch_bounds__(128) gr_down_q8_soa_rm_multi_kernel(GrMulti m) {
    const int row = blockIdx.x;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const bool inject_block = row >= LR;
    const int inj_row = row - LR;
    if (inject_block && (m.a[0].w_inject == nullptr || inj_row >= HC)) return;

    const __half2* xh2 = reinterpret_cast<const __half2*>(m.xn);
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

    if (inject_block) {
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_inject + (size_t) inj_row * D);
        for (int j = t; j < D / 8; j += 128) {
            const uint4 wv = __ldg(w4 + j);
            const float f0 = __uint_as_float(wv.x << 16), f1 = __uint_as_float(wv.x & 0xffff0000u);
            const float f2 = __uint_as_float(wv.y << 16), f3 = __uint_as_float(wv.y & 0xffff0000u);
            const float f4 = __uint_as_float(wv.z << 16), f5 = __uint_as_float(wv.z & 0xffff0000u);
            const float f6 = __uint_as_float(wv.w << 16), f7 = __uint_as_float(wv.w & 0xffff0000u);
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (k >= m.T) break;
                const uint4 xv = __ldg(reinterpret_cast<const uint4*>(xh2 + (size_t) k * (D / 2) + j * 4));
                const __half2* hp = reinterpret_cast<const __half2*>(&xv);
                const float2 a0 = __half22float2(hp[0]), a1 = __half22float2(hp[1]);
                const float2 a2 = __half22float2(hp[2]), a3 = __half22float2(hp[3]);
                acc[k] = fmaf(f0, a0.x, fmaf(f1, a0.y, fmaf(f2, a1.x, fmaf(f3, a1.y,
                         fmaf(f4, a2.x, fmaf(f5, a2.y, fmaf(f6, a3.x, fmaf(f7, a3.y, acc[k]))))))));
            }
        }
    } else {
        const uint4* wq_row = reinterpret_cast<const uint4*>(m.a[0].w_down_q8) + (size_t) row * (D / 16);
        const __half* wd_row = reinterpret_cast<const __half*>(
            reinterpret_cast<const uint8_t*>(m.a[0].w_down_q8) + (size_t) LR * D) + (size_t) row * (D / 32);
#pragma unroll
        for (int it = 0; it < 5; ++it) {
            const int idx = it * 128 + t;
            const int b = idx >> 1;
            const uint4 qv = __ldg(wq_row + idx);
            const float d = __half2float(__ldg(wd_row + b));
            const __half2 w0 = gr_u8_lo_h2(qv.x), w1 = gr_u8_hi_h2(qv.x);
            const __half2 w2 = gr_u8_lo_h2(qv.y), w3 = gr_u8_hi_h2(qv.y);
            const __half2 w4 = gr_u8_lo_h2(qv.z), w5 = gr_u8_hi_h2(qv.z);
            const __half2 w6 = gr_u8_lo_h2(qv.w), w7 = gr_u8_hi_h2(qv.w);
            const int xoff = idx * 8;
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (k >= m.T) break;
                const uint4 xa = __ldg(reinterpret_cast<const uint4*>(xh2 + (size_t) k * (D / 2) + xoff));
                const uint4 xb = __ldg(reinterpret_cast<const uint4*>(xh2 + (size_t) k * (D / 2) + xoff + 4));
                const __half2* ha = reinterpret_cast<const __half2*>(&xa);
                const __half2* hb = reinterpret_cast<const __half2*>(&xb);
                __half2 p = __hmul2(w0, ha[0]);
                p = __hfma2(w1, ha[1], p); p = __hfma2(w2, ha[2], p); p = __hfma2(w3, ha[3], p);
                p = __hfma2(w4, hb[0], p); p = __hfma2(w5, hb[1], p); p = __hfma2(w6, hb[2], p); p = __hfma2(w7, hb[3], p);
                const float2 f = __half22float2(p);
                acc[k] = fmaf(d, f.x + f.y, acc[k]);
            }
        }
    }

    __shared__ float ws[NT][4];
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        if (k >= m.T) break;
        const float s = warp_sum(acc[k]);
        if (lane == 0) ws[k][warp] = s;
    }
    __syncthreads();
    if (t < m.T) {
        const float s = ws[t][0] + ws[t][1] + ws[t][2] + ws[t][3];
        if (inject_block) {
            m.a[t].inject_out[inj_row] = s;
        } else {
            const float x = s * (1.0f / (float) HC);
            m.a[t].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// GP100 SMEM-Tiled 6-Row-Per-Block Q8_0 SoA hc_down (54 blocks x 768 threads = 24 warps/SM, 0 tail on 56 SMs!):
// Loads xh2 into __shared__[NT][1024] once per 6 rows -> 6x LESS L2 cache traffic (51.8 MB -> 8.64 MB)!
template<int NT>
__global__ void __launch_bounds__(768, 1) gr_down_q8_soa_rm_tiled_kernel(GrMulti m) {
    constexpr int TILE_H2 = 1024;   // 2048 elements = 5 exact tiles across D=10240 (128 uint4/row/tile)
    union GrDownSmem {
        __align__(16) __half2 sxh[NT][TILE_H2];
        float ws[6][NT][4];
    };
    __shared__ GrDownSmem smem;

    const int tid = threadIdx.x;
    const int rg = tid >> 7;         // 0..5 (6 rows per block)
    const int rl = tid & 127;        // 0..127 (4 warps per row)
    const int warp = rl >> 5;        // 0..3
    const int lane = rl & 31;
    const int row = (int) blockIdx.x * 6 + rg;
    const bool inject_row = row >= LR;
    const int inj_row = row - LR;
    const bool active = !inject_row || (m.a[0].w_inject != nullptr && inj_row < HC);

    const __half2* xh2 = reinterpret_cast<const __half2*>(m.xn);
    const uint4* wq_row = (!inject_row) ? (reinterpret_cast<const uint4*>(m.a[0].w_down_q8) + (size_t) row * (D / 16)) : nullptr;
    const __half* wd_row = (!inject_row) ? (reinterpret_cast<const __half*>(
        reinterpret_cast<const uint8_t*>(m.a[0].w_down_q8) + (size_t) LR * D) + (size_t) row * (D / 32)) : nullptr;
    const uint4* winj_row = (inject_row && active)
        ? reinterpret_cast<const uint4*>(m.a[0].w_inject + (size_t) inj_row * D) : nullptr;

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

    for (int it = 0; it < 5; ++it) {
        const int base_h2 = it * TILE_H2;
        for (int i = tid; i < NT * (TILE_H2 / 4); i += 768) {
            const int k = i >> 8;
            const int j = i & 255;
            reinterpret_cast<uint4*>(smem.sxh[k])[j] =
                __ldg(reinterpret_cast<const uint4*>(xh2 + (size_t) k * (D / 2) + base_h2) + j);
        }
        __syncthreads();

        if (active) {
            if (!inject_row) {
                const int idx = it * 128 + rl;
                const int b = idx >> 1;
                const uint4 qv = __ldg(wq_row + idx);
                const float d = __half2float(__ldg(wd_row + b));
                const __half2 w0 = gr_u8_lo_h2(qv.x), w1 = gr_u8_hi_h2(qv.x);
                const __half2 w2 = gr_u8_lo_h2(qv.y), w3 = gr_u8_hi_h2(qv.y);
                const __half2 w4 = gr_u8_lo_h2(qv.z), w5 = gr_u8_hi_h2(qv.z);
                const __half2 w6 = gr_u8_lo_h2(qv.w), w7 = gr_u8_hi_h2(qv.w);
                const int xoff = rl * 8;
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    const uint4 xa = *reinterpret_cast<const uint4*>(&smem.sxh[k][xoff + 0]);
                    const uint4 xb = *reinterpret_cast<const uint4*>(&smem.sxh[k][xoff + 4]);
                    const __half2* ha = reinterpret_cast<const __half2*>(&xa);
                    const __half2* hb = reinterpret_cast<const __half2*>(&xb);
                    __half2 p = __hmul2(w0, ha[0]);
                    p = __hfma2(w1, ha[1], p); p = __hfma2(w2, ha[2], p); p = __hfma2(w3, ha[3], p);
                    p = __hfma2(w4, hb[0], p); p = __hfma2(w5, hb[1], p); p = __hfma2(w6, hb[2], p); p = __hfma2(w7, hb[3], p);
                    const float2 f = __half22float2(p);
                    acc[k] = fmaf(d, f.x + f.y, acc[k]);
                }
            } else {
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const uint4 wv = __ldg(winj_row + it * 256 + rl * 2 + u);
                    const float f0 = __uint_as_float(wv.x << 16), f1 = __uint_as_float(wv.x & 0xffff0000u);
                    const float f2 = __uint_as_float(wv.y << 16), f3 = __uint_as_float(wv.y & 0xffff0000u);
                    const float f4 = __uint_as_float(wv.z << 16), f5 = __uint_as_float(wv.z & 0xffff0000u);
                    const float f6 = __uint_as_float(wv.w << 16), f7 = __uint_as_float(wv.w & 0xffff0000u);
                    const int xoff = rl * 8 + u * 4;
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(&smem.sxh[k][xoff]);
                        const __half2* hp = reinterpret_cast<const __half2*>(&xv);
                        const float2 a0 = __half22float2(hp[0]), a1 = __half22float2(hp[1]);
                        const float2 a2 = __half22float2(hp[2]), a3 = __half22float2(hp[3]);
                        acc[k] = fmaf(f0, a0.x, fmaf(f1, a0.y, fmaf(f2, a1.x, fmaf(f3, a1.y,
                                 fmaf(f4, a2.x, fmaf(f5, a2.y, fmaf(f6, a3.x, fmaf(f7, a3.y, acc[k]))))))));
                    }
                }
            }
        }
        __syncthreads();
    }

    if (active) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const float s = warp_sum(acc[k]);
            if (lane == 0) smem.ws[rg][k][warp] = s;
        }
    }
    __syncthreads();
    if (active && rl < NT) {
        const float s = smem.ws[rg][rl][0] + smem.ws[rg][rl][1] + smem.ws[rg][rl][2] + smem.ws[rg][rl][3];
        if (inject_row) {
            m.a[rl].inject_out[inj_row] = s;
        } else {
            const float x = s * (1.0f / (float) HC);
            m.a[rl].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// GP100 K-Major SoA Q8_0 hc_up Split-K5: 160 blocks x 320 threads (5 K-groups x 64 lanes = 28.6 warps/SM, 5x higher occupancy!),
// 100% coalesced 256B half-warp LDG.E.128, LDS.128 broadcast on slo (318 GB/s COLD HBM2, 10.95 us vs 12.87 us!).
template<int NT, bool R_DONE>
__global__ void __launch_bounds__(320) gr_up_q8_soa_km_multi_kernel(GrMulti m) {
    __shared__ __align__(16) __half2 slo[NT][LR / 2];
    __shared__ float sp[5][NT][64];
    const int t = threadIdx.x;
    const int T = m.T;
    const int d0 = blockIdx.x * 16;
    for (int i = t; i < T * (LR / 2); i += 320) {
        const int k = i / (LR / 2), j = i - k * (LR / 2);
        slo[k][j] = __floats2half2_rn(m.a[k].lo[2 * j], m.a[k].lo[2 * j + 1]);
    }
    __syncthreads();

    const int grp = t >> 6;   // 0..4 (5 K-groups -> 2 Q8_0 blocks each for LR=320!)
    const int sub = t & 63;   // 0..63
    const int c = sub >> 4;   // 0..3
    const int dd = sub & 15;  // 0..15 (16 consecutive uint4 = 256B = 2 full cache lines!)
    const int r = c * N + d0 + dd;
    const uint4* wq = reinterpret_cast<const uint4*>(m.a[0].w_up_q8);
    const __half* wd = reinterpret_cast<const __half*>(
        reinterpret_cast<const uint8_t*>(m.a[0].w_up_q8) + (size_t) D * LR);

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

#pragma unroll
    for (int it = 0; it < (LR / 32) / 5; ++it) {
        const int b = grp + it * 5;
        const float d = __half2float(__ldg(wd + (size_t) b * D + r));
        __half2 p[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) p[k] = __float2half2_rn(0.0f);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 qv = __ldg(wq + (size_t) (b * 2 + h) * D + r);
            const int xb = b * 16 + h * 8;
            const __half2 w0 = gr_u8_lo_h2(qv.x), w1 = gr_u8_hi_h2(qv.x);
            const __half2 w2 = gr_u8_lo_h2(qv.y), w3 = gr_u8_hi_h2(qv.y);
            const __half2 w4 = gr_u8_lo_h2(qv.z), w5 = gr_u8_hi_h2(qv.z);
            const __half2 w6 = gr_u8_lo_h2(qv.w), w7 = gr_u8_hi_h2(qv.w);
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (k >= T) break;
                const uint4 xa = *reinterpret_cast<const uint4*>(&slo[k][xb + 0]);
                const uint4 xb4 = *reinterpret_cast<const uint4*>(&slo[k][xb + 4]);
                __half2 pk = p[k];
                pk = __hfma2(w0, gr_u32_to_h2(xa.x), pk); pk = __hfma2(w1, gr_u32_to_h2(xa.y), pk);
                pk = __hfma2(w2, gr_u32_to_h2(xa.z), pk); pk = __hfma2(w3, gr_u32_to_h2(xa.w), pk);
                pk = __hfma2(w4, gr_u32_to_h2(xb4.x), pk); pk = __hfma2(w5, gr_u32_to_h2(xb4.y), pk);
                pk = __hfma2(w6, gr_u32_to_h2(xb4.z), pk); pk = __hfma2(w7, gr_u32_to_h2(xb4.w), pk);
                p[k] = pk;
            }
        }
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k >= T) break;
            const float2 f = __half22float2(p[k]);
            acc[k] = fmaf(d, f.x + f.y, acc[k]);
        }
    }

#pragma unroll
    for (int k = 0; k < NT; ++k) {
        if (k >= T) break;
        sp[grp][k][sub] = acc[k];
    }
    __syncthreads();
    if (grp == 0) {
        const float wnr = __ldg(m.a[0].w_norm + r);
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k >= T) break;
            const FusedGrArgs& a = m.a[k];
            const float tot = sp[0][k][sub] + sp[1][k][sub] + sp[2][k][sub] + sp[3][k][sub] + sp[4][k][sub];
            float rv;
            if constexpr (R_DONE) {
                rv = __ldg((a.apply ? a.R_out : a.R) + r);
            } else {
                rv = a.R[r];
                if (a.apply) {
                    rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] * (1.0f / (float) HC)), rv);
                    a.R_out[r] = rv;
                }
            }
            sp[0][k][sub] = rv * wnr * a.rs[c] * sigmoidf_(tot);
        }
    }
    __syncthreads();
    for (int i = t; i < T * 16; i += 320) {
        const int k = i >> 4, col = i & 15;
        m.a[k].mixed[d0 + col] = (sp[0][k][col] + sp[0][k][16 + col] + sp[0][k][32 + col] + sp[0][k][48 + col]) * (1.0f / (float) HC);
    }
}

struct GrQ81Block {
    __half2 ds;
    int8_t qs[32];
};

// GP100 K-Major SoA Q8_0 hc_up Split-K5 + FUSED Q8_1 Quantization (320t = 5 K-groups x 64 threads, 2 streams/thread,
// union SMEM <= 20.5 KB -> 2-3 blocks/SM even at T=8!, parallel 5-group epilogue, STACK:0! 13.92 us vs 15.15 us!).
template<int NT, bool R_DONE>
__global__ void __launch_bounds__(320) gr_up_q8_soa_km_quant_kernel(GrMulti m) {
    union GrUpQuantSmem {
        __align__(16) __half2 slo[NT][LR / 2];
        float sp[5][NT][128];
    };
    __shared__ GrUpQuantSmem smem;
    const int t = threadIdx.x;
    const int T = m.T;
    const int d0 = (int) blockIdx.x * 32;
    for (int i = t; i < T * (LR / 2); i += 320) {
        const int k = i / (LR / 2), j = i - k * (LR / 2);
        smem.slo[k][j] = __floats2half2_rn(m.a[k].lo[2 * j], m.a[k].lo[2 * j + 1]);
    }
    __syncthreads();

    const int grp = t >> 6;     // 0..4 (5 K-groups -> 2 Q8_0 blocks each for LR=320)
    const int sub = t & 63;     // 0..63 (c0 in {0, 1} x 32 cols; handles stream c0 and stream c0+2!)
    const int c0 = sub >> 5;    // 0..1
    const int dd = sub & 31;    // 0..31 (32 consecutive uint4 = 512B = 4 full 128B cache lines!)
    const int r0 = c0 * N + d0 + dd;
    const int r1 = (c0 + 2) * N + d0 + dd;
    const uint4* wq = reinterpret_cast<const uint4*>(m.a[0].w_up_q8);
    const __half* wd = reinterpret_cast<const __half*>(
        reinterpret_cast<const uint8_t*>(m.a[0].w_up_q8) + (size_t) D * LR);

    float a0[NT], a1[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) { a0[k] = 0.0f; a1[k] = 0.0f; }

#pragma unroll
    for (int it = 0; it < (LR / 32) / 5; ++it) {
        const int b = grp + it * 5;
        const float d0v = __half2float(__ldg(wd + (size_t) b * D + r0));
        const float d1v = __half2float(__ldg(wd + (size_t) b * D + r1));
        __half2 p0[NT], p1[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) { p0[k] = __float2half2_rn(0.0f); p1[k] = __float2half2_rn(0.0f); }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 q0 = __ldg(wq + (size_t) (b * 2 + h) * D + r0);
            const uint4 q1 = __ldg(wq + (size_t) (b * 2 + h) * D + r1);
            const int xb = b * 16 + h * 8;
            const __half2 u0 = gr_u8_lo_h2(q0.x), u1 = gr_u8_hi_h2(q0.x);
            const __half2 u2 = gr_u8_lo_h2(q0.y), u3 = gr_u8_hi_h2(q0.y);
            const __half2 u4 = gr_u8_lo_h2(q0.z), u5 = gr_u8_hi_h2(q0.z);
            const __half2 u6 = gr_u8_lo_h2(q0.w), u7 = gr_u8_hi_h2(q0.w);
            const __half2 v0 = gr_u8_lo_h2(q1.x), v1 = gr_u8_hi_h2(q1.x);
            const __half2 v2 = gr_u8_lo_h2(q1.y), v3 = gr_u8_hi_h2(q1.y);
            const __half2 v4 = gr_u8_lo_h2(q1.z), v5 = gr_u8_hi_h2(q1.z);
            const __half2 v6 = gr_u8_lo_h2(q1.w), v7 = gr_u8_hi_h2(q1.w);
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (k >= T) break;
                const uint4 xa = *reinterpret_cast<const uint4*>(&smem.slo[k][xb + 0]);
                const uint4 xb4 = *reinterpret_cast<const uint4*>(&smem.slo[k][xb + 4]);
                const __half2 x0 = gr_u32_to_h2(xa.x), x1 = gr_u32_to_h2(xa.y);
                const __half2 x2 = gr_u32_to_h2(xa.z), x3 = gr_u32_to_h2(xa.w);
                const __half2 x4 = gr_u32_to_h2(xb4.x), x5 = gr_u32_to_h2(xb4.y);
                const __half2 x6 = gr_u32_to_h2(xb4.z), x7 = gr_u32_to_h2(xb4.w);
                __half2 pk0 = p0[k], pk1 = p1[k];
                pk0 = __hfma2(u0, x0, pk0); pk0 = __hfma2(u1, x1, pk0);
                pk0 = __hfma2(u2, x2, pk0); pk0 = __hfma2(u3, x3, pk0);
                pk0 = __hfma2(u4, x4, pk0); pk0 = __hfma2(u5, x5, pk0);
                pk0 = __hfma2(u6, x6, pk0); pk0 = __hfma2(u7, x7, pk0);
                pk1 = __hfma2(v0, x0, pk1); pk1 = __hfma2(v1, x1, pk1);
                pk1 = __hfma2(v2, x2, pk1); pk1 = __hfma2(v3, x3, pk1);
                pk1 = __hfma2(v4, x4, pk1); pk1 = __hfma2(v5, x5, pk1);
                pk1 = __hfma2(v6, x6, pk1); pk1 = __hfma2(v7, x7, pk1);
                p0[k] = pk0; p1[k] = pk1;
            }
        }
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k >= T) break;
            const float2 f0 = __half22float2(p0[k]);
            const float2 f1 = __half22float2(p1[k]);
            a0[k] = fmaf(d0v, f0.x + f0.y, a0[k]);
            a1[k] = fmaf(d1v, f1.x + f1.y, a1[k]);
        }
    }

    __syncthreads();
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        if (k < T) {
            smem.sp[grp][k][sub] = a0[k];
            smem.sp[grp][k][sub + 64] = a1[k];
        }
    }
    __syncthreads();

    // Parallel epilogue across all 5 groups (each group handles tokens k = grp, grp + 5, ...)
    const float wn0 = __ldg(m.a[0].w_norm + r0);
    const float wn1 = __ldg(m.a[0].w_norm + r1);
    for (int k = grp; k < T; k += 5) {
        const FusedGrArgs& a = m.a[k];
        const float tot0 = smem.sp[0][k][sub] + smem.sp[1][k][sub] + smem.sp[2][k][sub] + smem.sp[3][k][sub] + smem.sp[4][k][sub];
        const float tot1 = smem.sp[0][k][sub + 64] + smem.sp[1][k][sub + 64] + smem.sp[2][k][sub + 64] + smem.sp[3][k][sub + 64] + smem.sp[4][k][sub + 64];
        float rv0, rv1;
        if constexpr (R_DONE) {
            const float* Rsrc = a.apply ? a.R_out : a.R;
            rv0 = __ldg(Rsrc + r0);
            rv1 = __ldg(Rsrc + r1);
        } else {
            rv0 = a.R[r0];
            rv1 = a.R[r1];
            if (a.apply) {
                const float bo = a.bo_prev[d0 + dd];
                rv0 = fmaf(bo, 2.0f * sigmoidf_(a.inj_prev[c0] * (1.0f / (float) HC)), rv0);
                rv1 = fmaf(bo, 2.0f * sigmoidf_(a.inj_prev[c0 + 2] * (1.0f / (float) HC)), rv1);
                a.R_out[r0] = rv0;
                a.R_out[r1] = rv1;
            }
        }
        smem.sp[0][k][sub]      = rv0 * wn0 * a.rs[c0]     * sigmoidf_(tot0);
        smem.sp[0][k][sub + 64] = rv1 * wn1 * a.rs[c0 + 2] * sigmoidf_(tot1);
    }
    __syncthreads();

    // Final write: 1 warp per token (T <= 8 <= 10 warps) writes BOTH `mixed` AND Q8_1 block!
    const int k = t >> 5;
    const int lane = t & 31;
    if (k < T) {
        const float xi = (smem.sp[0][k][lane] + smem.sp[0][k][32 + lane] + smem.sp[0][k][64 + lane] + smem.sp[0][k][96 + lane]) *
                         (1.0f / (float) HC);
        m.a[k].mixed[d0 + lane] = xi;
        float amax = fabsf(xi), sum = xi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            sum += __shfl_xor_sync(0xffffffffu, sum, o);
        }
        const float d = fminf(amax / 127.0f, 65504.0f);
        const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(xi / d)));
        GrQ81Block* bq = reinterpret_cast<GrQ81Block*>(m.a[0].mixed_q8) + (size_t) k * (N / 32) + blockIdx.x;
        bq->qs[lane] = q;
        if (lane == 0) bq->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
    }
}

}  // namespace

void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, unsigned long long* stamp_buf,
                         int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || (!x.w_down && !x.w_down_q8) || (!x.w_up && !x.w_up_q8) || !x.lo || !x.rs || !x.mixed ||
            (x.w_inject && !x.inject_out) || (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) ||
            x.w_down != a[0].w_down || x.w_up != a[0].w_up || x.w_down_q8 != a[0].w_down_q8 ||
            x.w_up_q8 != a[0].w_up_q8 || x.w_down_q8_soa != a[0].w_down_q8_soa ||
            x.w_up_q8_soa != a[0].w_up_q8_soa || x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    m.xn = xn_scratch;
    m.T = n_tok;
    cudaStream_t st = (cudaStream_t) stream;
    const bool down_soa = (a[0].w_down_q8 != nullptr && a[0].w_down_q8_soa);
    const bool up_soa = (a[0].w_up_q8 != nullptr && a[0].w_up_q8_soa);
    const bool write_r_in_norm = down_soa && up_soa;
    if (down_soa) {
        if (write_r_in_norm) gr_norm_h2_multi_kernel<true><<<dim3(HC, (unsigned) n_tok), THREADS, 0, st>>>(m);
        else gr_norm_h2_multi_kernel<false><<<dim3(HC, (unsigned) n_tok), THREADS, 0, st>>>(m);
    } else {
        gr_norm_multi_kernel<<<dim3(HC, (unsigned) n_tok), THREADS, 0, st>>>(m);
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
    if (a[0].w_down_q8 != nullptr && a[0].w_down_q8_soa) {
        const dim3 g_sk(LR / 32, 16);   // 10 x 16 = 160 blocks x 320 threads
        if (n_tok == 1) gr_down_q8_soa_km_splitk_kernel<1><<<g_sk, 320, 0, st>>>(m);
        else if (n_tok == 2) gr_down_q8_soa_km_splitk_kernel<2><<<g_sk, 320, 0, st>>>(m);
        else if (n_tok == 3) gr_down_q8_soa_km_splitk_kernel<3><<<g_sk, 320, 0, st>>>(m);
        else if (n_tok == 4) gr_down_q8_soa_km_splitk_kernel<4><<<g_sk, 320, 0, st>>>(m);
        else if (n_tok == 5) gr_down_q8_soa_km_splitk_kernel<5><<<g_sk, 320, 0, st>>>(m);
        else if (n_tok == 6) gr_down_q8_soa_km_splitk_kernel<6><<<g_sk, 320, 0, st>>>(m);
        else if (n_tok == 7) gr_down_q8_soa_km_splitk_kernel<7><<<g_sk, 320, 0, st>>>(m);
        else gr_down_q8_soa_km_splitk_kernel<kFusedGrMaxT><<<g_sk, 320, 0, st>>>(m);
        gr_down_q8_soa_km_reduce_kernel<<<dim3(LR / 32, (unsigned) n_tok), 32, 0, st>>>(m);
    } else if (a[0].w_down_q8 != nullptr) {
        if (n_tok == 1) gr_down_q8_0_multi_kernel<1><<<LR + HC, 128, 0, st>>>(m);
        else if (n_tok == 2) gr_down_q8_0_multi_kernel<2><<<LR + HC, 128, 0, st>>>(m);
        else if (n_tok <= 4) gr_down_q8_0_multi_kernel<4><<<LR + HC, 128, 0, st>>>(m);
        else gr_down_q8_0_multi_kernel<kFusedGrMaxT><<<LR + HC, 128, 0, st>>>(m);
    } else if (n_tok == 1) {
        gr_down_direct_multi_kernel<1><<<LR + HC, 128, 0, st>>>(m);
    } else if (n_tok == 2) {
        gr_down_direct_multi_kernel<2><<<LR + HC, 128, 0, st>>>(m);
    } else if (n_tok == 3) {
        gr_down_direct_multi_kernel<3><<<LR + HC, 128, 0, st>>>(m);
    } else if (n_tok == 4) {
        gr_down_direct_multi_kernel<4><<<LR + HC, 128, 0, st>>>(m);
    } else if (n_tok == 5) {
        gr_down_direct_multi_kernel<5><<<LR + HC, 128, 0, st>>>(m);
    } else if (n_tok == 6) {
        gr_down_direct_multi_kernel<6><<<LR + HC, 128, 0, st>>>(m);
    } else if (n_tok == 7) {
        gr_down_direct_multi_kernel<7><<<LR + HC, 128, 0, st>>>(m);
    } else if (n_tok == 8) {
        gr_down_direct_multi_kernel<8><<<LR + HC, 128, 0, st>>>(m);
    } else {
    // the shared-memory opt-in is a per-DEVICE setting: once per device, not once per process (a layer split
    // runs this kernel on two cards)
    static bool attr[64] = {};
    static int chunk[64] = {};   // Turing port: tokens the down kernel may carry in one launch on this card
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 64 && !attr[dev]) {
        // at most what the card allows (Turing: 64 KB - enough for windows of up to 6 tokens)
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        int want = (int) (kFusedGrMaxT * TILE * sizeof(float));
        if (optin > 0 && want > optin) want = optin;
        cudaFuncSetAttribute(gr_down_multi_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        cudaGetLastError();      // drop any error the attempt left behind
        const int capacity = (optin > 0 ? optin : 48 * 1024) / (int) (TILE * sizeof(float));
        chunk[dev] = capacity < 1 ? 1 : (capacity > kFusedGrMaxT ? kFusedGrMaxT : capacity);
        attr[dev] = true;
    }
    const int chunk_tok = (dev >= 0 && dev < 64 && chunk[dev]) ? chunk[dev] : kFusedGrMaxT;
    if (chunk_tok >= n_tok) {
        gr_down_multi_kernel<<<DOWN_BLOCKS + 1, THREADS, (size_t) n_tok * TILE * sizeof(float), st>>>(m);
    } else {
        for (int c0 = 0; c0 < n_tok; c0 += chunk_tok) {
            const int ct = n_tok - c0 < chunk_tok ? n_tok - c0 : chunk_tok;
            GrMulti c{};
            c.xn = xn_scratch + (size_t) c0 * D;
            c.T = ct;
            for (int k = 0; k < ct; ++k) c.a[k] = a[c0 + k];
            gr_down_multi_kernel<<<DOWN_BLOCKS + 1, THREADS, (size_t) ct * TILE * sizeof(float), st>>>(c);
        }
    }
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
    if (up_soa && a[0].mixed_q8 != nullptr) {
        if (write_r_in_norm) {
            if (n_tok == 1) gr_up_q8_soa_km_quant_kernel<1, true><<<N / 32, 320, 0, st>>>(m);
            else if (n_tok == 2) gr_up_q8_soa_km_quant_kernel<2, true><<<N / 32, 320, 0, st>>>(m);
            else if (n_tok == 3) gr_up_q8_soa_km_quant_kernel<3, true><<<N / 32, 320, 0, st>>>(m);
            else if (n_tok == 4) gr_up_q8_soa_km_quant_kernel<4, true><<<N / 32, 320, 0, st>>>(m);
            else if (n_tok == 5) gr_up_q8_soa_km_quant_kernel<5, true><<<N / 32, 320, 0, st>>>(m);
            else if (n_tok == 6) gr_up_q8_soa_km_quant_kernel<6, true><<<N / 32, 320, 0, st>>>(m);
            else if (n_tok == 7) gr_up_q8_soa_km_quant_kernel<7, true><<<N / 32, 320, 0, st>>>(m);
            else gr_up_q8_soa_km_quant_kernel<kFusedGrMaxT, true><<<N / 32, 320, 0, st>>>(m);
        } else {
            if (n_tok == 1) gr_up_q8_soa_km_quant_kernel<1, false><<<N / 32, 320, 0, st>>>(m);
            else if (n_tok == 2) gr_up_q8_soa_km_quant_kernel<2, false><<<N / 32, 320, 0, st>>>(m);
            else if (n_tok == 3) gr_up_q8_soa_km_quant_kernel<3, false><<<N / 32, 320, 0, st>>>(m);
            else if (n_tok == 4) gr_up_q8_soa_km_quant_kernel<4, false><<<N / 32, 320, 0, st>>>(m);
            else if (n_tok == 5) gr_up_q8_soa_km_quant_kernel<5, false><<<N / 32, 320, 0, st>>>(m);
            else if (n_tok == 6) gr_up_q8_soa_km_quant_kernel<6, false><<<N / 32, 320, 0, st>>>(m);
            else if (n_tok == 7) gr_up_q8_soa_km_quant_kernel<7, false><<<N / 32, 320, 0, st>>>(m);
            else gr_up_q8_soa_km_quant_kernel<kFusedGrMaxT, false><<<N / 32, 320, 0, st>>>(m);
        }
    } else if (up_soa) {
        if (write_r_in_norm) {
            if (n_tok == 1) gr_up_q8_soa_km_multi_kernel<1, true><<<N / 16, 320, 0, st>>>(m);
            else if (n_tok == 2) gr_up_q8_soa_km_multi_kernel<2, true><<<N / 16, 320, 0, st>>>(m);
            else if (n_tok == 3) gr_up_q8_soa_km_multi_kernel<3, true><<<N / 16, 320, 0, st>>>(m);
            else if (n_tok == 4) gr_up_q8_soa_km_multi_kernel<4, true><<<N / 16, 320, 0, st>>>(m);
            else if (n_tok == 5) gr_up_q8_soa_km_multi_kernel<5, true><<<N / 16, 320, 0, st>>>(m);
            else if (n_tok == 6) gr_up_q8_soa_km_multi_kernel<6, true><<<N / 16, 320, 0, st>>>(m);
            else if (n_tok == 7) gr_up_q8_soa_km_multi_kernel<7, true><<<N / 16, 320, 0, st>>>(m);
            else gr_up_q8_soa_km_multi_kernel<kFusedGrMaxT, true><<<N / 16, 320, 0, st>>>(m);
        } else {
            if (n_tok == 1) gr_up_q8_soa_km_multi_kernel<1, false><<<N / 16, 320, 0, st>>>(m);
            else if (n_tok == 2) gr_up_q8_soa_km_multi_kernel<2, false><<<N / 16, 320, 0, st>>>(m);
            else if (n_tok == 3) gr_up_q8_soa_km_multi_kernel<3, false><<<N / 16, 320, 0, st>>>(m);
            else if (n_tok == 4) gr_up_q8_soa_km_multi_kernel<4, false><<<N / 16, 320, 0, st>>>(m);
            else if (n_tok == 5) gr_up_q8_soa_km_multi_kernel<5, false><<<N / 16, 320, 0, st>>>(m);
            else if (n_tok == 6) gr_up_q8_soa_km_multi_kernel<6, false><<<N / 16, 320, 0, st>>>(m);
            else if (n_tok == 7) gr_up_q8_soa_km_multi_kernel<7, false><<<N / 16, 320, 0, st>>>(m);
            else gr_up_q8_soa_km_multi_kernel<kFusedGrMaxT, false><<<N / 16, 320, 0, st>>>(m);
        }
    } else if (a[0].w_up_q8 != nullptr) {
        if (n_tok == 1) gr_up_q8_0_multi_kernel<1><<<N / 8, 64, 0, st>>>(m);
        else if (n_tok == 2) gr_up_q8_0_multi_kernel<2><<<N / 8, 64, 0, st>>>(m);
        else if (n_tok <= 4) gr_up_q8_0_multi_kernel<4><<<N / 8, 64, 0, st>>>(m);
        else gr_up_q8_0_multi_kernel<kFusedGrMaxT><<<N / 8, 64, 0, st>>>(m);
    } else if (n_tok == 1) {
        gr_up_bf16_fast_multi_kernel<1><<<N / 4, 128, 0, st>>>(m);
    } else if (n_tok == 2) {
        gr_up_bf16_fast_multi_kernel<2><<<N / 4, 128, 0, st>>>(m);
    } else if (n_tok == 3) {
        gr_up_bf16_fast_multi_kernel<3><<<N / 4, 128, 0, st>>>(m);
    } else if (n_tok == 4) {
        gr_up_bf16_fast_multi_kernel<4><<<N / 4, 128, 0, st>>>(m);
    } else if (n_tok == 5) {
        gr_up_bf16_fast_multi_kernel<5><<<N / 4, 128, 0, st>>>(m);
    } else if (n_tok == 6) {
        gr_up_bf16_fast_multi_kernel<6><<<N / 4, 128, 0, st>>>(m);
    } else if (n_tok == 7) {
        gr_up_bf16_fast_multi_kernel<7><<<N / 4, 128, 0, st>>>(m);
    } else if (n_tok == 8) {
        gr_up_bf16_fast_multi_kernel<8><<<N / 4, 128, 0, st>>>(m);
    } else {
        gr_up_multi_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || (!a.w_down && !a.w_down_q8) || (!a.w_up && !a.w_up_q8) || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    gr_down_kernel<<<DOWN_BLOCKS + 1, THREADS, 0, st>>>(a);
    gr_up_kernel<<<UP_BLOCKS, THREADS, 0, st>>>(a);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
