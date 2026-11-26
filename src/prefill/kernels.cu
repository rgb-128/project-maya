// src/prefill/kernels.cu - see include/strata/prefill/kernels.hpp.
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/prefill/failure.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;

__device__ __forceinline__ float warp_sum(float v) {
#if defined(STRATA_USE_HIP) && defined(__gfx906__)
    v += __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401f));
    v += __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201f));
    v += __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101f));
    v += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x4e, 0xf, 0xf, false));
    v += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0xb1, 0xf, 0xf, false));
    return v;
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
#endif
}
__device__ __forceinline__ float warp_max(float v) {
#if defined(STRATA_USE_HIP) && defined(__gfx906__)
    v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401f)));
    v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201f)));
    v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101f)));
    v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x4e, 0xf, 0xf, false)));
    v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0xb1, 0xf, 0xf, false)));
    return v;
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
#endif
}
__device__ __forceinline__ uint16_t bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }
// block-wide sum for blockDim.x <= 1024, result broadcast
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
// A failed launch (an allocation, most of the time) ends this request, not the process: the reason is thrown and
// the caller turns it into an error for the request (include/strata/prefill/failure.hpp).
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill %s: %s\n", what, cudaGetErrorString(e));
        strata::prefill::fail(std::string("prefill ") + what + ": " + cudaGetErrorString(e));
    }
}
unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// ---------------------------------------------------------------- hyper-connection
__global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                               float* __restrict__ xn, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        xn[row * N + d] = v;
        xn16[row * N + d] = bf(v);
    }
}
// F-1: the row scale only (and the BF16 image); gr_mix_r_kernel recomputes r * rs * w itself, in the same order,
// so the FP32 copy of the normalized rows (T x 10240 floats) is neither written nor read
__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    for (int d = threadIdx.x; d < N; d += blockDim.x) xn16[row * N + d] = bf(r[d] * rs * w[c * N + d]);
}
// Vectorized float4 + SMEM rs gr_mix_r (1 block/token, 256 threads, 0 64-bit divs, 128-bit R/w/g/mixed + 64-bit bf16/f16!)
__global__ void __launch_bounds__(256) gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs,
                                                       const float* __restrict__ w, const float* __restrict__ g,
                                                       float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                                       int64_t T, uint16_t* __restrict__ mixed_h) {
    const int64_t t = blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x;
    __shared__ float s_rs[HC];
    if (tid < HC) s_rs[tid] = rs[t * HC + tid];
    __syncthreads();
    const float rs0 = s_rs[0], rs1 = s_rs[1], rs2 = s_rs[2], rs3 = s_rs[3];
    const float4* R0 = reinterpret_cast<const float4*>(R + t * D + 0 * N);
    const float4* R1 = reinterpret_cast<const float4*>(R + t * D + 1 * N);
    const float4* R2 = reinterpret_cast<const float4*>(R + t * D + 2 * N);
    const float4* R3 = reinterpret_cast<const float4*>(R + t * D + 3 * N);
    const float4* W0 = reinterpret_cast<const float4*>(w + 0 * N);
    const float4* W1 = reinterpret_cast<const float4*>(w + 1 * N);
    const float4* W2 = reinterpret_cast<const float4*>(w + 2 * N);
    const float4* W3 = reinterpret_cast<const float4*>(w + 3 * N);
    const float4* G0 = reinterpret_cast<const float4*>(g + t * D + 0 * N);
    const float4* G1 = reinterpret_cast<const float4*>(g + t * D + 1 * N);
    const float4* G2 = reinterpret_cast<const float4*>(g + t * D + 2 * N);
    const float4* G3 = reinterpret_cast<const float4*>(g + t * D + 3 * N);
    float4* m4 = reinterpret_cast<float4*>(mixed + t * N);
    uint2* m16 = mixed16 ? reinterpret_cast<uint2*>(mixed16 + t * N) : nullptr;
    uint2* mh  = mixed_h ? reinterpret_cast<uint2*>(mixed_h + t * N) : nullptr;
    constexpr float inv_hc = 1.0f / (float) HC;
    for (int j = tid; j < N / 4; j += 256) {
        const float4 r0 = __ldg(R0 + j), w0 = __ldg(W0 + j), g0 = __ldg(G0 + j);
        const float4 r1 = __ldg(R1 + j), w1 = __ldg(W1 + j), g1 = __ldg(G1 + j);
        const float4 r2 = __ldg(R2 + j), w2 = __ldg(W2 + j), g2 = __ldg(G2 + j);
        const float4 r3 = __ldg(R3 + j), w3 = __ldg(W3 + j), g3 = __ldg(G3 + j);
        float sx = fmaf(r0.x * rs0 * w0.x, sigm(g0.x), 0.0f);
        float sy = fmaf(r0.y * rs0 * w0.y, sigm(g0.y), 0.0f);
        float sz = fmaf(r0.z * rs0 * w0.z, sigm(g0.z), 0.0f);
        float sw = fmaf(r0.w * rs0 * w0.w, sigm(g0.w), 0.0f);
        sx = fmaf(r1.x * rs1 * w1.x, sigm(g1.x), sx);
        sy = fmaf(r1.y * rs1 * w1.y, sigm(g1.y), sy);
        sz = fmaf(r1.z * rs1 * w1.z, sigm(g1.z), sz);
        sw = fmaf(r1.w * rs1 * w1.w, sigm(g1.w), sw);
        sx = fmaf(r2.x * rs2 * w2.x, sigm(g2.x), sx);
        sy = fmaf(r2.y * rs2 * w2.y, sigm(g2.y), sy);
        sz = fmaf(r2.z * rs2 * w2.z, sigm(g2.z), sz);
        sw = fmaf(r2.w * rs2 * w2.w, sigm(g2.w), sw);
        sx = fmaf(r3.x * rs3 * w3.x, sigm(g3.x), sx) * inv_hc;
        sy = fmaf(r3.y * rs3 * w3.y, sigm(g3.y), sy) * inv_hc;
        sz = fmaf(r3.z * rs3 * w3.z, sigm(g3.z), sz) * inv_hc;
        sw = fmaf(r3.w * rs3 * w3.w, sigm(g3.w), sw) * inv_hc;
        m4[j] = make_float4(sx, sy, sz, sw);
        if (m16) m16[j] = make_uint2((uint32_t) bf(sx) | ((uint32_t) bf(sy) << 16),
                                     (uint32_t) bf(sz) | ((uint32_t) bf(sw) << 16));
        if (mh) {
            const __half2 h0 = __floats2half2_rn(sx, sy), h1 = __floats2half2_rn(sz, sw);
            mh[j] = make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
        }
    }
}
// F-2: gr_write_kernel for one row (t, c), then gr_norm_rs_kernel's reduction over it with the next half's norm
// weights - the same thread-to-element mapping (256 threads, stride 256) and block_sum, so rs and the BF16 image are
// the same bits, and R is not read back
constexpr int GRW_PER = (N + 255) / 256;
__global__ void __launch_bounds__(256) gr_write_norm_rs_kernel(float* __restrict__ R, const float* __restrict__ bo,
                                                               const float* __restrict__ inj, int64_t inj_ld,
                                                               const float* __restrict__ w, float eps,
                                                               float* __restrict__ rs_out, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int64_t t = row / HC;
    const int c = (int) (row % HC);
    float* r = R + row * N;
    const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
    float v[GRW_PER];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < GRW_PER; ++k) {
        const int d = threadIdx.x + k * 256;
        const float x = fmaf(bo[t * N + d], sc, r[d]);
        r[d] = x;
        v[k] = x;
        ss += x * x;
    }
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
#pragma unroll
    for (int k = 0; k < GRW_PER; ++k) {
        const int d = threadIdx.x + k * 256;
        xn16[row * N + d] = bf(v[k] * rs * w[c * N + d]);
    }
}
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = lo[i] / (float) HC;
    lo16[i] = bf(x / (1.0f + __expf(-x)));
}
__global__ void gr_silu_f16_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo_h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n4 = n >> 2;
    if (i >= n4) return;
    const float4 v = reinterpret_cast<const float4*>(lo)[i];
    constexpr float inv_hc = 1.0f / (float) HC;
    const float x0 = v.x * inv_hc, x1 = v.y * inv_hc, x2 = v.z * inv_hc, x3 = v.w * inv_hc;
    const __half2 h0 = __floats2half2_rn(x0 / (1.0f + __expf(-x0)), x1 / (1.0f + __expf(-x1)));
    const __half2 h1 = __floats2half2_rn(x2 / (1.0f + __expf(-x2)), x3 / (1.0f + __expf(-x3)));
    reinterpret_cast<uint2*>(lo_h)[i] = make_uint2(*reinterpret_cast<const uint32_t*>(&h0),
                                                   *reinterpret_cast<const uint32_t*>(&h1));
}
// Fused per-token (optional gr_write +) RMSNorm -> FP16 xn_h + inject[t, 0..3] dot product in ONE pass over R!
template <bool DO_WRITE>
__global__ void __launch_bounds__(256) gr_norm_rs_f16_inj_kernel(
    float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj_prev, int64_t inj_ld,
    const float* __restrict__ w, const uint16_t* __restrict__ w_inj, float eps,
    float* __restrict__ rs_out, uint16_t* __restrict__ xn_h, float* __restrict__ inj_out,
    int64_t t_off, int64_t t_end) {
    __shared__ float sh[32];
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x;
    const bool do_norm = (!DO_WRITE) || (t >= t_off && t < t_end);
    float sc[HC];
    if constexpr (DO_WRITE) {
#pragma unroll
        for (int c = 0; c < HC; ++c) sc[c] = 2.0f * sigm(inj_prev[t * inj_ld + c] / (float) HC);
    }
    float b_val[GRW_PER];
    if constexpr (DO_WRITE) {
#pragma unroll
        for (int k = 0; k < GRW_PER; ++k) b_val[k] = bo[t * N + tid + k * 256];
        if (!do_norm) {
            for (int c = 0; c < HC; ++c) {
                float* r = R + (t * HC + c) * N;
#pragma unroll
                for (int k = 0; k < GRW_PER; ++k) {
                    const int d = tid + k * 256;
                    r[d] = fmaf(b_val[k], sc[c], r[d]);
                }
            }
            return;
        }
    }
    const int64_t t_rel = DO_WRITE ? (t - t_off) : t;
    float inj_acc[HC] = {0.f, 0.f, 0.f, 0.f};
    for (int c = 0; c < HC; ++c) {
        const int64_t row = t * HC + c;
        const int64_t out_row = t_rel * HC + c;
        float* r = R + row * N;
        float v[GRW_PER];
        float ss = 0.0f;
#pragma unroll
        for (int k = 0; k < GRW_PER; ++k) {
            const int d = tid + k * 256;
            float x = r[d];
            if constexpr (DO_WRITE) {
                x = fmaf(b_val[k], sc[c], x);
                r[d] = x;
            }
            v[k] = x;
            ss += x * x;
        }
        const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
        if (tid == 0) rs_out[out_row] = rs;
        const int c_off = c * N;
#pragma unroll
        for (int k = 0; k < GRW_PER; ++k) {
            const int d = tid + k * 256;
            const float nv = v[k] * rs * w[c_off + d];
            xn_h[out_row * N + d] = hf(nv);
            const int col = c_off + d;
            inj_acc[0] = fmaf(nv, __uint_as_float((uint32_t) __ldg(w_inj + 0 * D + col) << 16), inj_acc[0]);
            inj_acc[1] = fmaf(nv, __uint_as_float((uint32_t) __ldg(w_inj + 1 * D + col) << 16), inj_acc[1]);
            inj_acc[2] = fmaf(nv, __uint_as_float((uint32_t) __ldg(w_inj + 2 * D + col) << 16), inj_acc[2]);
            inj_acc[3] = fmaf(nv, __uint_as_float((uint32_t) __ldg(w_inj + 3 * D + col) << 16), inj_acc[3]);
        }
    }
#pragma unroll
    for (int o = 0; o < HC; ++o) {
        const float s = block_sum(inj_acc[o], sh);
        if (tid == 0) inj_out[t * HC + o] = s;
    }
}
__global__ void gr_mix_kernel(const float* __restrict__ xn, const float* __restrict__ g, float* __restrict__ mixed,
                              uint16_t* __restrict__ mixed16, int64_t T, uint16_t* __restrict__ mixed_h) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        s = fmaf(xn[j], sigm(g[j]), s);
    }
    s /= (float) HC;
    mixed[i] = s;
    if (mixed16) mixed16[i] = bf(s);
    if (mixed_h) mixed_h[i] = hf(s);
}
__global__ void gr_write_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                int64_t inj_ld, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, c = (i % D) / N, d = i % N;
    R[i] = fmaf(bo[t * N + d], 2.0f * sigm(inj[t * inj_ld + c] / (float) HC), R[i]);
}
__global__ void gr_broadcast_kernel(const float* __restrict__ e, float* __restrict__ R, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, d = i % N;
    R[i] = e[t * N + d];
}

// ---------------------------------------------------------------- GDN
__global__ void gdn_gates_kernel(const float* __restrict__ ab, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int64_t T, int hv) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * hv) return;
    const int64_t t = i / hv, h = i % hv;
    const float v = ab[t * 2 * hv + h] + dt[h];
    gate[i] = (v > 20.0f ? v : log1pf(__expf(v))) * ssm_a[h];
    beta[i] = sigm(ab[t * 2 * hv + hv + h]);
}
// one thread per channel, walks the chunk; then a second kernel normalises
__global__ void gdn_conv_kernel(float* __restrict__ hist, const float* __restrict__ qkv, const float* __restrict__ w,
                                float* __restrict__ h, int64_t T, int c_dim) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= c_dim) return;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = 0; t < T; ++t) {
        const float x = qkv[t * c_dim + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * c_dim + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
    hist[c * 3] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
}
// C-3: the same 4-tap causal conv, tiled over tokens: thread (c, tile) reads its tile's 3 predecessors from the
// chunk (or the history before it) instead of carrying them - the conv reads inputs, not its own outputs, so the
// tiles are independent. The same expression per element (so the same bits); the history is written afterwards.
constexpr int CONV_TILE = 64;
__global__ void gdn_conv_tiled_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                      const float* __restrict__ w, float* __restrict__ h, int64_t T, int c_dim) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= c_dim) return;
    const int64_t t0 = (int64_t) blockIdx.y * CONV_TILE;
    if (t0 >= T) return;
    const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
    auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * c_dim + c] : hist[c * 3 + (int) (t + 3)]; };
    float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = t0; t < t1; ++t) {
        const float x = qkv[t * c_dim + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * c_dim + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
}
// the history after the chunk: its last three inputs (the older history where the chunk is shorter than 3)
__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T, int c_dim) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= c_dim) return;
    float v[3];
    for (int k = 0; k < 3; ++k) {
        const int64_t t = T - 3 + k;
        v[k] = t >= 0 ? qkv[t * c_dim + c] : hist[c * 3 + (int) (t + 3)];
    }
    hist[c * 3] = v[0]; hist[c * 3 + 1] = v[1]; hist[c * 3 + 2] = v[2];
}
__global__ void gdn_l2_kernel(float* __restrict__ h, float eps, int c_dim) {
    // block (t, head) over the 2*hk q/k heads, 128 threads
    const int64_t t = blockIdx.y;
    const int head = blockIdx.x;
    float* x = h + t * c_dim + head * S;
    const float v = x[threadIdx.x];
    float sq = warp_sum(v * v);
    __shared__ float part[4];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    x[threadIdx.x] = v * rsqrtf(ss + eps);
}
constexpr int RG = 4, RPG = S / RG;
__global__ void __launch_bounds__(S * RG) gdn_rec_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                         const float* __restrict__ gate,
                                                         const float* __restrict__ beta, const float* __restrict__ z,
                                                         const float* __restrict__ gamma, float eps,
                                                         float* __restrict__ y, uint16_t* __restrict__ y16, int64_t T,
                                                         int hk, int hv, int c_dim) {
    __shared__ float sk[S], sq[S], red[RG][S], wsum[16];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % hk;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * hv + head) * S + col;
    const size_t rs = (size_t) hv * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    const float g_col = gamma[col];
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * c_dim;
        __syncthreads();
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[hk * S + qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[t * hv + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * hk * S + head * S + col] - g * kv_col) * beta[t * hv + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sp = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);
            sp = oc * oc;
        }
        sp = warp_sum(sp);
        if ((tid & 31) == 0) wsum[tid >> 5] = sp;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float v = oc * rsqrtf(ss / (float) S + eps) * g_col * sigm(z[t * hv * S + head * S + col]);
            y[t * hv * S + head * S + col] = v;
            y16[t * hv * S + head * S + col] = hf(v);
        }
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

constexpr int CB = 32, NCB = S / CB;
__global__ void __launch_bounds__(CB * RG) gdn_rec_cols_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                const float* __restrict__ gate,
                                                                const float* __restrict__ beta,
                                                                float* __restrict__ oc_out, int64_t T,
                                                                int hk, int hv, int c_dim) {
    __shared__ float sk[S], sq[S], red[RG][CB];
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    const int qh = head % hk;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * hv + head) * S + col;
    const size_t rs = (size_t) hv * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * c_dim;
        __syncthreads();
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[hk * S + qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[t * hv + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][c] = kv;
        __syncthreads();
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (ht[2 * hk * S + head * S + col] - g * kv_col) * beta[t * hv + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][c] = o;
        __syncthreads();
        if (rg == 0) oc_out[t * hv * S + head * S + col] = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

// GP100 Chunkwise WY Matrix-Form GDN Prefill (CH=32, pure FP32, max_diff ~3e-9 vs token-by-token, STACK:0!):
// Uses caller's free scratch buffer (zero extra VRAM) -> 1.80x faster than gdn_rec_cols_kernel on TP=4!
constexpr int GDN_CH = 32;

__global__ void __launch_bounds__(128, 2) gdn_chunk_prep_v2_kernel(
    const float* __restrict__ h, const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ W_out, float* __restrict__ U_out, float* __restrict__ P_out,
    float* __restrict__ G_out, int64_t T, int hk, int hv, int c_dim) {
    const int cid = (int) blockIdx.x, head = (int) blockIdx.y, qh = head % hk;
    const int tid = threadIdx.x;
    const int t0 = cid * GDN_CH;
    const int len = (t0 + GDN_CH <= T) ? GDN_CH : (int) (T - t0);
    const size_t n_chunks = gridDim.x;
    const size_t ch_idx = (size_t) head * n_chunks + cid;

    __shared__ float sK[GDN_CH][S + 1];
    __shared__ float sM[GDN_CH][GDN_CH + 1];
    __shared__ float sQt[GDN_CH][17];
    __shared__ float sG[GDN_CH];
    __shared__ float sB[GDN_CH];

    if (tid < GDN_CH) {
        const float g = (tid < len) ? __ldg(gate + (size_t) (t0 + tid) * hv + head) : 0.0f;
        const float b = (tid < len) ? __ldg(beta + (size_t) (t0 + tid) * hv + head) : 0.0f;
        float cum = g;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float y = __shfl_up_sync(0xffffffffu, cum, o);
            if (tid >= o) cum += y;
        }
        sG[tid] = cum;
        sB[tid] = b;
        G_out[ch_idx * GDN_CH + tid] = cum;
    }
    for (int i = tid; i < GDN_CH * 32; i += 128) {
        const int r = i >> 5, c4 = i & 31;
        const float4 v = (r < len)
            ? __ldg(reinterpret_cast<const float4*>(h + (size_t) (t0 + r) * c_dim + (hk + qh) * S + c4 * 4))
            : make_float4(0.f, 0.f, 0.f, 0.f);
        sK[r][c4 * 4 + 0] = v.x; sK[r][c4 * 4 + 1] = v.y;
        sK[r][c4 * 4 + 2] = v.z; sK[r][c4 * 4 + 3] = v.w;
    }
    __syncthreads();

    // 1. A[32][32] = tril(beta_i * exp(G_i - G_j) * (K_i . K_j)) via 2x4 outer-product
    {
        const int tx = tid & 7, ty = tid >> 3;
        float acc[2][4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
#pragma unroll 4
        for (int d = 0; d < S; ++d) {
            const float ki0 = sK[ty][d], ki1 = sK[ty + 16][d];
            const float kj0 = sK[tx][d], kj1 = sK[tx + 8][d], kj2 = sK[tx + 16][d], kj3 = sK[tx + 24][d];
            acc[0][0] = fmaf(ki0, kj0, acc[0][0]); acc[0][1] = fmaf(ki0, kj1, acc[0][1]);
            acc[0][2] = fmaf(ki0, kj2, acc[0][2]); acc[0][3] = fmaf(ki0, kj3, acc[0][3]);
            acc[1][0] = fmaf(ki1, kj0, acc[1][0]); acc[1][1] = fmaf(ki1, kj1, acc[1][1]);
            acc[1][2] = fmaf(ki1, kj2, acc[1][2]); acc[1][3] = fmaf(ki1, kj3, acc[1][3]);
        }
#pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
            const int i = ty + ii * 16;
            const float bi = sB[i], gi = sG[i];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int j = tx + jj * 8;
                sM[i][j] = (i < len && j < i) ? (bi * __expf(gi - sG[j]) * acc[ii][jj]) : ((i == j) ? 1.f : 0.f);
            }
        }
    }
    __syncthreads();

    // 2. Unit lower-triangular inversion M = (I + A)^{-1} in warp 0
    if (tid < GDN_CH) {
        const int j = tid;
        for (int i = 1; i < len; ++i) {
            __syncwarp();
            const float a_ij = sM[i][j];
            float sum = -a_ij;
            if (j < i) {
                for (int k = j + 1; k < i; ++k) sum = fmaf(-sM[i][k], sM[k][j], sum);
            }
            __syncwarp();
            if (j < i) sM[i][j] = sum;
        }
    }
    __syncthreads();

    // 3. P[32][32] = tril(exp(G_i - G_j) * (Q_i . K_j))
    {
        const int tx = tid & 7, ty = tid >> 3;
        float acc[2][4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
        for (int d0 = 0; d0 < S; d0 += 16) {
            const int r = tid >> 2, c4 = tid & 3;
            const float4 v = (r < len)
                ? __ldg(reinterpret_cast<const float4*>(h + (size_t) (t0 + r) * c_dim + qh * S + d0 + c4 * 4))
                : make_float4(0.f, 0.f, 0.f, 0.f);
            sQt[r][c4 * 4 + 0] = v.x; sQt[r][c4 * 4 + 1] = v.y;
            sQt[r][c4 * 4 + 2] = v.z; sQt[r][c4 * 4 + 3] = v.w;
            __syncthreads();
#pragma unroll 4
            for (int d = 0; d < 16; ++d) {
                const float qi0 = sQt[ty][d], qi1 = sQt[ty + 16][d];
                const float kj0 = sK[tx][d0 + d], kj1 = sK[tx + 8][d0 + d], kj2 = sK[tx + 16][d0 + d], kj3 = sK[tx + 24][d0 + d];
                acc[0][0] = fmaf(qi0, kj0, acc[0][0]); acc[0][1] = fmaf(qi0, kj1, acc[0][1]);
                acc[0][2] = fmaf(qi0, kj2, acc[0][2]); acc[0][3] = fmaf(qi0, kj3, acc[0][3]);
                acc[1][0] = fmaf(qi1, kj0, acc[1][0]); acc[1][1] = fmaf(qi1, kj1, acc[1][1]);
                acc[1][2] = fmaf(qi1, kj2, acc[1][2]); acc[1][3] = fmaf(qi1, kj3, acc[1][3]);
            }
            __syncthreads();
        }
        float* P_dst = P_out + ch_idx * GDN_CH * GDN_CH;
#pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
            const int i = ty + ii * 16;
            const float gi = sG[i];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int j = tx + jj * 8;
                P_dst[i * GDN_CH + j] = (i < len && j <= i) ? (__expf(gi - sG[j]) * acc[ii][jj]) : 0.f;
            }
        }
    }

    // 4. W[32][128] = M @ (beta * exp(G) * K)
    for (int i = tid; i < GDN_CH * 32; i += 128) {
        const int r = i >> 5, c4 = i & 31;
        const float sc = (r < len) ? (sB[r] * __expf(sG[r])) : 0.f;
        sK[r][c4 * 4 + 0] *= sc; sK[r][c4 * 4 + 1] *= sc;
        sK[r][c4 * 4 + 2] *= sc; sK[r][c4 * 4 + 3] *= sc;
    }
    __syncthreads();
    {
        const int tx = tid & 31, ty = tid >> 5;
        float* W_dst = W_out + ch_idx * GDN_CH * S;
        for (int rb = 0; rb < 2; ++rb) {
            const int r0 = ty * 8 + rb * 4;
            float a[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) a[i][j] = 0.f;
            const int max_k = min(len, r0 + 4);
            for (int k = 0; k < max_k; ++k) {
                const float m0 = sM[r0 + 0][k], m1 = sM[r0 + 1][k], m2 = sM[r0 + 2][k], m3 = sM[r0 + 3][k];
                const float k0 = sK[k][tx], k1 = sK[k][tx + 32], k2 = sK[k][tx + 64], k3 = sK[k][tx + 96];
                a[0][0] = fmaf(m0, k0, a[0][0]); a[0][1] = fmaf(m0, k1, a[0][1]); a[0][2] = fmaf(m0, k2, a[0][2]); a[0][3] = fmaf(m0, k3, a[0][3]);
                a[1][0] = fmaf(m1, k0, a[1][0]); a[1][1] = fmaf(m1, k1, a[1][1]); a[1][2] = fmaf(m1, k2, a[1][2]); a[1][3] = fmaf(m1, k3, a[1][3]);
                a[2][0] = fmaf(m2, k0, a[2][0]); a[2][1] = fmaf(m2, k1, a[2][1]); a[2][2] = fmaf(m2, k2, a[2][2]); a[2][3] = fmaf(m2, k3, a[2][3]);
                a[3][0] = fmaf(m3, k0, a[3][0]); a[3][1] = fmaf(m3, k1, a[3][1]); a[3][2] = fmaf(m3, k2, a[3][2]); a[3][3] = fmaf(m3, k3, a[3][3]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = r0 + i;
                W_dst[(size_t) r * S + tx + 0]  = a[i][0];
                W_dst[(size_t) r * S + tx + 32] = a[i][1];
                W_dst[(size_t) r * S + tx + 64] = a[i][2];
                W_dst[(size_t) r * S + tx + 96] = a[i][3];
            }
        }
    }
    __syncthreads();

    // 5. U[32][128] = M @ (beta * V)
    for (int i = tid; i < GDN_CH * 32; i += 128) {
        const int r = i >> 5, c4 = i & 31;
        const float b = (r < len) ? sB[r] : 0.f;
        const float4 v = (r < len)
            ? __ldg(reinterpret_cast<const float4*>(h + (size_t) (t0 + r) * c_dim + (2 * hk + head) * S + c4 * 4))
            : make_float4(0.f, 0.f, 0.f, 0.f);
        sK[r][c4 * 4 + 0] = v.x * b; sK[r][c4 * 4 + 1] = v.y * b;
        sK[r][c4 * 4 + 2] = v.z * b; sK[r][c4 * 4 + 3] = v.w * b;
    }
    __syncthreads();
    {
        const int tx = tid & 31, ty = tid >> 5;
        float* U_dst = U_out + ch_idx * GDN_CH * S;
        for (int rb = 0; rb < 2; ++rb) {
            const int r0 = ty * 8 + rb * 4;
            float a[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) a[i][j] = 0.f;
            const int max_k = min(len, r0 + 4);
            for (int k = 0; k < max_k; ++k) {
                const float m0 = sM[r0 + 0][k], m1 = sM[r0 + 1][k], m2 = sM[r0 + 2][k], m3 = sM[r0 + 3][k];
                const float v0 = sK[k][tx], v1 = sK[k][tx + 32], v2 = sK[k][tx + 64], v3 = sK[k][tx + 96];
                a[0][0] = fmaf(m0, v0, a[0][0]); a[0][1] = fmaf(m0, v1, a[0][1]); a[0][2] = fmaf(m0, v2, a[0][2]); a[0][3] = fmaf(m0, v3, a[0][3]);
                a[1][0] = fmaf(m1, v0, a[1][0]); a[1][1] = fmaf(m1, v1, a[1][1]); a[1][2] = fmaf(m1, v2, a[1][2]); a[1][3] = fmaf(m1, v3, a[1][3]);
                a[2][0] = fmaf(m2, v0, a[2][0]); a[2][1] = fmaf(m2, v1, a[2][1]); a[2][2] = fmaf(m2, v2, a[2][2]); a[2][3] = fmaf(m2, v3, a[2][3]);
                a[3][0] = fmaf(m3, v0, a[3][0]); a[3][1] = fmaf(m3, v1, a[3][1]); a[3][2] = fmaf(m3, v2, a[3][2]); a[3][3] = fmaf(m3, v3, a[3][3]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = r0 + i;
                U_dst[(size_t) r * S + tx + 0]  = a[i][0];
                U_dst[(size_t) r * S + tx + 32] = a[i][1];
                U_dst[(size_t) r * S + tx + 64] = a[i][2];
                U_dst[(size_t) r * S + tx + 96] = a[i][3];
            }
        }
    }
}

__global__ void __launch_bounds__(64, 3) gdn_chunk_recur_v2_kernel(
    float* __restrict__ state, const float* __restrict__ h,
    const float* __restrict__ W_in, const float* __restrict__ U_in,
    const float* __restrict__ P_in, const float* __restrict__ G_in,
    float* __restrict__ oc_out, int64_t T, int hk, int hv, int c_dim) {
    constexpr int RCB_V = 16;
    constexpr int RNCB_V = S / RCB_V;   // 8 col blocks/head -> 96 blocks for hv=12
    constexpr int NC = RCB_V / 8;       // 2 cols/thread
    const int head = (int) blockIdx.x / RNCB_V;
    const int cb = (int) blockIdx.x % RNCB_V;
    const int qh = head % hk;
    const int col0 = cb * RCB_V;
    const int tid = threadIdx.x;
    const int tx = tid & 7;
    const int ty = tid >> 3;
    const int n_chunks = (int) ((T + GDN_CH - 1) / GDN_CH);

    __shared__ float sS[S][RCB_V + 1];
    __shared__ float sD[GDN_CH][RCB_V + 1];
    __shared__ float sW[GDN_CH][17];
    __shared__ float sQ[GDN_CH][17];
    __shared__ float sG[GDN_CH];

    float* st_base = state + (size_t) head * S + col0;
    const size_t st_rs = (size_t) hv * S;
    for (int i = tid; i < S * RCB_V; i += 64) {
        const int r = i / RCB_V, c = i - r * RCB_V;
        sS[r][c] = __ldg(st_base + (size_t) r * st_rs + c);
    }
    __syncthreads();

    const float inv_sqrt_s = rsqrtf((float) S);

    for (int cid = 0; cid < n_chunks; ++cid) {
        const int t0 = cid * GDN_CH;
        const int len = (t0 + GDN_CH <= T) ? GDN_CH : (int) (T - t0);
        const size_t ch_off = ((size_t) head * n_chunks + cid) * GDN_CH;
        const float* Wc = W_in + ch_off * S;
        const float* Uc = U_in + ch_off * S;
        const float* Pc = P_in + ch_off * GDN_CH;
        if (tid < GDN_CH) sG[tid] = __ldg(G_in + ch_off + tid);

        float d_acc[4][NC], o_acc[4][NC];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int i = ty + m * 8;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                d_acc[m][c] = (i < len) ? __ldg(Uc + (size_t) i * S + col0 + tx + c * 8) : 0.f;
                o_acc[m][c] = 0.f;
            }
        }

        // Fused Pass 1: Delta = U - W@S and O_inter = Q@S in ONE pass over sS
        for (int k0 = 0; k0 < S; k0 += 16) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int idx = tid + p * 64;
                const int r = idx >> 2, c4 = idx & 3;
                const float4 wv = __ldg(reinterpret_cast<const float4*>(Wc + (size_t) r * S + k0 + c4 * 4));
                const float4 qv = (r < len)
                    ? __ldg(reinterpret_cast<const float4*>(h + (size_t) (t0 + r) * c_dim + qh * S + k0 + c4 * 4))
                    : make_float4(0.f, 0.f, 0.f, 0.f);
                sW[r][c4 * 4 + 0] = wv.x; sW[r][c4 * 4 + 1] = wv.y; sW[r][c4 * 4 + 2] = wv.z; sW[r][c4 * 4 + 3] = wv.w;
                sQ[r][c4 * 4 + 0] = qv.x; sQ[r][c4 * 4 + 1] = qv.y; sQ[r][c4 * 4 + 2] = qv.z; sQ[r][c4 * 4 + 3] = qv.w;
            }
            __syncthreads();
#pragma unroll 4
            for (int k = 0; k < 16; ++k) {
                float sv[NC];
#pragma unroll
                for (int c = 0; c < NC; ++c) sv[c] = sS[k0 + k][tx + c * 8];
                const float w0 = sW[ty + 0][k], w1 = sW[ty + 8][k], w2 = sW[ty + 16][k], w3 = sW[ty + 24][k];
                const float q0 = sQ[ty + 0][k], q1 = sQ[ty + 8][k], q2 = sQ[ty + 16][k], q3 = sQ[ty + 24][k];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    d_acc[0][c] = fmaf(-w0, sv[c], d_acc[0][c]);
                    d_acc[1][c] = fmaf(-w1, sv[c], d_acc[1][c]);
                    d_acc[2][c] = fmaf(-w2, sv[c], d_acc[2][c]);
                    d_acc[3][c] = fmaf(-w3, sv[c], d_acc[3][c]);
                    o_acc[0][c] = fmaf(q0, sv[c], o_acc[0][c]);
                    o_acc[1][c] = fmaf(q1, sv[c], o_acc[1][c]);
                    o_acc[2][c] = fmaf(q2, sv[c], o_acc[2][c]);
                    o_acc[3][c] = fmaf(q3, sv[c], o_acc[3][c]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int i = ty + m * 8;
            const float eg = (i < len) ? __expf(sG[i]) : 0.f;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                sD[i][tx + c * 8] = (i < len) ? d_acc[m][c] : 0.f;
                o_acc[m][c] *= eg;
            }
        }
        for (int p = 0; p < 2; ++p) {
            const int idx = tid + p * 64;
            const int r = idx >> 2, c4 = idx & 3;
            const float4 p0 = __ldg(reinterpret_cast<const float4*>(Pc + r * GDN_CH + c4 * 4));
            const float4 p1 = __ldg(reinterpret_cast<const float4*>(Pc + r * GDN_CH + 16 + c4 * 4));
            sW[r][c4 * 4 + 0] = p0.x; sW[r][c4 * 4 + 1] = p0.y; sW[r][c4 * 4 + 2] = p0.z; sW[r][c4 * 4 + 3] = p0.w;
            sQ[r][c4 * 4 + 0] = p1.x; sQ[r][c4 * 4 + 1] = p1.y; sQ[r][c4 * 4 + 2] = p1.z; sQ[r][c4 * 4 + 3] = p1.w;
        }
        __syncthreads();
#pragma unroll 4
        for (int j = 0; j < 16; ++j) {
            float dv[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c) dv[c] = sD[j][tx + c * 8];
            const float p0 = sW[ty + 0][j], p1 = sW[ty + 8][j], p2 = sW[ty + 16][j], p3 = sW[ty + 24][j];
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                o_acc[0][c] = fmaf(p0, dv[c], o_acc[0][c]);
                o_acc[1][c] = fmaf(p1, dv[c], o_acc[1][c]);
                o_acc[2][c] = fmaf(p2, dv[c], o_acc[2][c]);
                o_acc[3][c] = fmaf(p3, dv[c], o_acc[3][c]);
            }
        }
#pragma unroll 4
        for (int j = 0; j < 16; ++j) {
            float dv[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c) dv[c] = sD[16 + j][tx + c * 8];
            const float p0 = sQ[ty + 0][j], p1 = sQ[ty + 8][j], p2 = sQ[ty + 16][j], p3 = sQ[ty + 24][j];
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                o_acc[0][c] = fmaf(p0, dv[c], o_acc[0][c]);
                o_acc[1][c] = fmaf(p1, dv[c], o_acc[1][c]);
                o_acc[2][c] = fmaf(p2, dv[c], o_acc[2][c]);
                o_acc[3][c] = fmaf(p3, dv[c], o_acc[3][c]);
            }
        }
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int i = ty + m * 8;
            if (i < len) {
                float* dst = oc_out + (size_t) (t0 + i) * hv * S + (size_t) head * S + col0 + tx;
#pragma unroll
                for (int c = 0; c < NC; ++c) dst[c * 8] = o_acc[m][c] * inv_sqrt_s;
            }
        }
        __syncthreads();

        // Pass 2: S = exp(G_end) * S + K_dec^T @ Delta
        const float g_end = sG[len - 1];
        const float decay_end = __expf(g_end);
        for (int k0 = 0; k0 < S; k0 += 32) {
            for (int p = 0; p < 2; ++p) {
                const int idx = tid + p * 64;
                const int r = idx >> 2, c4 = idx & 3;
                const float kd = (r < len) ? __expf(g_end - sG[r]) : 0.f;
                const float4 k0v = (r < len)
                    ? __ldg(reinterpret_cast<const float4*>(h + (size_t) (t0 + r) * c_dim + (hk + qh) * S + k0 + c4 * 4))
                    : make_float4(0.f, 0.f, 0.f, 0.f);
                const float4 k1v = (r < len)
                    ? __ldg(reinterpret_cast<const float4*>(h + (size_t) (t0 + r) * c_dim + (hk + qh) * S + k0 + 16 + c4 * 4))
                    : make_float4(0.f, 0.f, 0.f, 0.f);
                sW[r][c4 * 4 + 0] = k0v.x * kd; sW[r][c4 * 4 + 1] = k0v.y * kd; sW[r][c4 * 4 + 2] = k0v.z * kd; sW[r][c4 * 4 + 3] = k0v.w * kd;
                sQ[r][c4 * 4 + 0] = k1v.x * kd; sQ[r][c4 * 4 + 1] = k1v.y * kd; sQ[r][c4 * 4 + 2] = k1v.z * kd; sQ[r][c4 * 4 + 3] = k1v.w * kd;
            }
            __syncthreads();
            float su[4][NC];
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int c = 0; c < NC; ++c) su[m][c] = sS[k0 + ty + m * 8][tx + c * 8] * decay_end;
#pragma unroll 4
            for (int j = 0; j < GDN_CH; ++j) {
                float dv[NC];
#pragma unroll
                for (int c = 0; c < NC; ++c) dv[c] = sD[j][tx + c * 8];
                const float k00 = sW[j][ty + 0], k01 = sW[j][ty + 8];
                const float k10 = sQ[j][ty + 0], k11 = sQ[j][ty + 8];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    su[0][c] = fmaf(k00, dv[c], su[0][c]);
                    su[1][c] = fmaf(k01, dv[c], su[1][c]);
                    su[2][c] = fmaf(k10, dv[c], su[2][c]);
                    su[3][c] = fmaf(k11, dv[c], su[3][c]);
                }
            }
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int c = 0; c < NC; ++c) sS[k0 + ty + m * 8][tx + c * 8] = su[m][c];
            __syncthreads();
        }
    }

    for (int i = tid; i < S * RCB_V; i += 64) {
        const int r = i / RCB_V, c = i - r * RCB_V;
        st_base[(size_t) r * st_rs + c] = sS[r][c];
    }
}
__global__ void __launch_bounds__(S) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                         float eps, float* __restrict__ y, uint16_t* __restrict__ y16,
                                                         int hv) {
    __shared__ float wsum[4];
    const int64_t t = blockIdx.x;
    const int head = blockIdx.y, col = threadIdx.x;
    const size_t at = (size_t) t * hv * S + (size_t) head * S + col;
    const float oc = y[at];
    float sp = warp_sum(oc * oc);
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(z[t * hv * S + head * S + col]);
    y[at] = v;
    y16[at] = hf(v);
}

// ---------------------------------------------------------------- MoE
template <int REG>
__global__ void route_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ wout,
                             int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    const float* lg = logits + t * (REG * 32);
    float v[REG];
#pragma unroll
    for (int i = 0; i < REG; ++i) v[i] = lg[lane + i * 32];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < REG; ++i) mx = fmaxf(mx, v[i]);
    mx = warp_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] = expf(v[i] - mx); sum += v[i]; }
    const float rcp = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] *= rcp; if (isnan(v[i])) v[i] = -FLT_MAX; }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = lane;
#pragma unroll
        for (int i = 1; i < REG; ++i) if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m);
            const int oi = __shfl_xor_sync(0xffffffffu, ex, m);
            if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
        }
        if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
        if (lane == 0) ids[t * 10 + rank] = ex;
        if (rank == lane) selected = best;
    }
    selected_sum = fmaxf(warp_sum(selected_sum), 6.103515625e-5f);
    if (lane < 10) wout[t * 10 + lane] = selected / selected_sum;
}
// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales [2560][10] f16
template <bool HALF>
__global__ void blob_dequant_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu16,
                                    uint16_t* __restrict__ d16) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one thread per 4 weights (one code byte)
    const int64_t n_gu = 1280LL * 640, n_d = 2560LL * 160;
    if (i < n_gu) {
        const int64_t row = i / 640, byte = i % 640;
        const uint8_t c = blob[row * 640 + byte];
        const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = gu16 + row * 2560 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 160, byte = j % 160;
        const uint8_t c = blob[O_D_CODES + row * 160 + byte];
        const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = d16 + row * 640 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    }
}
__global__ void swiglu_il_kernel(const float* __restrict__ gu, uint16_t* __restrict__ h16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const int64_t r = i / 640, k = i % 640;
    const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
    h16[i] = hf(g / (1.0f + __expf(-g)) * u);
}
__global__ void swiglu_split_kernel(const float* __restrict__ gu, uint16_t* __restrict__ h16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const int64_t r = i / 640, k = i % 640;
    const float g = gu[r * 1280 + k], u = gu[r * 1280 + 640 + k];
    h16[i] = hf(g / (1.0f + __expf(-g)) * u);
}
__global__ void swiglu_pair_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h16,
                                   int64_t n_elems) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;
    const float a = g[i];
    h16[i] = hf(a / (1.0f + __expf(-a)) * u[i]);
}
__global__ void gather_rows16_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                     uint16_t* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one uint4 (8 bf16)
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    reinterpret_cast<uint4*>(dst)[r * per + j] = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
}
// Vectorized float4 + SMEM metadata moe_combine (3.87x faster: 428 us -> 110 us at T=2048!)
__global__ void __launch_bounds__(256) moe_combine_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                                          const float* __restrict__ w, const float* __restrict__ shared,
                                                          const float* __restrict__ sg, float* __restrict__ bo, int64_t T) {
    const int64_t t = blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x;
    __shared__ int32_t s_sl[10];
    __shared__ float s_w[10];
    __shared__ float s_sg;
    if (tid < 10) {
        s_sl[tid] = slot[t * 10 + tid];
        s_w[tid] = w[t * 10 + tid];
    }
    if (tid == 0) s_sg = sigm(sg[t]);
    __syncthreads();
    const float g = s_sg;
    const float4* sh4 = reinterpret_cast<const float4*>(shared + t * N);
    float4* bo4 = reinterpret_cast<float4*>(bo + t * N);
    for (int j = tid; j < N / 4; j += 256) {
        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int k = 0; k < 10; ++k) {
            const int32_t sl = s_sl[k];
            if (sl >= 0) {
                const float wk = s_w[k];
                const float4 v = __ldg(reinterpret_cast<const float4*>(Dm + (int64_t) sl * N) + j);
                a.x = fmaf(wk, v.x, a.x);
                a.y = fmaf(wk, v.y, a.y);
                a.z = fmaf(wk, v.z, a.z);
                a.w = fmaf(wk, v.w, a.w);
            }
        }
        const float4 s4 = __ldg(sh4 + j);
        bo4[j] = make_float4(fmaf(s4.x, g, a.x), fmaf(s4.y, g, a.y), fmaf(s4.z, g, a.z), fmaf(s4.w, g, a.w));
    }
}

// ---------------------------------------------------------------- QSA helpers
__global__ void rms_rows_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t cols, int64_t ld, float eps) {
    __shared__ float sh[32];
    float* r = x + (int64_t) blockIdx.x * ld;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) ss += r[c] * r[c];
    const float s = rsqrtf(block_sum(ss, sh) / (float) cols + eps);
    __syncthreads();
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) r[c] = s * r[c] * w[c];
}
__global__ void rope_kernel(float* __restrict__ x, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
                            float theta_scale, const int32_t* __restrict__ mtab) {
    const int64_t row = blockIdx.x;             // t * heads + h
    const int pair = threadIdx.x;               // 0..31
    const int64_t t = row / heads, h = row % heads;
    float* p = x + t * ld + h * dim;
    const float theta = (float) strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair) * powf(theta_scale, (float) pair);
    const float c = cosf(theta), s = sinf(theta);
    const float a = p[pair], b = p[pair + 32];
    p[pair] = a * c - b * s;
    p[pair + 32] = a * s + b * c;
}
__global__ void __launch_bounds__(64) split_q_kernel(const float* __restrict__ qf, float* __restrict__ q, int64_t T, int64_t nh) {
    const int64_t row = blockIdx.x;   // t * nh + h
    const int j = threadIdx.x;        // 0..63 -> float4 (256 floats)
    const int64_t t = row / nh, h = row - t * nh;
    reinterpret_cast<float4*>(q + row * 256)[j] = __ldg(reinterpret_cast<const float4*>(qf + (t * nh + h) * 512) + j);
}
__global__ void __launch_bounds__(64) gate_attn_kernel(const float* __restrict__ a, const float* __restrict__ qf, uint16_t* __restrict__ o16,
                                                       int64_t T, int64_t nh) {
    const int64_t row = blockIdx.x;   // t * nh + h
    const int j = threadIdx.x;        // 0..63 -> float4 -> uint2 half4
    const float4 av = __ldg(reinterpret_cast<const float4*>(a + row * 256) + j);
    const float4 gv = __ldg(reinterpret_cast<const float4*>(qf + row * 512 + 256) + j);
    const __half2 h0 = __floats2half2_rn(av.x * sigm(gv.x), av.y * sigm(gv.y));
    const __half2 h1 = __floats2half2_rn(av.z * sigm(gv.z), av.w * sigm(gv.w));
    reinterpret_cast<uint2*>(o16 + row * 256)[j] = make_uint2(*reinterpret_cast<const uint32_t*>(&h0),
                                                              *reinterpret_cast<const uint32_t*>(&h1));
}

// one block per (token, kv head, 64-value group); 64 threads. KV streaming: the pool page only if the block is
// resident (table >= 0), and the host copy and the prompt path's staging pool (both identity layout) when given.
__global__ void kv_append_kernel(const float* __restrict__ K, const float* __restrict__ V, int64_t pos0,
                                 const int32_t* __restrict__ table, int64_t page_size, int nkv, uint16_t* k_pool,
                                 uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                                 strata::kernels::KvHostPools host, strata::kernels::KvHostPools stage) {
    const int64_t t = blockIdx.x;
    const int kvh = blockIdx.y, g = blockIdx.z >> 1;
    const bool is_v = (blockIdx.z & 1) != 0;
    const int d = g * 64 + threadIdx.x;
    const float x = (is_v ? V : K)[t * (nkv * 256) + kvh * 256 + d];
    const int64_t pos = pos0 + t;
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * nkv + kvh) * page_size + pos % page_size;
    const int64_t row_id = ((pos / page_size) * nkv + kvh) * page_size + pos % page_size;
    if (k_pool != nullptr) {
        const uint16_t h = hf(x);
        if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = h;
        if (host.k_pool != nullptr) (is_v ? host.v_pool : host.k_pool)[row_id * 256 + d] = h;
        if (stage.k_pool != nullptr) (is_v ? stage.v_pool : stage.k_pool)[row_id * 256 + d] = h;
        return;
    }
    float a = warp_max(fabsf(x));
    __shared__ float wm[2];
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(wm[0], wm[1]);
    const uint16_t sb = hf(amax / 127.0f);
    const float sf = __half2float(__ushort_as_half(sb));
    int q = 0;
    if (sf > 0.0f) { q = __float2int_rn(x / sf); q = q < -127 ? -127 : (q > 127 ? 127 : q); }
    if (page >= 0) {
        (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
    }
    if (host.k_q != nullptr) {
        (is_v ? host.v_q : host.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? host.v_scale : host.k_scale)[row_id * 4 + g] = sb;
    }
    if (stage.k_q != nullptr) {
        (is_v ? stage.v_q : stage.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? stage.v_scale : stage.k_scale)[row_id * 4 + g] = sb;
    }
}
__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = hf(x[i]);
}
__global__ void bf16_to_f16_kernel(const uint16_t* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = hf(__uint_as_float((uint32_t) x[i] << 16));
}
__global__ void round_f16_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = __half2float(__float2half_rn(x[i]));
}
__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = bf(x[i]);
}

}  // namespace

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host, const strata::kernels::KvHostPools* stage,
               int64_t nkv) {
    if (T <= 0) return;
    kv_append_kernel<<<dim3((unsigned) T, (unsigned) nkv, 8), 64, 0, (cudaStream_t) stream>>>(
        K, V, pos0, page_table, page_size, (int) nkv, k_pool, v_pool, k_q, v_q, k_scale, v_scale,
        host ? *host : strata::kernels::KvHostPools{}, stage ? *stage : strata::kernels::KvHostPools{});
    check("kv_append");
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_f16");
}
void bf16_to_f16(const uint16_t* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    bf16_to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("bf16_to_f16");
}
void round_f16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    round_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("round_f16");
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_bf16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_bf16");
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream) {
    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16);
    check("gr_norm");
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream) {
    gr_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, rs, xn16);
    check("gr_norm_rs");
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h) {
    if (T <= 0) return;
    gr_mix_r_kernel<<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(R, rs, w_norm, gated, mixed, mixed16, T, mixed_h);
    check("gr_mix_r");
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream) {
    gr_write_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, w_norm_next, eps,
                                                                                      rs, xn16);
    check("gr_write_norm_rs");
}
void gr_norm_rs_f16_inj(const float* R, const float* w_norm, const uint16_t* w_inj_bf16, float eps, float* rs,
                        uint16_t* xn_h, float* inj_out, int64_t T, void* stream) {
    if (T <= 0) return;
    gr_norm_rs_f16_inj_kernel<false><<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(
        const_cast<float*>(R), nullptr, nullptr, 0, w_norm, w_inj_bf16, eps, rs, xn_h, inj_out, 0, T);
    check("gr_norm_rs_f16_inj");
}
void gr_write_norm_rs_f16_inj(float* R, const float* bo, const float* inj_prev, int64_t inj_ld,
                              const float* w_norm_next, const uint16_t* w_inj_bf16, float eps, float* rs,
                              uint16_t* xn_h, float* inj_out, int64_t T, void* stream,
                              int64_t t_off, int64_t t_cnt) {
    if (T <= 0) return;
    const int64_t t_end = (t_cnt >= 0) ? (t_off + t_cnt) : T;
    gr_norm_rs_f16_inj_kernel<true><<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(
        R, bo, inj_prev, inj_ld, w_norm_next, w_inj_bf16, eps, rs, xn_h, inj_out, t_off, t_end);
    check("gr_write_norm_rs_f16_inj");
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream) {
    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, T * LR);
    check("gr_silu");
}
void gr_silu_f16(const float* lo, uint16_t* lo_h, int64_t T, void* stream) {
    if (T <= 0) return;
    gr_silu_f16_kernel<<<blocks_for((T * LR) >> 2), 256, 0, (cudaStream_t) stream>>>(lo, lo_h, T * LR);
    check("gr_silu_f16");
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h) {
    gr_mix_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(xn, gated, mixed, mixed16, T, mixed_h);
    check("gr_mix");
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    gr_write_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, T);
    check("gr_write");
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    gr_broadcast_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(e, R, T);
    check("gr_broadcast");
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream,
               int64_t hv) {
    gdn_gates_kernel<<<blocks_for(T * hv), 256, 0, (cudaStream_t) stream>>>(ab, dt, ssm_a, gate, beta, T, (int) hv);
    check("gdn_gates");
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream,
              int64_t c, int64_t hk) {
    static const bool serial = std::getenv("MAYA_GDN_CONV_SERIAL") != nullptr;   // the old walk (A/B)
    if (serial || T <= CONV_TILE) {
        gdn_conv_kernel<<<(unsigned) (c / 128), 128, 0, (cudaStream_t) stream>>>(history, qkv, conv_w, h, T, (int) c);
    } else {
        gdn_conv_tiled_kernel<<<dim3((unsigned) (c / 128), (unsigned) ((T + CONV_TILE - 1) / CONV_TILE)), 128, 0,
                                (cudaStream_t) stream>>>(history, qkv, conv_w, h, T, (int) c);
        gdn_conv_hist_kernel<<<(unsigned) (c / 128), 128, 0, (cudaStream_t) stream>>>(history, qkv, T, (int) c);
    }
    gdn_l2_kernel<<<dim3((unsigned) (2 * hk), (unsigned) T), S, 0, (cudaStream_t) stream>>>(h, eps, (int) c);
    check("gdn_conv");
}
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream,
                    int64_t hk, int64_t hv, int64_t c, float* wy_scratch, size_t wy_scratch_floats) {
    static const bool serial = std::getenv("MAYA_GDN_REC_HEADS") != nullptr;   // the one-block-per-head kernel (A/B)
    static const bool no_wy = std::getenv("MAYA_GDN_REC_OLD") != nullptr;
    if (serial || T <= 0) {
        gdn_rec_kernel<<<(unsigned) hv, dim3(S, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, z, gamma, eps, y, y16, T,
                                                                                  (int) hk, (int) hv, (int) c);
    } else {
        const int64_t n_chunks = (T + GDN_CH - 1) / GDN_CH;
        const size_t n_w = (size_t) hv * (size_t) n_chunks * GDN_CH * S;
        const size_t n_u = n_w;
        const size_t n_p = (size_t) hv * (size_t) n_chunks * GDN_CH * GDN_CH;
        const size_t n_g = (size_t) hv * (size_t) n_chunks * GDN_CH;
        const size_t need = n_w + n_u + n_p + n_g;
        if (!no_wy && wy_scratch != nullptr && T >= 128 && need <= wy_scratch_floats) {
            float* W_buf = wy_scratch;
            float* U_buf = W_buf + n_w;
            float* P_buf = U_buf + n_u;
            float* G_buf = P_buf + n_p;
            gdn_chunk_prep_v2_kernel<<<dim3((unsigned) n_chunks, (unsigned) hv), 128, 0, (cudaStream_t) stream>>>(
                h, gate, beta, W_buf, U_buf, P_buf, G_buf, T, (int) hk, (int) hv, (int) c);
            gdn_chunk_recur_v2_kernel<<<(unsigned) (hv * 8), 64, 0, (cudaStream_t) stream>>>(
                state, h, W_buf, U_buf, P_buf, G_buf, y, T, (int) hk, (int) hv, (int) c);
        } else {
            gdn_rec_cols_kernel<<<(unsigned) (hv * NCB), dim3(CB, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T,
                                                                                                    (int) hk, (int) hv, (int) c);
        }
        gdn_out_norm_kernel<<<dim3((unsigned) T, (unsigned) hv), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16, (int) hv);
    }
    check("gdn_recurrence");
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (n_expert == 512)
        route_kernel<16><<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    else if (n_expert == 256)
        route_kernel<8><<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    else
        strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
    check("route");
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<false><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant");
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<true><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant_f16");
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    swiglu_il_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(gu, h16, n);
    check("swiglu_interleaved");
}
void swiglu_split(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    swiglu_split_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(gu, h16, n);
    check("swiglu_split");
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream, int64_t ff) {
    swiglu_pair_kernel<<<blocks_for(n * ff), 256, 0, (cudaStream_t) stream>>>(g, u, h16, n * ff);
    check("swiglu_pair");
}
namespace {
__global__ void copy_i32_kernel(int32_t* __restrict__ dst, const int32_t* __restrict__ src, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}
__global__ void signal_u32_kernel(uint32_t* __restrict__ dst, uint32_t val) {
    __threadfence_system();
    *reinterpret_cast<volatile uint32_t*>(dst) = val;
    __threadfence_system();
}
__global__ void __launch_bounds__(512) prefill_route_count_kernel(
    const int32_t* __restrict__ ids, const int8_t* __restrict__ owner_row,
    int tp_rank, int tp_size, int64_t tk, int32_t n_expert,
    int32_t* __restrict__ cnt_out, uint32_t* __restrict__ sync_dev, uint32_t target) {
    __shared__ int32_t s_cnt[512];
    __shared__ uint8_t s_own[512];
    __shared__ int32_t s_err;
    const int tid = threadIdx.x;
    if (tid < 512) {
        s_cnt[tid] = 0;
        uint8_t own = 1;
        if (tid < n_expert && tp_size > 1) {
            const int8_t ow = owner_row ? owner_row[tid] : (int8_t) -1;
            own = (ow >= 0) ? (uint8_t) (ow == tp_rank) : (uint8_t) ((tid % tp_size) == tp_rank);
        }
        s_own[tid] = own;
    }
    if (tid == 0) s_err = 0;
    __syncthreads();
    for (int64_t i = tid; i < tk; i += 512) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert) {
            atomicExch(&s_err, 1);
        } else if (s_own[e]) {
            atomicAdd(&s_cnt[e], 1);
        }
    }
    __syncthreads();
    if (tid < n_expert) cnt_out[tid] = s_cnt[tid];
    if (tid == 0) cnt_out[n_expert] = s_err;
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *reinterpret_cast<volatile uint32_t*>(sync_dev) = target;
        __threadfence_system();
    }
}
__global__ void __launch_bounds__(512) prefill_route_scatter_kernel(
    const int32_t* __restrict__ ids, const int8_t* __restrict__ owner_row,
    int tp_rank, int tp_size, int64_t tk, int32_t k, int32_t n_expert,
    const int32_t* __restrict__ off_in, int32_t* __restrict__ slot_dev, int32_t* __restrict__ src_dev) {
    __shared__ int32_t s_fill[512];
    __shared__ uint8_t s_own[512];
    const int tid = threadIdx.x;
    if (tid < 512) {
        s_fill[tid] = (tid < n_expert) ? off_in[tid] : 0;
        uint8_t own = 1;
        if (tid < n_expert && tp_size > 1) {
            const int8_t ow = owner_row ? owner_row[tid] : (int8_t) -1;
            own = (ow >= 0) ? (uint8_t) (ow == tp_rank) : (uint8_t) ((tid % tp_size) == tp_rank);
        }
        s_own[tid] = own;
    }
    __syncthreads();
    for (int64_t i = tid; i < tk; i += 512) {
        const int32_t e = ids[i];
        if (!s_own[e]) {
            slot_dev[i] = -1;
        } else {
            const int32_t p = atomicAdd(&s_fill[e], 1);
            slot_dev[i] = p;
            src_dev[p] = (int32_t) (i / k);
        }
    }
}
}  // namespace
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t b = (n + 255) / 256;
    copy_i32_kernel<<<(unsigned) (b < 256 ? b : 256), 256, 0, (cudaStream_t) stream>>>(dst, src, n);
    check("copy_i32");
}
void signal_u32(uint32_t* dst, uint32_t val, void* stream) {
    signal_u32_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(dst, val);
    check("signal_u32");
}
void prefill_route_count(const int32_t* ids, const int8_t* owner_row, int tp_rank, int tp_size, int64_t tk,
                         int32_t n_expert, int32_t* cnt_out, uint32_t* sync_dev, uint32_t target, void* stream) {
    prefill_route_count_kernel<<<1, 512, 0, (cudaStream_t) stream>>>(
        ids, owner_row, tp_rank, tp_size, tk, n_expert, cnt_out, sync_dev, target);
    check("prefill_route_count");
}
void prefill_route_scatter(const int32_t* ids, const int8_t* owner_row, int tp_rank, int tp_size, int64_t tk,
                           int32_t k, int32_t n_expert, const int32_t* off_in, int32_t* slot_dev, int32_t* src_dev,
                           void* stream) {
    prefill_route_scatter_kernel<<<1, 512, 0, (cudaStream_t) stream>>>(
        ids, owner_row, tp_rank, tp_size, tk, k, n_expert, off_in, slot_dev, src_dev);
    check("prefill_route_scatter");
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_kernel<<<blocks_for(n * (width / 8)), 256, 0, (cudaStream_t) stream>>>(x16, src, dst16, n, width);
    check("gather_rows16");
}
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream) {
    if (T <= 0) return;
    moe_combine_kernel<<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, bo, T);
    check("moe_combine");
}
namespace {
__global__ void bf16_dot_row_kernel(const uint16_t* __restrict__ X, const uint16_t* __restrict__ w,
                                    float* __restrict__ y, int64_t K) {
    __shared__ float sh[8];
    const int64_t t = blockIdx.x;
    const uint16_t* xt = X + t * K;
    float acc = 0.0f;
    for (int64_t k = threadIdx.x; k < K; k += 256) {
        const float xv = __uint_as_float((uint32_t) xt[k] << 16);
        const float wv = __uint_as_float((uint32_t) w[k] << 16);
        acc = fmaf(xv, wv, acc);
    }
    const float s = block_sum(acc, sh);
    if (threadIdx.x == 0) y[t] = s;
}
}  // namespace
void bf16_dot_row(const uint16_t* X, const uint16_t* w, float* y, int64_t T, int64_t K, void* stream) {
    if (T <= 0 || K <= 0) return;
    bf16_dot_row_kernel<<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(X, w, y, K);
    check("bf16_dot_row");
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    rms_rows_kernel<<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, w, cols, ld, eps);
    check("rms_rows");
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float freq_base, void* stream) {
    const float theta_scale = powf(freq_base, -2.0f / 64.0f);
    rope_kernel<<<(unsigned) (T * heads), 32, 0, (cudaStream_t) stream>>>(x, heads, dim, ld, pos0, theta_scale, strata::kernels::mrope_table());
    check("rope");
}
void split_q(const float* q_full, float* q, int64_t T, void* stream, int64_t nh) {
    if (T <= 0 || nh <= 0) return;
    split_q_kernel<<<(unsigned) (T * nh), 64, 0, (cudaStream_t) stream>>>(q_full, q, T, nh);
    check("split_q");
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream, int64_t nh) {
    if (T <= 0 || nh <= 0) return;
    gate_attn_kernel<<<(unsigned) (T * nh), 64, 0, (cudaStream_t) stream>>>(attn, q_full, out16, T, nh);
    check("gate_attn");
}

}  // namespace strata::prefill
