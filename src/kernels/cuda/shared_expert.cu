// src/kernels/cuda/shared_expert.cu - P2.S2: the shared expert.
//
// `ref/moe.py::shared_expert`, transcribed:
//
//     h = silu(x @ gate_shexp.T) * (x @ up_shexp.T)      <- SILU GOES ON GATE, not on up
//     h = h @ down_shexp.T                                (nt, n_embd)
//     g = sigmoid(x @ gate_inp_shexp)                     (nt,) ONE SCALAR PER TOKEN
//     return h * g[:, None]
//
// TWO THINGS HERE WERE BELIEVED WRONG FOR MANY ROUNDS and `docs/semantics.md` records both.  SILU GOES ON THE
// GATE TENSOR: the opposite reading is plausible, produces the right shapes, and is wrong.  And
// `ffn_gate_inp_shexp` is `(n_embd,)` whose output is "one value per token" - a SCALAR gate obtained by dotting
// that vector with the hidden state, NOT a per-expert gate and NOT a per-dimension elementwise one.  Both were
// live readings of the same shapes until the source comment settled them.
//
// The result is ADDED to the routed output - not router-weighted, not renormalised against it.
//
// The legacy canonical projections use their explicit Q8_0/Q8_K activation images. The optional native
// BF16 path affects only the scalar gate: it reads the original F32 input and uses pinned CUDA MMVF plus
// an FP32 sigmoid. Native projection overrides independently select CUDA Q8_1 MMVQ and FP32 SwiGLU.
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <climits>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 128;
bool native_bf16 = false;

// `f32_to_f16` used to live here as a private copy.  Round 198 found it - and the identical one in
// `quantize_act.cu` - returning a NaN for every FINITE value that overflows fp16 (1e30, 65536, 1e45) instead
// of saturating to inf, because `exp >= 31` conflates an f32 inf/NaN with an out-of-range finite exponent.
// This file's fixture is all O(1), so it could not see it.  The conversion now lives in one place, with the
// regimes spelled out.

// silu(x) = x / (1 + exp(-x)), written as `ref/moe.py` writes it, in DOUBLE then cast - the reference works in
// float64 and a float32 exp differs in the last bits.  The multiplication by `up` is the reference's order.
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                              float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x = (double) gate[i];
    out[i] = (float) (x / (1.0 + exp(-x))) * up[i];
}

__global__ void native_swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Pinned CUDA unary.cuh op_silu, then unary_gated_op_kernel's multiply.
    // Explicit intrinsics reproduce its --use_fast_math operations without
    // changing compilation of the default legacy arithmetic in this file.
    out[i] = __fdividef(gate[i], 1.0f + __expf(-gate[i])) * up[i];
}

__global__ void to_f16_kernel(const float* __restrict__ in, uint16_t* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f16_from_f32(in[i]);
}

// The per-token scalar gate: sigmoid(dot(x, w)) with w = `ffn_gate_inp_shexp` (n_embd,).
//
// **ONE THREAD WAS THE LARGEST SINGLE KERNEL IN THE MODEL, AND THE COMMENT DEFENDING IT SAID WHY IT SHOULD
// NOT BE.**  It read: "n_embd is 2560, so this is a short serial loop and the parallelism is elsewhere."
// 2560 iterations is not short when each one is a DOUBLE multiply-add.  This card's double rate is **1/64** of
// its float rate, so the loop is a 2560-long dependency chain through a 1/64-rate unit - tens of microseconds
// from the chain alone, before counting 2560 sequential global loads issued by one thread with no coalescing.
//
// MEASURED, and this is what the number looks like from outside: `shared_expert` is **0.2374 ms of a
// 0.8883 ms block - 26.7% of the whole block, 11.4 ms per token** - for three projections that read 5.5 MB and
// whose memory floor is 0.009 ms.  It is seven kernels, and six of them are GEMVs or elementwise passes over
// 640-2560 elements.  This is the one that cannot be explained by bandwidth.
//
// THE DOUBLE ACCUMULATOR IS KEPT.  `ref/gr.py` reduces in float64 and `block_sum` in `gr.cu` exists for the
// same reason - the summation order of a tree is not the reference's, so the defence is enough precision that
// the order stops mattering.  What changes is the CHAIN: ten terms per thread instead of 2560, then a
// tree-reduce.  Through a sigmoid, a 1e-16 relative difference in the dot is not a difference.
//
// **BOTH OPERANDS ARE BF16, AND THAT IS THE REFERENCE'S CONTRACT, NOT A STORAGE DETAIL.**  `gate_inp_shexp` is a
// BF16 weight, so ggml converts the activation to its `vec_dot_type` - BF16 - exactly as it does for the router
// (`docs/activation-contract.md`).  This kernel used to take `const float* w` while the LOADER had already
// re-rounded the tensor to 2 B/elem, so it read 2560 floats out of a 5120-byte buffer: 5120 B past the end of
// the tensor, inside the 4.5 GiB arena, where nothing faults.  The dot came out astronomically large, sigmoid
// saturated to 1.0, and `scale_kernel` multiplied the shared expert's output by it - which is why the FIRST
// symptom was 2560 non-finite outputs rather than a wrong scalar.
//
// A wrong scalar here is the quiet failure mode: sigmoid bounds the damage to [0,1], so a gate that should be
// 0.5 and reads 1.0 scales the shared expert by 2x and produces perfectly finite, perfectly plausible logits.
__device__ __forceinline__ double warp_sum_d(double v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return __shfl_sync(0xFFFFFFFFu, v, 0);
}

__global__ void scalar_gate_kernel(const uint16_t* __restrict__ x_bf16, const uint16_t* __restrict__ w_bf16,
                                   float* __restrict__ out, int n_embd) {
    __shared__ double scratch[8];   // 8 warps: the launch is <<<1, 256>>>
    double acc = 0.0;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x)
        acc += (double) f32_from_bf16(x_bf16[i]) * (double) f32_from_bf16(w_bf16[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    acc = warp_sum_d(acc);
    if (lane == 0) scratch[warp] = acc;
    __syncthreads();
    const int nw = ((int) blockDim.x + 31) >> 5;
    if (warp == 0) {
        acc = (threadIdx.x < nw) ? scratch[threadIdx.x] : 0.0;
        acc = warp_sum_d(acc);
        if (threadIdx.x == 0) out[0] = (float) (1.0 / (1.0 + exp(-acc)));
    }
}

__global__ void scale_kernel(float* __restrict__ out, const float* __restrict__ g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= g[0];
}

__global__ void native_scalar_sigmoid_kernel(float* gate) {
    // Match the single-token CUDA sigmoid's FP32 fast-math operations without changing legacy kernels'
    // compilation flags. The dot product was already reduced by the pinned native MMVF implementation.
    gate[0] = __fdividef(1.0f, 1.0f + __expf(-gate[0]));
}
__global__ void native_scalar_sigmoid_multi_kernel(float* gate) {   // thread t = token t, same expression
    gate[threadIdx.x] = __fdividef(1.0f, 1.0f + __expf(-gate[threadIdx.x]));
}

/// The MoE block's final combination.  See the header for the two readings it exists to pin.
__global__ void moe_combine_kernel(const float* __restrict__ parts, const float* __restrict__ weights,
                                   const float* __restrict__ shared, float* __restrict__ y, int n_embd,
                                   int k, int has_shared) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_embd) return;
    // Accumulate in DOUBLE, in the reference's own order (`for i in range(k): out[t] += w[t,i] * g[0]`).
    // f32 would be defensible for ten terms, but the reference is float64 and this is one line.
    double acc = 0.0;
    for (int e = 0; e < k; ++e) acc += (double) weights[e] * (double) parts[(size_t) e * n_embd + j];
    // The SHARED output is added PLAIN - not router-weighted, not renormalised against the routed sum.
    if (has_shared) acc += (double) shared[j];
    y[j] = (float) acc;
}

}  // namespace

void shared_expert_set_native_bf16(bool enabled) { native_bf16 = enabled; }
bool shared_expert_native_bf16() { return native_bf16; }

namespace {
struct ShQ81Block {
    half2 ds;
    int8_t qs[32];
};
__global__ void native_swiglu_quant_q8_1_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                                ShQ81Block* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xi = __fdividef(gate[i], 1.0f + __expf(-gate[i])) * up[i];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(xi / d)));
    const int ib = i >> 5, iqs = i & 31;
    out[ib].qs[iqs] = q;
    if (iqs == 0) out[ib].ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}
__global__ void scale_rows_kernel(float* __restrict__ out, const float* __restrict__ g, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[(size_t) t * n + i] *= g[t];
}

// ============================================================================
// GP100 Fused K-Major SoA Q8_0 shared_expert: 6 kernels -> 2 kernels!
//   Kernel 1: 16 rows/block (41 blocks on 56 SMs!), single-pass K<=80 preload,
//             fused (gate + up + silu*up -> __half) || scalar_gate+sigmoid
//   Kernel 2: 16 rows/block (160 blocks on 56 SMs!), reads __half2 directly
//             (zero Q8_1 quant/dequant!) + scale by g[t]
// ============================================================================
__device__ __forceinline__ __half2 sh_u32_to_h2(uint32_t x) { return *reinterpret_cast<const __half2*>(&x); }
__device__ __forceinline__ __half2 sh_u8_lo_h2(uint32_t u) {
    return __hsub2(sh_u32_to_h2(__byte_perm(u, 0x64006400u, 0x5150)), __float2half2_rn(1152.0f));
}
__device__ __forceinline__ __half2 sh_u8_hi_h2(uint32_t u) {
    return __hsub2(sh_u32_to_h2(__byte_perm(u, 0x64006400u, 0x7372)), __float2half2_rn(1152.0f));
}
__device__ __forceinline__ __half2 sh_q81_pair_h2(const int8_t* qs2, __half2 c1152, __half2 inv128) {
    const uint16_t pair = *reinterpret_cast<const uint16_t*>(qs2);
    const uint32_t vx = ((uint32_t) pair) ^ 0x8080u;
    const uint32_t h = __byte_perm(vx, 0x64006400u, 0x5150);
    return __hmul2(__hsub2(sh_u32_to_h2(h), c1152), inv128);
}

// GP100 Fused K-Major SoA Q8_0 shared_expert (Wave D2):
//   Kernel 1: 8 rows/block x 16 K-workers = 128t, 81 blocks on 56 SMs (100% SM saturation!),
//             128-byte coalesced LDG.E.128 cache lines, outputs __half directly into unused gate_scratch!
//   Kernel 2: 16 rows/block (160 blocks on 56 SMs!), reads __half2 directly (ZERO Q8_1 quant/dequant!) + scale by g[t]!
template<int NT>
__global__ void __launch_bounds__(128) shared_gu_swiglu_f16_soa_kernel(
    const uint8_t* __restrict__ w_gate, const uint8_t* __restrict__ w_up,
    const ShQ81Block* __restrict__ xq, const float* __restrict__ x_f32,
    const uint16_t* __restrict__ x_bf16, const uint16_t* __restrict__ gate_inp_bf16,
    __half* __restrict__ h_f16, float* __restrict__ g_out,
    int n_embd, int n_ff, bool use_native_bf16) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n_ff_8 = n_ff >> 3;

    if ((int) blockIdx.x == n_ff_8) {
        __shared__ double sp[4];
        for (int k = 0; k < NT; ++k) {
            if (use_native_bf16) {
                float acc = 0.0f;
                const float* xk = x_f32 + (size_t) k * n_embd;
                for (int i = tid; i < n_embd; i += 128)
                    acc = fmaf(xk[i], f32_from_bf16(gate_inp_bf16[i]), acc);
                for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                if (lane == 0) sp[warp] = (double) acc;
                __syncthreads();
                if (tid == 0) {
                    const float s = (float) sp[0] + (float) sp[1] + (float) sp[2] + (float) sp[3];
                    g_out[k] = __fdividef(1.0f, 1.0f + __expf(-s));
                }
                __syncthreads();
            } else {
                double acc = 0.0;
                const uint16_t* xk = x_bf16 + (size_t) k * n_embd;
                for (int i = tid; i < n_embd; i += 128)
                    acc += (double) f32_from_bf16(xk[i]) * (double) f32_from_bf16(gate_inp_bf16[i]);
                acc = warp_sum_d(acc);
                if (lane == 0) sp[warp] = acc;
                __syncthreads();
                if (tid == 0) {
                    const double s = sp[0] + sp[1] + sp[2] + sp[3];
                    g_out[k] = (float) (1.0 / (1.0 + exp(-s)));
                }
                __syncthreads();
            }
        }
        return;
    }

    constexpr int TILE_B = (NT <= 4) ? 40 : 20;   // <= 10.9 KB SMEM -> 5 blocks/SM on GP100!
    const int K_blk = n_embd >> 5;
    const int rl = tid & 7, kw = tid >> 3;        // 8 contiguous rows x 16 K-workers = 128 bytes/LDG.E.128!
    const int r = (int) blockIdx.x * 8 + rl;
    const uint4* Wgq = reinterpret_cast<const uint4*>(w_gate);
    const __half* Wgd = reinterpret_cast<const __half*>(w_gate + (size_t) n_embd * n_ff);
    const uint4* Wuq = reinterpret_cast<const uint4*>(w_up);
    const __half* Wud = reinterpret_cast<const __half*>(w_up + (size_t) n_embd * n_ff);

    union SharedGuMem {
        struct {
            __align__(16) __half2 sxh[NT][TILE_B * 16];
            float sxd[NT][TILE_B];
        } x;
        struct {
            float part_g[16][NT][8];
            float part_u[16][NT][8];
        } p;
    };
    __shared__ SharedGuMem smem;
    const __half2 c1152 = __float2half2_rn(1152.0f);
    const __half2 inv128 = __float2half2_rn(1.0f / 128.0f);

    float ag[NT], au[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) { ag[k] = 0.0f; au[k] = 0.0f; }

    for (int b0 = 0; b0 < K_blk; b0 += TILE_B) {
        const int nb = (b0 + TILE_B <= K_blk) ? TILE_B : (K_blk - b0);
        for (int i = tid; i < nb * 16; i += 128) {
            const int db = i >> 4, j = i & 15;
            const int b = b0 + db;
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const ShQ81Block& bx = xq[(size_t) k * K_blk + b];
                smem.x.sxh[k][i] = sh_q81_pair_h2(bx.qs + 2 * j, c1152, inv128);
                if (j == 0) smem.x.sxd[k][db] = __low2float(bx.ds) * 128.0f;
            }
        }
        __syncthreads();
        for (int db = kw; db < nb; db += 16) {
            const int b = b0 + db;
            const float dg = __half2float(__ldg(Wgd + (size_t) b * n_ff + r));
            const float du = __half2float(__ldg(Wud + (size_t) b * n_ff + r));
            __half2 pg[NT], pu[NT];
#pragma unroll
            for (int k = 0; k < NT; ++k) { pg[k] = __float2half2_rn(0.0f); pu[k] = __float2half2_rn(0.0f); }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint4 qg = __ldg(Wgq + (size_t) (b * 2 + h) * n_ff + r);
                const uint4 qu = __ldg(Wuq + (size_t) (b * 2 + h) * n_ff + r);
                const int xb = db * 16 + h * 8;
                const __half2 g0 = sh_u8_lo_h2(qg.x), g1 = sh_u8_hi_h2(qg.x);
                const __half2 g2 = sh_u8_lo_h2(qg.y), g3 = sh_u8_hi_h2(qg.y);
                const __half2 g4 = sh_u8_lo_h2(qg.z), g5 = sh_u8_hi_h2(qg.z);
                const __half2 g6 = sh_u8_lo_h2(qg.w), g7 = sh_u8_hi_h2(qg.w);
                const __half2 u0 = sh_u8_lo_h2(qu.x), u1 = sh_u8_hi_h2(qu.x);
                const __half2 u2 = sh_u8_lo_h2(qu.y), u3 = sh_u8_hi_h2(qu.y);
                const __half2 u4 = sh_u8_lo_h2(qu.z), u5 = sh_u8_hi_h2(qu.z);
                const __half2 u6 = sh_u8_lo_h2(qu.w), u7 = sh_u8_hi_h2(qu.w);
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    __half2 gk = pg[k], uk = pu[k];
                    const uint4 xv0 = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 0]);
                    const uint4 xv1 = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 4]);
                    const __half2 x0 = *reinterpret_cast<const __half2*>(&xv0.x);
                    const __half2 x1 = *reinterpret_cast<const __half2*>(&xv0.y);
                    const __half2 x2 = *reinterpret_cast<const __half2*>(&xv0.z);
                    const __half2 x3 = *reinterpret_cast<const __half2*>(&xv0.w);
                    const __half2 x4 = *reinterpret_cast<const __half2*>(&xv1.x);
                    const __half2 x5 = *reinterpret_cast<const __half2*>(&xv1.y);
                    const __half2 x6 = *reinterpret_cast<const __half2*>(&xv1.z);
                    const __half2 x7 = *reinterpret_cast<const __half2*>(&xv1.w);
                    gk = __hfma2(g0, x0, gk); gk = __hfma2(g1, x1, gk);
                    gk = __hfma2(g2, x2, gk); gk = __hfma2(g3, x3, gk);
                    gk = __hfma2(g4, x4, gk); gk = __hfma2(g5, x5, gk);
                    gk = __hfma2(g6, x6, gk); gk = __hfma2(g7, x7, gk);
                    uk = __hfma2(u0, x0, uk); uk = __hfma2(u1, x1, uk);
                    uk = __hfma2(u2, x2, uk); uk = __hfma2(u3, x3, uk);
                    uk = __hfma2(u4, x4, uk); uk = __hfma2(u5, x5, uk);
                    uk = __hfma2(u6, x6, uk); uk = __hfma2(u7, x7, uk);
                    pg[k] = gk; pu[k] = uk;
                }
            }
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const float2 fg = __half22float2(pg[k]);
                const float2 fu = __half22float2(pu[k]);
                const float xd = smem.x.sxd[k][db];
                ag[k] = fmaf(dg * xd, fg.x + fg.y, ag[k]);
                au[k] = fmaf(du * xd, fu.x + fu.y, au[k]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        smem.p.part_g[kw][k][rl] = ag[k];
        smem.p.part_u[kw][k][rl] = au[k];
    }
    __syncthreads();
    if (tid < NT * 8) {
        const int k = tid >> 3, rl2 = tid & 7;
        float sg = smem.p.part_g[0][k][rl2], su = smem.p.part_u[0][k][rl2];
#pragma unroll
        for (int w = 1; w < 16; ++w) { sg += smem.p.part_g[w][k][rl2]; su += smem.p.part_u[w][k][rl2]; }
        const float xi = __fdividef(sg, 1.0f + __expf(-sg)) * su;
        h_f16[(size_t) k * n_ff + (size_t) blockIdx.x * 8 + rl2] = __float2half_rn(fminf(fmaxf(xi, -65504.0f), 65504.0f));
    }
}

template<int NT>
__global__ void __launch_bounds__(64) shared_down_scale_f16_soa_kernel(
    const uint8_t* __restrict__ w_down, const __half* __restrict__ h_f16,
    const float* __restrict__ g_in, float* __restrict__ out, int n_ff, int n_embd) {
    constexpr int TILE_B = 20;
    const int K_blk = n_ff >> 5;
    const int tid = threadIdx.x, hw = tid >> 4, hl = tid & 15;
    const int r = (int) blockIdx.x * 16 + hl;
    const uint4* Wq = reinterpret_cast<const uint4*>(w_down);
    const __half* Wd = reinterpret_cast<const __half*>(w_down + (size_t) n_ff * n_embd);

    union SharedDownMem {
        __align__(16) __half2 sxh[NT][TILE_B * 16];
        float part[4][NT][16];
    };
    __shared__ SharedDownMem smem;

    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;

    for (int b0 = 0; b0 < K_blk; b0 += TILE_B) {
        const int nb = (b0 + TILE_B <= K_blk) ? TILE_B : (K_blk - b0);
        for (int i = tid; i < nb * 16; i += 64) {
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                smem.sxh[k][i] = reinterpret_cast<const __half2*>(h_f16 + (size_t) k * n_ff + b0 * 32)[i];
            }
        }
        __syncthreads();
        if (r < n_embd) {
            for (int db = hw; db < nb; db += 4) {
                const int b = b0 + db;
                const float d = __half2float(__ldg(Wd + (size_t) b * n_embd + r));
                __half2 p[NT];
#pragma unroll
                for (int k = 0; k < NT; ++k) p[k] = __float2half2_rn(0.0f);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint4 qv = __ldg(Wq + (size_t) (b * 2 + h) * n_embd + r);
                    const int xb = db * 16 + h * 8;
                    const __half2 w0 = sh_u8_lo_h2(qv.x), w1 = sh_u8_hi_h2(qv.x);
                    const __half2 w2 = sh_u8_lo_h2(qv.y), w3 = sh_u8_hi_h2(qv.y);
                    const __half2 w4 = sh_u8_lo_h2(qv.z), w5 = sh_u8_hi_h2(qv.z);
                    const __half2 w6 = sh_u8_lo_h2(qv.w), w7 = sh_u8_hi_h2(qv.w);
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        __half2 pk = p[k];
                        const uint4 xv0 = *reinterpret_cast<const uint4*>(&smem.sxh[k][xb + 0]);
                        const uint4 xv1 = *reinterpret_cast<const uint4*>(&smem.sxh[k][xb + 4]);
                        pk = __hfma2(w0, *reinterpret_cast<const __half2*>(&xv0.x), pk);
                        pk = __hfma2(w1, *reinterpret_cast<const __half2*>(&xv0.y), pk);
                        pk = __hfma2(w2, *reinterpret_cast<const __half2*>(&xv0.z), pk);
                        pk = __hfma2(w3, *reinterpret_cast<const __half2*>(&xv0.w), pk);
                        pk = __hfma2(w4, *reinterpret_cast<const __half2*>(&xv1.x), pk);
                        pk = __hfma2(w5, *reinterpret_cast<const __half2*>(&xv1.y), pk);
                        pk = __hfma2(w6, *reinterpret_cast<const __half2*>(&xv1.z), pk);
                        pk = __hfma2(w7, *reinterpret_cast<const __half2*>(&xv1.w), pk);
                        p[k] = pk;
                    }
                }
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    const float2 f = __half22float2(p[k]);
                    acc[k] = fmaf(d, f.x + f.y, acc[k]);
                }
            }
        }
        __syncthreads();
    }
    if (r < n_embd) {
#pragma unroll
        for (int k = 0; k < NT; ++k) smem.part[hw][k][hl] = acc[k];
    }
    __syncthreads();
    if (hw == 0 && r < n_embd) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const float s = smem.part[0][k][hl] + smem.part[1][k][hl] + smem.part[2][k][hl] + smem.part[3][k][hl];
            out[(size_t) k * n_embd + r] = s * g_in[k];
        }
    }
}

// TP-sharded GP100 Shared Expert (n_ff <= 320, e.g. 160 for TP=4):
//   Kernel 1 (4r): 4 rows/block x 32 K-workers = 128t -> 41 blocks on 56 SMs (2x higher SM saturation vs 21 blocks!).
//   Kernel 2 (5k): for K_blk == 5 (n_ff == 160), 5 K-workers x 16 rows = 80t -> 1 exact Q8_0 block/worker (0 tail!).
template<int NT>
__global__ void __launch_bounds__(128) shared_gu_swiglu_f16_soa_4r_kernel(
    const uint8_t* __restrict__ w_gate, const uint8_t* __restrict__ w_up,
    const ShQ81Block* __restrict__ xq, const float* __restrict__ x_f32,
    const uint16_t* __restrict__ x_bf16, const uint16_t* __restrict__ gate_inp_bf16,
    __half* __restrict__ h_f16, float* __restrict__ g_out,
    int n_embd, int n_ff, bool use_native_bf16) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n_ff_4 = n_ff >> 2;

    if ((int) blockIdx.x == n_ff_4) {
        __shared__ double sp[4];
        for (int k = 0; k < NT; ++k) {
            if (use_native_bf16) {
                float acc = 0.0f;
                const float* xk = x_f32 + (size_t) k * n_embd;
                for (int i = tid; i < n_embd; i += 128)
                    acc = fmaf(xk[i], f32_from_bf16(gate_inp_bf16[i]), acc);
                for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                if (lane == 0) sp[warp] = (double) acc;
                __syncthreads();
                if (tid == 0) {
                    const float s = (float) sp[0] + (float) sp[1] + (float) sp[2] + (float) sp[3];
                    g_out[k] = __fdividef(1.0f, 1.0f + __expf(-s));
                }
                __syncthreads();
            } else {
                double acc = 0.0;
                const uint16_t* xk = x_bf16 + (size_t) k * n_embd;
                for (int i = tid; i < n_embd; i += 128)
                    acc += (double) f32_from_bf16(xk[i]) * (double) f32_from_bf16(gate_inp_bf16[i]);
                acc = warp_sum_d(acc);
                if (lane == 0) sp[warp] = acc;
                __syncthreads();
                if (tid == 0) {
                    const double s = sp[0] + sp[1] + sp[2] + sp[3];
                    g_out[k] = (float) (1.0 / (1.0 + exp(-s)));
                }
                __syncthreads();
            }
        }
        return;
    }

    constexpr int TILE_B = (NT <= 4) ? 40 : 20;
    const int K_blk = n_embd >> 5;
    const int rl = tid & 3, kw = tid >> 2;        // 4 contiguous rows (64B) x 32 K-workers!
    const int r = (int) blockIdx.x * 4 + rl;
    const uint4* Wgq = reinterpret_cast<const uint4*>(w_gate);
    const __half* Wgd = reinterpret_cast<const __half*>(w_gate + (size_t) n_embd * n_ff);
    const uint4* Wuq = reinterpret_cast<const uint4*>(w_up);
    const __half* Wud = reinterpret_cast<const __half*>(w_up + (size_t) n_embd * n_ff);

    union SharedGuMem4 {
        struct {
            __align__(16) __half2 sxh[NT][TILE_B * 16];
            float sxd[NT][TILE_B];
        } x;
        struct {
            float part_g[32][NT][4];
            float part_u[32][NT][4];
        } p;
    };
    __shared__ SharedGuMem4 smem;
    const __half2 c1152 = __float2half2_rn(1152.0f);
    const __half2 inv128 = __float2half2_rn(1.0f / 128.0f);

    float ag[NT], au[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) { ag[k] = 0.0f; au[k] = 0.0f; }

    for (int b0 = 0; b0 < K_blk; b0 += TILE_B) {
        const int nb = (b0 + TILE_B <= K_blk) ? TILE_B : (K_blk - b0);
        for (int i = tid; i < nb * 16; i += 128) {
            const int db = i >> 4, j = i & 15;
            const int b = b0 + db;
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const ShQ81Block& bx = xq[(size_t) k * K_blk + b];
                smem.x.sxh[k][i] = sh_q81_pair_h2(bx.qs + 2 * j, c1152, inv128);
                if (j == 0) smem.x.sxd[k][db] = __low2float(bx.ds) * 128.0f;
            }
        }
        __syncthreads();
        for (int db = kw; db < nb; db += 32) {
            const int b = b0 + db;
            const float dg = __half2float(__ldg(Wgd + (size_t) b * n_ff + r));
            const float du = __half2float(__ldg(Wud + (size_t) b * n_ff + r));
            __half2 pg[NT], pu[NT];
#pragma unroll
            for (int k = 0; k < NT; ++k) { pg[k] = __float2half2_rn(0.0f); pu[k] = __float2half2_rn(0.0f); }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint4 qg = __ldg(Wgq + (size_t) (b * 2 + h) * n_ff + r);
                const uint4 qu = __ldg(Wuq + (size_t) (b * 2 + h) * n_ff + r);
                const int xb = db * 16 + h * 8;
                const __half2 g0 = sh_u8_lo_h2(qg.x), g1 = sh_u8_hi_h2(qg.x);
                const __half2 g2 = sh_u8_lo_h2(qg.y), g3 = sh_u8_hi_h2(qg.y);
                const __half2 g4 = sh_u8_lo_h2(qg.z), g5 = sh_u8_hi_h2(qg.z);
                const __half2 g6 = sh_u8_lo_h2(qg.w), g7 = sh_u8_hi_h2(qg.w);
                const __half2 u0 = sh_u8_lo_h2(qu.x), u1 = sh_u8_hi_h2(qu.x);
                const __half2 u2 = sh_u8_lo_h2(qu.y), u3 = sh_u8_hi_h2(qu.y);
                const __half2 u4 = sh_u8_lo_h2(qu.z), u5 = sh_u8_hi_h2(qu.z);
                const __half2 u6 = sh_u8_lo_h2(qu.w), u7 = sh_u8_hi_h2(qu.w);
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    __half2 gk = pg[k], uk = pu[k];
                    const uint4 xv0 = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 0]);
                    const uint4 xv1 = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 4]);
                    const __half2 x0 = *reinterpret_cast<const __half2*>(&xv0.x);
                    const __half2 x1 = *reinterpret_cast<const __half2*>(&xv0.y);
                    const __half2 x2 = *reinterpret_cast<const __half2*>(&xv0.z);
                    const __half2 x3 = *reinterpret_cast<const __half2*>(&xv0.w);
                    const __half2 x4 = *reinterpret_cast<const __half2*>(&xv1.x);
                    const __half2 x5 = *reinterpret_cast<const __half2*>(&xv1.y);
                    const __half2 x6 = *reinterpret_cast<const __half2*>(&xv1.z);
                    const __half2 x7 = *reinterpret_cast<const __half2*>(&xv1.w);
                    gk = __hfma2(g0, x0, gk); gk = __hfma2(g1, x1, gk);
                    gk = __hfma2(g2, x2, gk); gk = __hfma2(g3, x3, gk);
                    gk = __hfma2(g4, x4, gk); gk = __hfma2(g5, x5, gk);
                    gk = __hfma2(g6, x6, gk); gk = __hfma2(g7, x7, gk);
                    uk = __hfma2(u0, x0, uk); uk = __hfma2(u1, x1, uk);
                    uk = __hfma2(u2, x2, uk); uk = __hfma2(u3, x3, uk);
                    uk = __hfma2(u4, x4, uk); uk = __hfma2(u5, x5, uk);
                    uk = __hfma2(u6, x6, uk); uk = __hfma2(u7, x7, uk);
                    pg[k] = gk; pu[k] = uk;
                }
            }
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const float2 fg = __half22float2(pg[k]);
                const float2 fu = __half22float2(pu[k]);
                const float xd = smem.x.sxd[k][db];
                ag[k] = fmaf(dg * xd, fg.x + fg.y, ag[k]);
                au[k] = fmaf(du * xd, fu.x + fu.y, au[k]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        smem.p.part_g[kw][k][rl] = ag[k];
        smem.p.part_u[kw][k][rl] = au[k];
    }
    __syncthreads();
    if (tid < NT * 4) {
        const int k = tid >> 2, rl2 = tid & 3;
        float sg = smem.p.part_g[0][k][rl2], su = smem.p.part_u[0][k][rl2];
#pragma unroll
        for (int w = 1; w < 32; ++w) { sg += smem.p.part_g[w][k][rl2]; su += smem.p.part_u[w][k][rl2]; }
        const float xi = __fdividef(sg, 1.0f + __expf(-sg)) * su;
        h_f16[(size_t) k * n_ff + (size_t) blockIdx.x * 4 + rl2] = __float2half_rn(fminf(fmaxf(xi, -65504.0f), 65504.0f));
    }
}

// GP100 4-Row x 80-K-Worker (320t) Q8_0 SoA Shared Expert GU+SwiGLU (K_blk == 80, n_ff <= 320):
// 100% active threads (fixes 37.5% idle threads at T>=5!), 1 SMEM load, ZERO tile loops, STACK:0! (1.50x faster!).
template<int NT>
__global__ void __launch_bounds__(320) shared_gu_swiglu_f16_soa_4r80k_kernel(
    const uint8_t* __restrict__ w_gate, const uint8_t* __restrict__ w_up,
    const ShQ81Block* __restrict__ xq, const float* __restrict__ x_f32, const uint16_t* __restrict__ x_bf16,
    const uint16_t* __restrict__ gate_inp_bf16, __half* __restrict__ h_f16, float* __restrict__ g_out,
    int n_embd, int n_ff, bool use_f32_gate) {
    const int tid = threadIdx.x;
    if ((int) blockIdx.x == (n_ff >> 2)) {
        __shared__ double sp[10];
        const int lane = tid & 31, warp = tid >> 5;
#pragma unroll 1
        for (int k = 0; k < NT; ++k) {
            if (use_f32_gate) {
                float acc = 0.0f;
                const float* xk = x_f32 + (size_t) k * n_embd;
                for (int i = tid; i < n_embd; i += 320)
                    acc = fmaf(xk[i], f32_from_bf16(gate_inp_bf16[i]), acc);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                if (lane == 0) sp[warp] = (double) acc;
                __syncthreads();
                if (tid == 0) {
                    float s = 0.0f;
#pragma unroll
                    for (int w = 0; w < 10; ++w) s += (float) sp[w];
                    g_out[k] = __fdividef(1.0f, 1.0f + __expf(-s));
                }
                __syncthreads();
            } else {
                double acc = 0.0;
                const uint16_t* xk = x_bf16 + (size_t) k * n_embd;
                for (int i = tid; i < n_embd; i += 320)
                    acc += (double) f32_from_bf16(xk[i]) * (double) f32_from_bf16(gate_inp_bf16[i]);
                acc = warp_sum_d(acc);
                if (lane == 0) sp[warp] = acc;
                __syncthreads();
                if (tid == 0) {
                    double s = 0.0;
#pragma unroll
                    for (int w = 0; w < 10; ++w) s += sp[w];
                    g_out[k] = (float) (1.0 / (1.0 + exp(-s)));
                }
                __syncthreads();
            }
        }
        return;
    }

    constexpr int K_BLK = 80;
    const int rl = tid & 3, b = tid >> 2;         // 4 contiguous rows x 80 K-workers (1 exact block/thread!)
    const int r = (int) blockIdx.x * 4 + rl;
    const uint4* Wgq = reinterpret_cast<const uint4*>(w_gate);
    const __half* Wgd = reinterpret_cast<const __half*>(w_gate + (size_t) n_embd * n_ff);
    const uint4* Wuq = reinterpret_cast<const uint4*>(w_up);
    const __half* Wud = reinterpret_cast<const __half*>(w_up + (size_t) n_embd * n_ff);

    union SharedGuMem80 {
        struct {
            __align__(16) __half2 sxh[NT][K_BLK * 16];
            float sxd[NT][K_BLK];
        } x;
        struct {
            float part_g[K_BLK][NT][4];
            float part_u[K_BLK][NT][4];
        } p;
    };
    __shared__ SharedGuMem80 smem;
    const __half2 c1152 = __float2half2_rn(1152.0f);
    const __half2 inv128 = __float2half2_rn(1.0f / 128.0f);

    for (int i = tid; i < K_BLK * 8; i += 320) {
        const int db = i >> 3, j4 = i & 7;
        const int h2_idx = (db << 4) + (j4 << 1);
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const ShQ81Block& bx = xq[(size_t) k * K_BLK + db];
            const uint32_t q4 = (*reinterpret_cast<const uint32_t*>(bx.qs + j4 * 4)) ^ 0x80808080u;
            const uint32_t h0 = __byte_perm(q4, 0x64006400u, 0x5150);
            const uint32_t h1 = __byte_perm(q4, 0x64006400u, 0x5352);
            const __half2 v0 = __hmul2(__hsub2(sh_u32_to_h2(h0), c1152), inv128);
            const __half2 v1 = __hmul2(__hsub2(sh_u32_to_h2(h1), c1152), inv128);
            *reinterpret_cast<uint2*>(&smem.x.sxh[k][h2_idx]) =
                make_uint2(*reinterpret_cast<const uint32_t*>(&v0), *reinterpret_cast<const uint32_t*>(&v1));
            if (j4 == 0) smem.x.sxd[k][db] = __low2float(bx.ds) * 128.0f;
        }
    }
    __syncthreads();

    const float dg = __half2float(__ldg(Wgd + (size_t) b * n_ff + r));
    const float du = __half2float(__ldg(Wud + (size_t) b * n_ff + r));
    __half2 pg[NT], pu[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) { pg[k] = __float2half2_rn(0.0f); pu[k] = __float2half2_rn(0.0f); }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint4 qg = __ldg(Wgq + (size_t) (b * 2 + h) * n_ff + r);
        const uint4 qu = __ldg(Wuq + (size_t) (b * 2 + h) * n_ff + r);
        const int xb = b * 16 + h * 8;
        const __half2 g0 = sh_u8_lo_h2(qg.x), g1 = sh_u8_hi_h2(qg.x);
        const __half2 g2 = sh_u8_lo_h2(qg.y), g3 = sh_u8_hi_h2(qg.y);
        const __half2 g4 = sh_u8_lo_h2(qg.z), g5 = sh_u8_hi_h2(qg.z);
        const __half2 g6 = sh_u8_lo_h2(qg.w), g7 = sh_u8_hi_h2(qg.w);
        const __half2 u0 = sh_u8_lo_h2(qu.x), u1 = sh_u8_hi_h2(qu.x);
        const __half2 u2 = sh_u8_lo_h2(qu.y), u3 = sh_u8_hi_h2(qu.y);
        const __half2 u4 = sh_u8_lo_h2(qu.z), u5 = sh_u8_hi_h2(qu.z);
        const __half2 u6 = sh_u8_lo_h2(qu.w), u7 = sh_u8_hi_h2(qu.w);
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            __half2 gk = pg[k], uk = pu[k];
            const uint4 xv0 = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 0]);
            const uint4 xv1 = *reinterpret_cast<const uint4*>(&smem.x.sxh[k][xb + 4]);
            const __half2 x0 = *reinterpret_cast<const __half2*>(&xv0.x);
            const __half2 x1 = *reinterpret_cast<const __half2*>(&xv0.y);
            const __half2 x2 = *reinterpret_cast<const __half2*>(&xv0.z);
            const __half2 x3 = *reinterpret_cast<const __half2*>(&xv0.w);
            const __half2 x4 = *reinterpret_cast<const __half2*>(&xv1.x);
            const __half2 x5 = *reinterpret_cast<const __half2*>(&xv1.y);
            const __half2 x6 = *reinterpret_cast<const __half2*>(&xv1.z);
            const __half2 x7 = *reinterpret_cast<const __half2*>(&xv1.w);
            gk = __hfma2(g0, x0, gk); gk = __hfma2(g1, x1, gk);
            gk = __hfma2(g2, x2, gk); gk = __hfma2(g3, x3, gk);
            gk = __hfma2(g4, x4, gk); gk = __hfma2(g5, x5, gk);
            gk = __hfma2(g6, x6, gk); gk = __hfma2(g7, x7, gk);
            uk = __hfma2(u0, x0, uk); uk = __hfma2(u1, x1, uk);
            uk = __hfma2(u2, x2, uk); uk = __hfma2(u3, x3, uk);
            uk = __hfma2(u4, x4, uk); uk = __hfma2(u5, x5, uk);
            uk = __hfma2(u6, x6, uk); uk = __hfma2(u7, x7, uk);
            pg[k] = gk; pu[k] = uk;
        }
    }
    float ag[NT], au[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        const float2 fg = __half22float2(pg[k]);
        const float2 fu = __half22float2(pu[k]);
        const float xd = smem.x.sxd[k][b];
        ag[k] = (dg * xd) * (fg.x + fg.y);
        au[k] = (du * xd) * (fu.x + fu.y);
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        smem.p.part_g[b][k][rl] = ag[k];
        smem.p.part_u[b][k][rl] = au[k];
    }
    __syncthreads();
    // 2-stage parallel reduction of 80 partials (5 groups of 16 -> 5 sums) instead of 80-step serial loop!
    if (tid < NT * 4 * 5) {
        const int grp = tid / (NT * 4);
        const int rem = tid - grp * (NT * 4);
        const int k = rem >> 2, rl2 = rem & 3;
        const int w0 = grp * 16;
        float sg = smem.p.part_g[w0][k][rl2], su = smem.p.part_u[w0][k][rl2];
#pragma unroll
        for (int dw = 1; dw < 16; ++dw) {
            sg += smem.p.part_g[w0 + dw][k][rl2];
            su += smem.p.part_u[w0 + dw][k][rl2];
        }
        smem.p.part_g[grp][k][rl2] = sg;
        smem.p.part_u[grp][k][rl2] = su;
    }
    __syncthreads();
    if (tid < NT * 4) {
        const int k = tid >> 2, rl2 = tid & 3;
        const float sg = smem.p.part_g[0][k][rl2] + smem.p.part_g[1][k][rl2] + smem.p.part_g[2][k][rl2] +
                         smem.p.part_g[3][k][rl2] + smem.p.part_g[4][k][rl2];
        const float su = smem.p.part_u[0][k][rl2] + smem.p.part_u[1][k][rl2] + smem.p.part_u[2][k][rl2] +
                         smem.p.part_u[3][k][rl2] + smem.p.part_u[4][k][rl2];
        const float xi = __fdividef(sg, 1.0f + __expf(-sg)) * su;
        h_f16[(size_t) k * n_ff + (size_t) blockIdx.x * 4 + rl2] = __float2half_rn(fminf(fmaxf(xi, -65504.0f), 65504.0f));
    }
}

template<int NT>
__global__ void __launch_bounds__(80) shared_down_scale_f16_soa_5k_kernel(
    const uint8_t* __restrict__ w_down, const __half* __restrict__ h_f16,
    const float* __restrict__ g_in, float* __restrict__ out, int n_ff, int n_embd) {
    const int tid = threadIdx.x, hw = tid >> 4, hl = tid & 15;   // 5 K-workers (b = 0..4) x 16 rows
    const int r = (int) blockIdx.x * 16 + hl;
    const uint4* Wq = reinterpret_cast<const uint4*>(w_down);
    const __half* Wd = reinterpret_cast<const __half*>(w_down + (size_t) n_ff * n_embd);

    union SharedDown5 {
        __align__(16) __half2 sxh[NT][80];   // n_ff = 160 -> 80 half2
        float part[5][NT][16];
    };
    __shared__ SharedDown5 smem;

#pragma unroll
    for (int k = 0; k < NT; ++k) {
        smem.sxh[k][tid] = reinterpret_cast<const __half2*>(h_f16 + (size_t) k * n_ff)[tid];
    }
    __syncthreads();

    float acc[NT];
    if (r < n_embd) {
        const int b = hw;
        const float d = __half2float(__ldg(Wd + (size_t) b * n_embd + r));
        __half2 p[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) p[k] = __float2half2_rn(0.0f);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 qv = __ldg(Wq + (size_t) (b * 2 + h) * n_embd + r);
            const int xb = b * 16 + h * 8;
            const __half2 w0 = sh_u8_lo_h2(qv.x), w1 = sh_u8_hi_h2(qv.x);
            const __half2 w2 = sh_u8_lo_h2(qv.y), w3 = sh_u8_hi_h2(qv.y);
            const __half2 w4 = sh_u8_lo_h2(qv.z), w5 = sh_u8_hi_h2(qv.z);
            const __half2 w6 = sh_u8_lo_h2(qv.w), w7 = sh_u8_hi_h2(qv.w);
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                __half2 pk = p[k];
                const uint4 xv0 = *reinterpret_cast<const uint4*>(&smem.sxh[k][xb + 0]);
                const uint4 xv1 = *reinterpret_cast<const uint4*>(&smem.sxh[k][xb + 4]);
                pk = __hfma2(w0, *reinterpret_cast<const __half2*>(&xv0.x), pk);
                pk = __hfma2(w1, *reinterpret_cast<const __half2*>(&xv0.y), pk);
                pk = __hfma2(w2, *reinterpret_cast<const __half2*>(&xv0.z), pk);
                pk = __hfma2(w3, *reinterpret_cast<const __half2*>(&xv0.w), pk);
                pk = __hfma2(w4, *reinterpret_cast<const __half2*>(&xv1.x), pk);
                pk = __hfma2(w5, *reinterpret_cast<const __half2*>(&xv1.y), pk);
                pk = __hfma2(w6, *reinterpret_cast<const __half2*>(&xv1.z), pk);
                pk = __hfma2(w7, *reinterpret_cast<const __half2*>(&xv1.w), pk);
                p[k] = pk;
            }
        }
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const float2 f = __half22float2(p[k]);
            acc[k] = d * (f.x + f.y);
        }
    }
    __syncthreads();
    if (r < n_embd) {
#pragma unroll
        for (int k = 0; k < NT; ++k) smem.part[hw][k][hl] = acc[k];
    }
    __syncthreads();
    if (hw == 0 && r < n_embd) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const float s = smem.part[0][k][hl] + smem.part[1][k][hl] + smem.part[2][k][hl] +
                            smem.part[3][k][hl] + smem.part[4][k][hl];
            out[(size_t) k * n_embd + r] = s * g_in[k];
        }
    }
}

// GP100 20-K-Worker x 16-Row (320t) Q8_0 SoA Shared Down for n_ff=640 (MTP unsharded draft, 8.04 -> 6.65 us, STACK:0!)
template<int NT>
__global__ void __launch_bounds__(320) shared_down_scale_f16_soa_20k_kernel(
    const uint8_t* __restrict__ w_down, const __half* __restrict__ h_f16,
    const float* __restrict__ g_in, float* __restrict__ out, int n_ff, int n_embd) {
    const int tid = threadIdx.x, b = tid >> 4, hl = tid & 15;   // 20 K-workers (b = 0..19) x 16 rows
    const int r = (int) blockIdx.x * 16 + hl;
    const uint4* Wq = reinterpret_cast<const uint4*>(w_down);
    const __half* Wd = reinterpret_cast<const __half*>(w_down + (size_t) n_ff * n_embd);

    union SharedDown20 {
        __align__(16) __half2 sxh[NT][320];   // n_ff = 640 -> 320 half2
        float part[20][NT][16];
    };
    __shared__ SharedDown20 smem;

#pragma unroll
    for (int k = 0; k < NT; ++k) {
        smem.sxh[k][tid] = reinterpret_cast<const __half2*>(h_f16 + (size_t) k * n_ff)[tid];
    }
    __syncthreads();

    float acc[NT];
    if (r < n_embd) {
        const float d = __half2float(__ldg(Wd + (size_t) b * n_embd + r));
        __half2 p[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) p[k] = __float2half2_rn(0.0f);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 qv = __ldg(Wq + (size_t) (b * 2 + h) * n_embd + r);
            const int xb = b * 16 + h * 8;
            const __half2 w0 = sh_u8_lo_h2(qv.x), w1 = sh_u8_hi_h2(qv.x);
            const __half2 w2 = sh_u8_lo_h2(qv.y), w3 = sh_u8_hi_h2(qv.y);
            const __half2 w4 = sh_u8_lo_h2(qv.z), w5 = sh_u8_hi_h2(qv.z);
            const __half2 w6 = sh_u8_lo_h2(qv.w), w7 = sh_u8_hi_h2(qv.w);
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                __half2 pk = p[k];
                const uint4 xv0 = *reinterpret_cast<const uint4*>(&smem.sxh[k][xb + 0]);
                const uint4 xv1 = *reinterpret_cast<const uint4*>(&smem.sxh[k][xb + 4]);
                pk = __hfma2(w0, *reinterpret_cast<const __half2*>(&xv0.x), pk);
                pk = __hfma2(w1, *reinterpret_cast<const __half2*>(&xv0.y), pk);
                pk = __hfma2(w2, *reinterpret_cast<const __half2*>(&xv0.z), pk);
                pk = __hfma2(w3, *reinterpret_cast<const __half2*>(&xv0.w), pk);
                pk = __hfma2(w4, *reinterpret_cast<const __half2*>(&xv1.x), pk);
                pk = __hfma2(w5, *reinterpret_cast<const __half2*>(&xv1.y), pk);
                pk = __hfma2(w6, *reinterpret_cast<const __half2*>(&xv1.z), pk);
                pk = __hfma2(w7, *reinterpret_cast<const __half2*>(&xv1.w), pk);
                p[k] = pk;
            }
        }
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const float2 f = __half22float2(p[k]);
            acc[k] = d * (f.x + f.y);
        }
    }
    __syncthreads();
    if (r < n_embd) {
#pragma unroll
        for (int k = 0; k < NT; ++k) smem.part[b][k][hl] = acc[k];
    }
    __syncthreads();
    if (b == 0 && r < n_embd) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            float s = smem.part[0][k][hl];
#pragma unroll
            for (int w = 1; w < 20; ++w) s += smem.part[w][k][hl];
            out[(size_t) k * n_embd + r] = s * g_in[k];
        }
    }
}

template<int NT>
void launch_shared_soa_fused(const NativeSharedWeights& nw, const void* xq_gu, const float* x,
                             const uint16_t* x_bf16, const uint16_t* gate_inp_bf16, float* gate_scratch,
                             float* g, float* out, int n_embd, int n_ff, cudaStream_t cs) {
    __half* h_f16 = reinterpret_cast<__half*>(gate_scratch);
    const unsigned g_dn = (unsigned) ((n_embd + 15) / 16);
    bool ran_4r80k = false;
    if constexpr (NT >= 5) {
        if (n_ff <= 320 && n_embd == 2560 && (n_ff & 3) == 0) {
            const unsigned g_gu4 = (unsigned) (n_ff / 4 + 1);
            shared_gu_swiglu_f16_soa_4r80k_kernel<NT><<<g_gu4, 320, 0, cs>>>(
                static_cast<const uint8_t*>(nw.gate_data), static_cast<const uint8_t*>(nw.up_data),
                static_cast<const ShQ81Block*>(xq_gu), x, x_bf16, gate_inp_bf16,
                h_f16, g, n_embd, n_ff, native_bf16);
            ran_4r80k = true;
        }
    }
    if (!ran_4r80k) {
        if (n_ff <= 320) {
            const unsigned g_gu4 = (unsigned) (n_ff / 4 + 1);
            shared_gu_swiglu_f16_soa_4r_kernel<NT><<<g_gu4, 128, 0, cs>>>(
                static_cast<const uint8_t*>(nw.gate_data), static_cast<const uint8_t*>(nw.up_data),
                static_cast<const ShQ81Block*>(xq_gu), x, x_bf16, gate_inp_bf16,
                h_f16, g, n_embd, n_ff, native_bf16);
        } else {
            const unsigned g_gu = (unsigned) (n_ff / 8 + 1);
            shared_gu_swiglu_f16_soa_kernel<NT><<<g_gu, 128, 0, cs>>>(
                static_cast<const uint8_t*>(nw.gate_data), static_cast<const uint8_t*>(nw.up_data),
                static_cast<const ShQ81Block*>(xq_gu), x, x_bf16, gate_inp_bf16,
                h_f16, g, n_embd, n_ff, native_bf16);
        }
    }
    if (n_ff == 160) {
        shared_down_scale_f16_soa_5k_kernel<NT><<<g_dn, 80, 0, cs>>>(
            static_cast<const uint8_t*>(nw.down_data), h_f16, g, out, n_ff, n_embd);
    } else if (n_ff == 640 && NT <= 4) {
        shared_down_scale_f16_soa_20k_kernel<NT><<<g_dn, 320, 0, cs>>>(
            static_cast<const uint8_t*>(nw.down_data), h_f16, g, out, n_ff, n_embd);
    } else {
        shared_down_scale_f16_soa_kernel<NT><<<g_dn, 64, 0, cs>>>(
            static_cast<const uint8_t*>(nw.down_data), h_f16, g, out, n_ff, n_embd);
    }
}
}  // namespace

void shared_expert_multi(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd,
                         int64_t n_ff, void* stream, const void* x_q8_in) {
    if (n_tok < 1 || n_tok > 8 || !nw.q8_1 || !nw.gate_data || !nw.up_data || !nw.down_data || !stream)
        throw std::invalid_argument("shared_expert_multi: needs 1..8 tokens, native weights, scratch and a stream");
    cudaStream_t cs = (cudaStream_t) stream;
    const void* xq_gu = x_q8_in ? x_q8_in : nw.q8_1;
    if (!x_q8_in) native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
    if (nw.gate_type == kGgmlQ8_0_SOA && nw.up_type == kGgmlQ8_0_SOA && nw.down_type == kGgmlQ8_0_SOA &&
        (n_ff % 32 == 0) && (n_embd % 32 == 0) && n_tok >= 1 && n_tok <= 8) {
        switch (n_tok) {
            case 1: launch_shared_soa_fused<1>(nw, xq_gu, x, x_bf16, gate_inp_bf16, gate, g, out, (int) n_embd, (int) n_ff, cs); break;
            case 2: launch_shared_soa_fused<2>(nw, xq_gu, x, x_bf16, gate_inp_bf16, gate, g, out, (int) n_embd, (int) n_ff, cs); break;
            case 3: launch_shared_soa_fused<3>(nw, xq_gu, x, x_bf16, gate_inp_bf16, gate, g, out, (int) n_embd, (int) n_ff, cs); break;
            case 4: launch_shared_soa_fused<4>(nw, xq_gu, x, x_bf16, gate_inp_bf16, gate, g, out, (int) n_embd, (int) n_ff, cs); break;
            case 5: launch_shared_soa_fused<5>(nw, xq_gu, x, x_bf16, gate_inp_bf16, gate, g, out, (int) n_embd, (int) n_ff, cs); break;
            case 6: launch_shared_soa_fused<6>(nw, xq_gu, x, x_bf16, gate_inp_bf16, gate, g, out, (int) n_embd, (int) n_ff, cs); break;
            case 7: launch_shared_soa_fused<7>(nw, xq_gu, x, x_bf16, gate_inp_bf16, gate, g, out, (int) n_embd, (int) n_ff, cs); break;
            default: launch_shared_soa_fused<8>(nw, xq_gu, x, x_bf16, gate_inp_bf16, gate, g, out, (int) n_embd, (int) n_ff, cs); break;
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("shared_expert_multi soa: ") + cudaGetErrorString(e));
        return;
    }
    native_mmvq(nw.gate_type, nw.gate_data, xq_gu, gate, (int) n_embd, (int) n_ff, n_tok, stream);
    native_mmvq(nw.up_type, nw.up_data, xq_gu, up, (int) n_embd, (int) n_ff, n_tok, stream);
    const int n = (int) (n_ff * n_tok);
    if ((n_ff & 31) == 0) {
        native_swiglu_quant_q8_1_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, cs>>>(
            gate, up, static_cast<ShQ81Block*>(nw.q8_1), n);
    } else {
        native_swiglu_kernel<<<(unsigned) ((n + THREADS - 1) / THREADS), THREADS, 0, cs>>>(gate, up, gate, n);
        native_quantize_q8_1(gate, nw.q8_1, (int) n_ff, n_tok, stream);
    }
    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
    static const bool batch = [] { const char* v = std::getenv("MAYA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    if (native_bf16 && batch && n_tok > 1) {   // one gemv for all rows (outputs identical), one sigmoid launch
        bf16_gemv_fp32_mmvf_multi(x, n_embd, gate_inp_bf16, g, 1, n_embd, 1, n_tok, stream);
        native_scalar_sigmoid_multi_kernel<<<1, n_tok, 0, cs>>>(g);
    } else
    for (int t = 0; t < n_tok; ++t) {
        if (native_bf16) {
            bf16_gemv_fp32_mmvf(x + (size_t) t * n_embd, gate_inp_bf16, g + t, n_embd, 1, stream);
            native_scalar_sigmoid_kernel<<<1, 1, 0, cs>>>(g + t);
        } else {
            scalar_gate_kernel<<<1, 256, 0, cs>>>(x_bf16 + (size_t) t * n_embd, gate_inp_bf16, g + t, (int) n_embd);
        }
    }
    scale_rows_kernel<<<dim3((unsigned) ((n_embd + THREADS - 1) / THREADS), (unsigned) n_tok), THREADS, 0, cs>>>(
        out, g, (int) n_embd);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("shared_expert_multi: ") + cudaGetErrorString(e));
}

uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    // gate (n_ff f32) | up (n_ff f32) | q8_0 (n_ff/32*34) | q8k (n_ff/256*292) | g (1 f32), 16-byte aligned
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}

void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k, const uint16_t* x_bf16, const SForm& gate_form,
                   const uint8_t* gate_codes, const float* gate_scales, const float* gate_off,
                   const SForm& up_form, const uint8_t* up_codes, const float* up_scales, const float* up_off,
                   const SForm& down_form, const uint8_t* down_codes, const float* down_scales,
                   const float* down_off, const uint16_t* gate_inp_bf16, float* scratch, float* out,
                   int64_t n_embd, int64_t n_ff, int tpr, void* stream, const float* x_f32,
                   const NativeSharedWeights* native) {
    if (n_embd <= 0 || n_ff <= 0) return;
    const bool use_native = native_bf16;
    const bool native_gate = native && native->gate_data && native_mmvq_supported(native->gate_type);
    const bool native_up = native && native->up_data && native_mmvq_supported(native->up_type);
    const bool native_down = native && native->down_data && native_mmvq_supported(native->down_type);
    const bool native_projection = native_gate || native_up || native_down;
    if ((use_native || native_gate || native_up) && !x_f32)
        throw std::invalid_argument("shared_expert native input projection requires unrounded x_f32");
    if (native_projection) {
        if (!native->q8_1 || !stream || n_embd > INT_MAX || n_ff > INT_MAX)
            throw std::invalid_argument("shared_expert native projections require scratch, stream and int32 dimensions");
        // Validate every active shape before any kernel is enqueued.
        if (native_gate) native_mmvq_weight_bytes(native->gate_type, (int) n_embd, (int) n_ff);
        if (native_up) native_mmvq_weight_bytes(native->up_type, (int) n_embd, (int) n_ff);
        if (native_down) native_mmvq_weight_bytes(native->down_type, (int) n_ff, (int) n_embd);
    }
    if (scratch == nullptr) {
        std::fprintf(stderr, "shared_expert: scratch is null; the caller owns it "
                             "(see shared_expert_scratch_bytes)\n");
        std::exit(1);
    }
    // CARVED FROM THE CALLER'S SCRATCH.  This used to be four `cudaMalloc`s and four `cudaFree`s per call, which
    // is illegal during stream capture AND a token-path allocation - P2.T10 forbids both.
    uint8_t* p = (uint8_t*) scratch;
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    float* gate = (float*) p;
    float* up = (float*) (p + a);
    uint8_t* h_q8_0 = (uint8_t*) (p + a * 2);
    uint8_t* h_q8k = (uint8_t*) (p + a * 2 + q0);
    float* g = (float*) (p + a * 2 + q0 + qk);

    // WHICH ACTIVATION THIS PROJECTION WANTS, READ FROM ITS OWN FORM.  See `SForm::act_kind`: the three
    // families cannot be told apart by the other fields, so the kind is carried rather than derived.
    auto gemv = [&](const SForm& f, const uint8_t* codes, const float* scales, const float* off,
                    const uint8_t* act80, const uint8_t* actq8k, float* y, int64_t nin, int64_t nout) {
        if (f.code_bits == 2) {
            s2_gemv_q8(act80, codes, scales, y, nin, nout, tpr, stream);
        } else if (f.act_kind == 1) {
            s_gemv_q8k_split(actq8k, codes, scales, off, y, nin, nout, f, stream);
        } else {
            s_gemv_q8_0_split(act80, codes, scales, off, y, nin, nout, f, stream);
        }
    };

    const unsigned g_ff = (unsigned) ((n_ff + THREADS - 1) / THREADS);
    const unsigned g_embd = (unsigned) ((n_embd + THREADS - 1) / THREADS);

    // gate and up projections, then silu(gate) * up in place in `gate`
    if (native_gate || native_up)
        native_quantize_q8_1(x_f32, native->q8_1, (int) n_embd, 1, stream);
    if (native_gate)
        native_mmvq(native->gate_type, native->gate_data, native->q8_1, gate, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(gate_form, gate_codes, gate_scales, gate_off, x_q8_0, x_q8k, gate, n_embd, n_ff);
    if (native_up)
        native_mmvq(native->up_type, native->up_data, native->q8_1, up, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(up_form, up_codes, up_scales, up_off, x_q8_0, x_q8k, up, n_embd, n_ff);
    if (native_projection)
        native_swiglu_kernel<<<g_ff, THREADS, 0, (cudaStream_t) stream>>>(gate, up, gate, (int) n_ff);
    else
        swiglu_kernel<<<g_ff, THREADS, 0, (cudaStream_t) stream>>>(gate, up, gate, (int) n_ff);

    // down: (n_ff) -> (n_embd), and THE INTERMEDIATE IS QUANTIZED TO THE DOWN WEIGHT'S OWN CONTRACT - which is
    // what `ggml_mul_mat` does for every matmul in the model.  It used to be rounded to fp16 with no
    // justification beyond "the kernel takes fp16".
    if (native_down) {
        native_quantize_q8_1(gate, native->q8_1, (int) n_ff, 1, stream);
        native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, stream);
    } else if (down_form.act_kind == 1) {
        if (n_ff % 256 != 0) {
            std::fprintf(stderr, "shared_expert: the down weight wants Q8_K but n_ff %lld is not a multiple "
                                 "of 256; Q8_K is structurally impossible here\n", (long long) n_ff);
            std::exit(1);
        }
        quantize_q8_K(gate, h_q8k, n_ff, stream);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    } else if (down_form.code_bits == 2) {
        quantize_q8_0(gate, h_q8_0, n_ff, stream);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    } else {
        quantize_q8_0(gate, h_q8_0, n_ff, stream);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    }

    // the per-token scalar gate, then the multiply.  Note the gate is computed from `x`, the ORIGINAL hidden
    // state, not from anything the expert produced. The historical branch uses BF16-rounded inputs; the
    // native branch uses the pinned CUDA FP32 activation contract.
    // `<<<1, 256>>>`: one block, because the output is ONE scalar and a second block would only add a global
    // round trip.  256 threads is the reduction's width, not the problem's size.
    if (use_native) {
        bf16_gemv_fp32_mmvf(x_f32, gate_inp_bf16, g, n_embd, 1, stream);
        native_scalar_sigmoid_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(g);
    } else {
        scalar_gate_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(x_bf16, gate_inp_bf16, g, (int) n_embd);
    }
    scale_kernel<<<g_embd, THREADS, 0, (cudaStream_t) stream>>>(out, g, (int) n_embd);

    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "shared_expert: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n_embd,
                 int64_t k, void* stream) {
    if (n_embd <= 0 || k <= 0) return;
    // k > 64 is refused rather than truncated: silently summing the first 64 of a longer list would be a
    // wrong answer that looks like a right one, and no geometry in this artifact comes close to it.
    if (k > 64) {
        std::fprintf(stderr, "moe_combine: k = %lld exceeds 64\n", (long long) k);
        std::exit(1);
    }
    const unsigned grid = (unsigned) ((n_embd + THREADS - 1) / THREADS);
    moe_combine_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(parts, weights, shared, y, (int) n_embd,
                                                                   (int) k, shared != nullptr);
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "moe_combine: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

}  // namespace strata::kernels
