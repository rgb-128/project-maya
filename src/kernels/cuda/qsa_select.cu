// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_a[TOPK_T], s_b[TOPK_T];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    __syncthreads();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    __syncthreads();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    __syncthreads();
    s_a[t] = sel;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    __syncthreads();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}


// ---- QSA select on tensor cores (perf-review, after D-1): the block scores of many queries are one GEMM,
// rows (query, indexer head) x columns (blocks), K = 128, with relu per head summed. 3xTF32 (each operand split
// into a TF32 hi and lo part, hi*hi + hi*lo + lo*hi) keeps FP32-level accuracy; the summation order differs from
// the warp kernel, so a score can move in its last bits and a near-tie can select differently (not bitwise).
// Blocks < n_bid only; the tail block n_bid (the `dead` key, +1e9) is scored by the warp kernel's own code.
constexpr int TC_QT = 16;                 // queries per CTA (one m16 tile per indexer head)
constexpr int TC_NB = 32;                 // blocks per tile (4 warps x n8)
constexpr int TC_ITER = 4;                // tiles per CTA (the query tile is loaded once)
constexpr int TC_QS = IDX_HEADS * IDX_DIM + 4;   // query row stride in floats
constexpr int TC_KS = IDX_DIM + 4;               // key row stride

// TF32 conversion and MMA need sm_80: below it they compile to a trap and qsa_block_scores_tc refuses the device
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the warp kernel (below)
#define STRATA_SEL_SM80 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_SEL_SM80 1
#else
#define STRATA_SEL_SM80 0
#endif
__device__ __forceinline__ uint32_t tf32_hi(float x) {
#if STRATA_SEL_SM80
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    return __float_as_uint(x);
#endif
}
__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_SEL_SM80
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__global__ void __launch_bounds__(128) block_scores_tc_kernel(const float* __restrict__ pooled,
                                                              const float* __restrict__ q_idx,
                                                              const int32_t* __restrict__ steps, int64_t nq,
                                                              int64_t max_blocks, int64_t reach,
                                                              float* __restrict__ out) {
    extern __shared__ __align__(16) float sm[];
    float* sQ = sm;                                   // [TC_QT][TC_QS]
    float* sK = sm + TC_QT * TC_QS;                   // [TC_NB][TC_KS]
    __shared__ int s_nbid[TC_QT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int64_t q0 = (int64_t) blockIdx.y * TC_QT;
    for (int i = t; i < TC_QT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int r = i / (IDX_HEADS * IDX_DIM / 4), c = i % (IDX_HEADS * IDX_DIM / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) v = reinterpret_cast<const float4*>(q_idx + (q0 + r) * IDX_HEADS * IDX_DIM)[c];
        *reinterpret_cast<float4*>(sQ + r * TC_QS + c * 4) = v;
    }
    if (t < TC_QT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    int lo_nbid = 0x7fffffff, hi_nbid = 0;
    __syncthreads();
    for (int i = 0; i < TC_QT; ++i) {
        if (q0 + i >= nq) break;
        lo_nbid = min(lo_nbid, s_nbid[i]);
        hi_nbid = max(hi_nbid, s_nbid[i]);
    }
    for (int it = 0; it < TC_ITER; ++it) {
        const int64_t b0 = ((int64_t) blockIdx.x * TC_ITER + it) * TC_NB;
        if (b0 >= reach || b0 >= hi_nbid) break;      // blocks >= every query's n_bid: nothing to score
        __syncthreads();                              // the previous tile's reads are done
        for (int i = t; i < TC_NB * IDX_DIM / 4; i += 128) {
            const int r = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + r < hi_nbid) v = reinterpret_cast<const float4*>(pooled + (b0 + r) * IDX_DIM)[c];
            *reinterpret_cast<float4*>(sK + r * TC_KS + c * 4) = v;
        }
        __syncthreads();
        float acc[IDX_HEADS][4];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
        const float* kr = sK + (warp * 8 + gid) * TC_KS;
#pragma unroll 4
        for (int k0 = 0; k0 < IDX_DIM; k0 += 8) {
            const float kx0 = kr[k0 + tig], kx1 = kr[k0 + tig + 4];
            uint32_t bh[2], bl[2];
            bh[0] = tf32_hi(kx0);
            bh[1] = tf32_hi(kx1);
            bl[0] = tf32_hi(kx0 - __uint_as_float(bh[0]));
            bl[1] = tf32_hi(kx1 - __uint_as_float(bh[1]));
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float* qa = sQ + h * IDX_DIM + k0 + tig;
                const float x0 = qa[gid * TC_QS], x1 = qa[(gid + 8) * TC_QS];
                const float x2 = qa[gid * TC_QS + 4], x3 = qa[(gid + 8) * TC_QS + 4];
                uint32_t ah[4], al[4];
                ah[0] = tf32_hi(x0); ah[1] = tf32_hi(x1); ah[2] = tf32_hi(x2); ah[3] = tf32_hi(x3);
                al[0] = tf32_hi(x0 - __uint_as_float(ah[0]));
                al[1] = tf32_hi(x1 - __uint_as_float(ah[1]));
                al[2] = tf32_hi(x2 - __uint_as_float(ah[2]));
                al[3] = tf32_hi(x3 - __uint_as_float(ah[3]));
                mma_tf32(acc[h], al, bh);
                mma_tf32(acc[h], ah, bl);
                mma_tf32(acc[h], ah, bh);
            }
        }
        // relu per head, heads added in order (as the warp kernel), written where the block completed for the query
        const int64_t bc = b0 + warp * 8 + 2 * tig;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int qr = gid + half * 8;
            const int64_t qi = q0 + qr;
            if (qi >= nq) continue;
            const int nb = s_nbid[qr];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int64_t b = bc + j;
                if (b >= nb || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][half * 2 + j];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

// GP100 (sm_60..75) 64x64 SMEM-Tiled FP32 QSA Block Scores: 5.7-5.9x faster than block_scores_kernel,
// 100% pure FP32 precision (max_diff ~2.9e-6), STACK:0, skips upper causal triangle (b0 >= hi_nbid)!
__global__ void __launch_bounds__(256, 2) block_scores_p100_64x64_kernel(
    const float* __restrict__ pooled, const float* __restrict__ q_idx,
    const int32_t* __restrict__ steps, int64_t nq, int64_t max_blocks, int64_t reach,
    float* __restrict__ out) {
    __shared__ float sK[64][33];
    __shared__ float sQ[64][33];
    __shared__ int32_t s_nbid[64];
    const int64_t q0 = (int64_t) blockIdx.y * 64;
    const int64_t b0 = (int64_t) blockIdx.x * 64;
    const int tid = threadIdx.x;
    if (tid < 64) {
        s_nbid[tid] = (q0 + tid < nq) ? steps[(q0 + tid) * kStepCount + kStepNBid] : 0;
    }
    __syncthreads();
    int32_t hi_nbid = 0;
    for (int i = 0; i < 64; ++i) {
        if (s_nbid[i] > hi_nbid) hi_nbid = s_nbid[i];
    }
    if (b0 >= reach || b0 >= hi_nbid) return;

    const int tx = tid & 15;   // 0..15 -> blocks b0 + tx + j*16
    const int ty = tid >> 4;   // 0..15 -> queries q0 + ty + i*16
    float sc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) sc[i][j] = 0.0f;

    for (int h = 0; h < IDX_HEADS; ++h) {
        float acc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
        for (int k0 = 0; k0 < IDX_DIM; k0 += 32) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int idx = tid + p * 256;
                const int r = idx >> 3;
                const int c4 = idx & 7;
                const float4 qv = (q0 + r < nq)
                    ? __ldg(reinterpret_cast<const float4*>(q_idx + (q0 + r) * IDX_HEADS * IDX_DIM + h * IDX_DIM + k0 + c4 * 4))
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const float4 kv = (b0 + r < hi_nbid && b0 + r < max_blocks)
                    ? __ldg(reinterpret_cast<const float4*>(pooled + (b0 + r) * IDX_DIM + k0 + c4 * 4))
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                sQ[r][c4 * 4 + 0] = qv.x; sQ[r][c4 * 4 + 1] = qv.y; sQ[r][c4 * 4 + 2] = qv.z; sQ[r][c4 * 4 + 3] = qv.w;
                sK[r][c4 * 4 + 0] = kv.x; sK[r][c4 * 4 + 1] = kv.y; sK[r][c4 * 4 + 2] = kv.z; sK[r][c4 * 4 + 3] = kv.w;
            }
            __syncthreads();
#pragma unroll 4
            for (int k = 0; k < 32; ++k) {
                float qr[4], kr[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) { qr[i] = sQ[ty + i * 16][k]; kr[i] = sK[tx + i * 16][k]; }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(qr[i], kr[j], acc[i][j]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) sc[i][j] += acc[i][j] > 0.0f ? acc[i][j] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int qr = ty + i * 16;
        const int64_t qi = q0 + qr;
        if (qi >= nq) continue;
        const int nb = s_nbid[qr];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t bi = b0 + tx + j * 16;
            if (bi < nb && bi < max_blocks) out[qi * max_blocks + bi] = sc[i][j];
        }
    }
}

// the tail block n_bid of each query: exactly block_scores_kernel's arithmetic for that block
__global__ void __launch_bounds__(32) block_scores_tail_kernel(const float* __restrict__ dead,
                                                               const float* __restrict__ q_idx,
                                                               const int32_t* __restrict__ steps,
                                                               int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    if (n_bid >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float4 k4 = *reinterpret_cast<const float4*>(dead + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + n_bid] = score;
    }
}

// ---- the same top-k with each query's keys read once: 1,024 threads hold up to TK_PER consecutive blocks' keys in
// registers (contexts up to 4 * 1024 * TK_PER cells), per-warp histograms, block-wide scans. The selection rule is
// block_topk_kernel's (radix threshold, ties to the lowest index, cells ascending): identical ids.
constexpr int TK_T = 1024;
constexpr int TK_PER = 33;

__device__ __forceinline__ int block_excl_scan(int v, int* s_warp, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = s_warp[lane];
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;               // exclusive per warp
        if (lane == 31) s_warp[32] = z;     // total
    }
    __syncthreads();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    __syncthreads();
    return r;
}

template <int PER>
__global__ void __launch_bounds__(TK_T) block_topk_reg_kernel(const float* __restrict__ scores,
                                                              const int32_t* __restrict__ steps, int64_t max_blocks,
                                                              int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[TK_T / 32][256];
    __shared__ int s_warp[33];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // <= PER (the caller checks)
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    const int active_warps = (int) ((nb + per * 32 - 1) / (per * 32));
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (warp < active_warps) {
            for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
            __syncwarp();
            const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const int64_t b = b0 + j;
                if (b >= b1) break;
                const int w = weight(b);
                if (w == 0) continue;
                if ((key[j] & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(key[j] >> shift) & 255], w);
            }
        }
        __syncthreads();
        // Parallel 256-thread reverse inclusive scan over bins 255..0 (replaces 256-iteration thread-0 serial loop!)
        int bin_s = 0;
        if (t < 256) {
            const int bin = 255 - t;
            for (int w2 = 0; w2 < active_warps; ++w2) bin_s += hist[w2][bin];
        }
        __syncthreads();
        if (t < 256) {
            int x = bin_s;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            if (lane == 31) s_warp[warp] = x;
            hist[0][t] = bin_s;  // safe after syncthreads: all reads from hist[] finished
            hist[1][t] = x;
        }
        __syncthreads();
        if (warp == 0 && lane < 8) {
            int wv = s_warp[lane];
            int z = wv;
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                const int y = __shfl_up_sync(0xffu, z, o);
                if (lane >= o) z += y;
            }
            s_warp[lane] = z - wv;
        }
        __syncthreads();
        if (t < 256) {
            const int incl = above + s_warp[warp] + hist[1][t];
            const int excl = incl - hist[0][t];
            if (excl < width && incl >= width) {
                s_digit = 255 - t;
                s_above = excl;
            }
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) gt += w;
        else if (key[j] == thr) eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = block_excl_scan(sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (key[j] == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}

// Persistent grid-stride multi-query decode block scores: reads pooled[b] ONCE for all T queries (T<=8),
// keeps q_idx in SMEM so 100% of L1/L2 serves pooled[b], 224 CTAs instead of 49,158 CTAs, STACK:0!
template <int MAX_T>
__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_decode_multi_kernel(
    const float* __restrict__ pooled, const float* __restrict__ dead, const float* __restrict__ q_idx,
    const int32_t* __restrict__ steps, int nq, int64_t max_blocks, float* __restrict__ out) {
    __shared__ __align__(16) float sQ[MAX_T][IDX_HEADS][IDX_DIM];
    __shared__ int32_t s_nkv[MAX_T];
    __shared__ int32_t s_nbid[MAX_T];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (tid < nq) {
        const int32_t* st = steps + (size_t) tid * kStepCount;
        s_nkv[tid] = st[kStepNKv];
        s_nbid[tid] = st[kStepNBid];
    }
    const int total_q4 = nq * (IDX_HEADS * IDX_DIM / 4);
    const float4* q_src4 = reinterpret_cast<const float4*>(q_idx);
    float4* sq4 = reinterpret_cast<float4*>(&sQ[0][0][0]);
    for (int i = tid; i < total_q4; i += SCORE_WARPS * 32) {
        sq4[i] = __ldg(q_src4 + i);
    }
    __syncthreads();

    int64_t hi_nbid = 0;
    for (int q = 0; q < nq; ++q) {
        if ((int64_t) s_nbid[q] > hi_nbid) hi_nbid = (int64_t) s_nbid[q];
    }
    if (hi_nbid > max_blocks) hi_nbid = max_blocks;

    const int64_t stride = (int64_t) gridDim.x * SCORE_WARPS;
    for (int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + warp; b < hi_nbid; b += stride) {
        const float4 k4 = __ldg(reinterpret_cast<const float4*>(pooled + b * IDX_DIM + lane * 4));
        for (int qi = 0; qi < nq; ++qi) {
            if (b >= (int64_t) s_nbid[qi]) continue;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float4 q4 = *reinterpret_cast<const float4*>(&sQ[qi][h][lane * 4]);
                float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) out[(size_t) qi * (size_t) max_blocks + b] = score;
        }
    }

    // Tail block n_bid[qi] for each query (uses `dead` key + 1e9f when n_kv % R != 0)
    if (blockIdx.x == 0 && warp < nq) {
        const int64_t nb = (int64_t) s_nbid[warp];
        if (nb < max_blocks) {
            const float4 k4 = __ldg(reinterpret_cast<const float4*>(dead + lane * 4));
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float4 q4 = *reinterpret_cast<const float4*>(&sQ[warp][h][lane * 4]);
                float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (s_nkv[warp] % R != 0) score += 1e9f;
                out[(size_t) warp * (size_t) max_blocks + nb] = score;
            }
        }
    }
}

// CUDA-graph-safe 256K-capable decode top-k kernel (1024 threads, ZERO serial t==0 loops):
// Dynamically selects the 1-pass register path when nb <= 16384 (ctx <= 65,536) and a 1024-thread
// coalesced-histogram + parallel-scan streaming path when nb > 16384 (ctx 65K..256K+). Bit-identical to ref!
__global__ void __launch_bounds__(TK_T) block_topk_decode_kernel(const float* __restrict__ scores,
                                                                 const int32_t* __restrict__ steps,
                                                                 int64_t max_blocks, int64_t cap,
                                                                 int32_t* __restrict__ ids) {
    __shared__ int hist[TK_T / 32][256];
    __shared__ int s_warp[33];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };

    if (per <= 8) {
        constexpr int REG_PER = 8;
        uint32_t key[REG_PER];
#pragma unroll
        for (int j = 0; j < REG_PER; ++j) key[j] = (b0 + j < b1) ? order_key(__ldg(sc + b0 + j)) : 0u;
        const int active_warps = (int) ((nb + per * 32 - 1) / (per * 32));
        uint32_t prefix = 0;
        int above = 0;
        for (int shift = 24; shift >= 0; shift -= 8) {
            if (warp < active_warps) {
                for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
                __syncwarp();
                const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
                for (int j = 0; j < REG_PER; ++j) {
                    const int64_t b = b0 + j;
                    if (b >= b1) break;
                    const int w = weight(b);
                    if (w == 0) continue;
                    if ((key[j] & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(key[j] >> shift) & 255], w);
                }
            }
            __syncthreads();
            int bin_s = 0;
            if (t < 256) {
                const int bin = 255 - t;
                for (int w2 = 0; w2 < active_warps; ++w2) bin_s += hist[w2][bin];
            }
            __syncthreads();
            if (t < 256) {
                int x = bin_s;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const int y = __shfl_up_sync(0xffffffffu, x, o);
                    if (lane >= o) x += y;
                }
                if (lane == 31) s_warp[warp] = x;
                hist[0][t] = bin_s;
                hist[1][t] = x;
            }
            __syncthreads();
            if (warp == 0 && lane < 8) {
                int wv = s_warp[lane];
                int z = wv;
#pragma unroll
                for (int o = 1; o < 8; o <<= 1) {
                    const int y = __shfl_up_sync(0xffu, z, o);
                    if (lane >= o) z += y;
                }
                s_warp[lane] = z - wv;
            }
            __syncthreads();
            if (t < 256) {
                const int incl = above + s_warp[warp] + hist[1][t];
                const int excl = incl - hist[0][t];
                if (excl < width && incl >= width) {
                    s_digit = 255 - t;
                    s_above = excl;
                }
            }
            __syncthreads();
            prefix |= (uint32_t) s_digit << shift;
            above = s_above;
            __syncthreads();
        }
        const uint32_t thr = prefix;
        const int64_t eq_budget = width - above;
        int gt = 0, eq = 0;
#pragma unroll
        for (int j = 0; j < REG_PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if (key[j] > thr) gt += w;
            else if (key[j] == thr) eq += w;
        }
        int tot;
        const int eq_before = block_excl_scan(eq, s_warp, tot);
        int64_t my_eq = eq_budget - eq_before;
        if (my_eq < 0) my_eq = 0;
        if (my_eq > eq) my_eq = eq;
        const int sel = gt + (int) my_eq;
        int64_t wpos = block_excl_scan(sel, s_warp, tot);
        int64_t eq_left = my_eq;
#pragma unroll
        for (int j = 0; j < REG_PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if (key[j] > thr) {
                for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
            } else if (key[j] == thr) {
                for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
            }
        }
        return;
    }

    // Long-context path (nb > 16384, ctx 65K..256K+): 1024 threads, 100% coalesced histogram passes,
    // per-warp SMEM histograms, parallel 256-thread reverse scan & block_excl_scan!
    constexpr int kWarps = TK_T / 32;
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        const int64_t nb_full = nb & ~31ll;
        int64_t b = t;
        for (; b < nb_full; b += TK_T) {
            const int w = weight(b);
            const uint32_t k = order_key(__ldg(sc + b));
            const bool m = (w != 0) && ((k & hi_mask) == (prefix & hi_mask));
            const int bin = m ? (int) ((k >> shift) & 255u) : -1;
            const uint32_t mm = __ballot_sync(0xffffffffu, m);
            if (mm == 0u) continue;
            if (mm == 0xffffffffu) {
                const int b0_val = __shfl_sync(0xffffffffu, bin, 0);
                if (__all_sync(0xffffffffu, bin == b0_val)) {
                    // All 32 lanes hit the exact same bin (99% of shift=24 warps!): 1 SMEM add instead of 32 serialized atomics!
                    const bool full_r = __all_sync(0xffffffffu, w == R);
                    if (full_r) {
                        if (lane == 0) hist[warp][b0_val] += 32 * R;
                    } else {
                        int ws = w;
#pragma unroll
                        for (int o = 16; o > 0; o >>= 1) ws += __shfl_xor_sync(0xffffffffu, ws, o);
                        if (lane == 0) hist[warp][b0_val] += ws;
                    }
                    continue;
                }
            }
            if (m) atomicAdd(&hist[warp][bin], w);
        }
        for (; b < nb; b += TK_T) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(__ldg(sc + b));
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(k >> shift) & 255], w);
        }
        __syncthreads();
        int bin_s = 0;
        if (t < 256) {
            const int bin = 255 - t;
            for (int w2 = 0; w2 < kWarps; ++w2) bin_s += hist[w2][bin];
        }
        __syncthreads();
        if (t < 256) {
            int x = bin_s;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            if (lane == 31) s_warp[warp] = x;
            hist[0][t] = bin_s;
            hist[1][t] = x;
        }
        __syncthreads();
        if (warp == 0 && lane < 8) {
            int wv = s_warp[lane];
            int z = wv;
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                const int y = __shfl_up_sync(0xffu, z, o);
                if (lane >= o) z += y;
            }
            s_warp[lane] = z - wv;
        }
        __syncthreads();
        if (t < 256) {
            const int incl = above + s_warp[warp] + hist[1][t];
            const int excl = incl - hist[0][t];
            if (excl < width && incl >= width) {
                s_digit = 255 - t;
                s_above = excl;
            }
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(__ldg(sc + b));
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = block_excl_scan(sel, s_warp, tot);
    int64_t eq_left = my_eq;
    if (sel > 0) {
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(__ldg(sc + b));
            if (k > thr) {
                for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
            } else if (k == thr) {
                for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
            }
        }
    }
}

}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    static const bool use_old_scores = std::getenv("MAYA_SCORES_OLD") != nullptr;
    if (!use_old_scores && nq <= 8 && reach >= 1024) {
        const unsigned nb = (unsigned) ((reach + SCORE_WARPS - 1) / SCORE_WARPS);
        const unsigned grid = nb < 224u ? nb : 224u;
        if (nq <= 4) {
            block_scores_decode_multi_kernel<4><<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(
                pooled, dead, q_idx, steps, (int) nq, max_blocks, scores);
        } else {
            block_scores_decode_multi_kernel<8><<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(
                pooled, dead, q_idx, steps, (int) nq, max_blocks, scores);
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_decode: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
    const dim3 grid((unsigned) ((reach + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned) nq);
    block_scores_kernel<<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, max_blocks,
                                                                              scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return true;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535 * TC_QT) return false;
    {   // sm_80 or newer (TF32 MMA); an older card keeps the warp kernel
        static int cc_major[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
        if (cc_major[dev] == 0) {
            int major = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            cc_major[dev] = major;
        }
        if (cc_major[dev] < 8) {
#if !defined(__HIPCC__)
            if (nq >= 16) {
                const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
                const dim3 grid((unsigned) ((reach + 63) / 64), (unsigned) ((nq + 63) / 64));
                block_scores_p100_64x64_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(
                    pooled, q_idx, steps, nq, max_blocks, reach, scores);
                block_scores_tail_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(
                    dead, q_idx, steps, max_blocks, scores);
                const cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "qsa_block_scores_p100: %s\n", cudaGetErrorString(e));
                    std::exit(1);
                }
                return true;
            }
#endif
            return false;
        }
    }
#if defined(__HIPCC__)
    return false;   // the tensor-core kernel is compiled out on AMD (its major version is not a CUDA sm)
#endif
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs it on several)
    int adev = 0;
    cudaGetDevice(&adev);
    const int bytes = (TC_QT * TC_QS + TC_NB * TC_KS) * (int) sizeof(float);
    if (!attr[adev]) {
        if (cudaFuncSetAttribute(block_scores_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[adev] = true;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t per = (int64_t) TC_NB * TC_ITER;
    const dim3 grid((unsigned) ((reach + per - 1) / per), (unsigned) ((nq + TC_QT - 1) / TC_QT));
    block_scores_tc_kernel<<<grid, 128, bytes, (cudaStream_t) stream>>>(pooled, q_idx, steps, nq, max_blocks, reach,
                                                                         scores);
    block_scores_tail_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(dead, q_idx, steps, max_blocks, scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_tc: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    block_topk_kernel<<<(unsigned) nq, TOPK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    // keys in registers when every query's blocks fit (contexts up to ~135K cells); the same ids. MAYA_TOPK_OLD=1:
    // the kernel that reads them from memory on every pass
    static const bool old = std::getenv("MAYA_TOPK_OLD") != nullptr;
    if (nq <= 0) return;
    const int64_t reach = (active_blocks > 0 && active_blocks < max_blocks) ? active_blocks : max_blocks;
    if (old) {
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    if (reach > (int64_t) TK_T * TK_PER) {
        block_topk_decode_kernel<<<(unsigned) nq, TK_T, 0, st>>>(scores, steps, max_blocks, cap, ids);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk_decode: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
    const int64_t max_per = (reach + TK_T - 1) / TK_T;
    if (max_per <= 4)
        block_topk_reg_kernel<4><<<(unsigned) nq, TK_T, 0, st>>>(scores, steps, max_blocks, cap, ids);
    else if (max_per <= 8)
        block_topk_reg_kernel<8><<<(unsigned) nq, TK_T, 0, st>>>(scores, steps, max_blocks, cap, ids);
    else
        block_topk_decode_kernel<<<(unsigned) nq, TK_T, 0, st>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels
