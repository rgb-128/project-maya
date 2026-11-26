// src/kernels/cuda/qsa_decode_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int CHUNK = 64;        // cells per block
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// 8 consecutive values of one cell's key or value row for KV head `kvh`, dimensions [d0, d0+8).
// Per-format bodies; `load8` below is the KV_MODE dispatcher. value=false is the K side, true the V side.
__device__ __forceinline__ void load8_f16(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(base));
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(h2[j]);
        out[2 * j] = f.x;
        out[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ void load8_q8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
    const uint16_t sbits = __ldg((value ? p.v_scale : p.k_scale) + row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP);
    const float sc = __half2float(__ushort_as_half(sbits));
    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(codes));
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}
__device__ __forceinline__ void load8_q4(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
    const int b = d0 / QK4_0;
    const int rem = d0 % QK4_0;
    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
    const float d = __half2float(__ushort_as_half(blk->d));
    const int j = (rem == 0 || rem == 16) ? 0 : 8;
    const uint8_t* bytes = blk->qs + j;
    if (rem < 16) {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] & 0x0F) - 8) * d;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] >> 4) - 8) * d;
    }
}
template <int KV_MODE>
__device__ __forceinline__ void load8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    if constexpr (KV_MODE == 0) load8_f16(p, value, row, d0, out);
    else if constexpr (KV_MODE == 1) load8_q8(p, value, row, d0, out);
    else if constexpr (KV_MODE == 3) {
        // hybrid K8V4: both sides are defined - K unrotated INT8, V rotated Q4_0 - so a value=true call
        // reads the Q4_0 pool instead of dereferencing the null v_q (no call site does today; PR review)
        if (value) load8_q4(p, value, row, d0, out);
        else load8_q8(p, value, row, d0, out);
    } else load8_q4(p, value, row, d0, out);
}

template <int KV_MODE, int G>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB: this KV head's query heads
    __shared__ float sp[G][CHUNK];                // scores, then probabilities
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = __ldg(q + (size_t) (kvh * G) * HD + i) * scale;
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
            const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
            float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                      k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            s = warp_sum(s);
            if (lane == 0) sp[h][c] = s;
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here) ? __expf(b - m) : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        float v;
        if constexpr (KV_MODE == 0) {
            v = __half2float(__ushort_as_half(__ldg(p.v_pool + srow[c] * HD + t)));
        } else if constexpr (KV_MODE == 1) {
            const float sc = __half2float(__ushort_as_half(__ldg(p.v_scale + srow[c] * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP)));
            v = (float) __ldg(p.v_q + srow[c] * HD + t) * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
            const float d = __half2float(__ushort_as_half(blk->d));
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

__device__ __forceinline__ __half2 qsa_u32_h2(uint32_t x) { return *reinterpret_cast<const __half2*>(&x); }

// GP100 HFMA2 INT8 KV decode attention chunk kernel (KV_MODE==1):
// Stages Q as half2 (6 KB SMEM), decodes INT8 K/V via __byte_perm+0x64006400 into dual-FP16 HFMA2,
// pre-converts softmax weights to __half, splits PV 2-way across CHUNK=64, and skips dead chunks.
template <int G>
__global__ void __launch_bounds__(THREADS) attn_chunk_q8_p100_kernel(
    const float* __restrict__ q, QsaAttnPools p, const int32_t* __restrict__ ids,
    const int32_t* __restrict__ step, int n_kv_heads, int page_size, float scale,
    float* __restrict__ part_acc, float* __restrict__ part_m, float* __restrict__ part_l,
    int n_chunks, int cap = 0, long long scratch_stride = 0) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;

    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    if (n_here <= 0) return;

    __shared__ __align__(16) __half sph[G][CHUNK];  // 1536 B
    __shared__ long long srow[CHUNK];               //  512 B
    union Work {
        struct {
            __align__(16) __half2 sqh[G][HD / 2];   // 6144 B
            __align__(16) float sp[G][CHUNK];       // 3072 B
        } qk;
        __align__(16) __half2 vacc[G][HD / 2];      // 6144 B -> total 11,264 B (4 blocks/SM on GP100!)
    };
    __shared__ Work wk;

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int slot = kvh * n_chunks + chunk;

    for (int i = t; i < G * (HD / 2); i += THREADS) {
        const float2 qv = __ldg(reinterpret_cast<const float2*>(q + (size_t) (kvh * G) * HD) + i);
        wk.qk.sqh[i / (HD / 2)][i % (HD / 2)] = __floats2half2_rn(qv.x * scale, qv.y * scale);
    }
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();

    const __half2 c1152 = __float2half2_rn(1152.0f);
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here) {
            if (lane < G) wk.qk.sp[lane][c] = -FLT_MAX;
            continue;
        }
        const long long row = srow[c];
        const uint2 raw = __ldg(reinterpret_cast<const uint2*>(p.k_q + row * HD + lane * 8));
        const uint16_t sbits = __ldg(p.k_scale + row * (HD / KV_Q8_GROUP) + (lane * 8) / KV_Q8_GROUP);
        const __half2 ks = __half2half2(__ushort_as_half(sbits));
        const uint32_t ux = raw.x ^ 0x80808080u;
        const uint32_t uy = raw.y ^ 0x80808080u;
        const __half2 k0 = __hmul2(__hsub2(qsa_u32_h2(__byte_perm(ux, 0x64006400u, 0x5150)), c1152), ks);
        const __half2 k1 = __hmul2(__hsub2(qsa_u32_h2(__byte_perm(ux, 0x64006400u, 0x7372)), c1152), ks);
        const __half2 k2 = __hmul2(__hsub2(qsa_u32_h2(__byte_perm(uy, 0x64006400u, 0x5150)), c1152), ks);
        const __half2 k3 = __hmul2(__hsub2(qsa_u32_h2(__byte_perm(uy, 0x64006400u, 0x7372)), c1152), ks);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const uint4 qv = *reinterpret_cast<const uint4*>(&wk.qk.sqh[h][lane * 4]);
            __half2 dot = __hmul2(k0, qsa_u32_h2(qv.x));
            dot = __hfma2(k1, qsa_u32_h2(qv.y), dot);
            dot = __hfma2(k2, qsa_u32_h2(qv.z), dot);
            dot = __hfma2(k3, qsa_u32_h2(qv.w), dot);
            const float2 f = __half22float2(dot);
            const float s = warp_sum(f.x + f.y);
            if (lane == 0) wk.qk.sp[h][c] = s;
        }
    }
    __syncthreads();

    for (int h = warp; h < G; h += WARPS) {
        const float a = wk.qk.sp[h][lane], b = wk.qk.sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here) ? __expf(b - m) : 0.0f;
        sph[h][lane] = __float2half_rn(ea);
        sph[h][lane + 32] = __float2half_rn(eb);
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();

    const int grp = t >> 7;
    const int d2 = t & 127;
    const int c_beg = grp * 32;
    const int c_end = min(n_here, c_beg + 32);
    const int sc_idx = (d2 * 2) / KV_Q8_GROUP;
    __half2 acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = __float2half2_rn(0.0f);
    for (int c = c_beg; c < c_end; ++c) {
        const long long row = srow[c];
        const uint16_t raw2 = __ldg(reinterpret_cast<const uint16_t*>(p.v_q + row * HD + d2 * 2));
        const uint16_t sbits = __ldg(p.v_scale + row * (HD / KV_Q8_GROUP) + sc_idx);
        const uint32_t u = (uint32_t) raw2 ^ 0x8080u;
        const __half2 vh = __hmul2(__hsub2(qsa_u32_h2(__byte_perm(u, 0x64006400u, 0x5150)), c1152),
                                   __half2half2(__ushort_as_half(sbits)));
#pragma unroll
        for (int h = 0; h < G; ++h) {
            acc[h] = __hfma2(__half2half2(sph[h][c]), vh, acc[h]);
        }
    }
    if (grp == 1) {
#pragma unroll
        for (int h = 0; h < G; ++h) wk.vacc[h][d2] = acc[h];
    }
    __syncthreads();
    if (grp == 0) {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float2 f0 = __half22float2(acc[h]);
            const float2 f1 = __half22float2(wk.vacc[h][d2]);
            *reinterpret_cast<float2*>(part_acc + ((size_t) slot * G + h) * HD + d2 * 2) =
                make_float2(f0.x + f1.x, f0.y + f1.y);
        }
    }
}

bool p100_attn_dec_enabled() {
    static const bool on = [] {
        if (const char* v = std::getenv("MAYA_P100_ATTN_DEC")) return std::atoi(v) != 0;
        int dev = 0;
        cudaDeviceProp prop{};
        if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
            return prop.major == 6 && prop.minor == 0;
        return false;
    }();
    return on;
}

template <int G>
__global__ void __launch_bounds__(HD) attn_merge_kernel(const float* __restrict__ part_acc,
                                                        const float* __restrict__ part_m,
                                                        const float* __restrict__ part_l,
                                                        const int32_t* __restrict__ step, int n_chunks,
                                                        float* __restrict__ attn, long long scratch_stride = 0) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    const int n_ids = __ldg(step + (size_t) blockIdx.y * kStepCount + kStepWidth);
    const int live = min(n_chunks, (n_ids + CHUNK - 1) / CHUNK);
    const int h = blockIdx.x;                 // global query head
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    if (live <= 0) {
        attn[(size_t) h * HD + d] = 0.0f;
        return;
    }
    if (live == 1) {
        const int slot = kvh * n_chunks;
        const float l = part_l[slot * G + hl];
        attn[(size_t) h * HD + d] = l > 0.0f ? part_acc[((size_t) slot * G + hl) * HD + d] / l : 0.0f;
        return;
    }
    float M = -FLT_MAX;
    for (int c = 0; c < live; ++c) M = fmaxf(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < live; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[slot * G + hl], w, L);
        acc = fmaf(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

template <int G>
void launch_decode_attn_batch_g(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                                int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q,
                                int kv_mode, cudaStream_t st) {
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    if (kv_mode == 3)
        attn_chunk_kernel<3, G><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else if (kv_mode == 2)
        attn_chunk_kernel<2, G><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else if (kv_mode == 1) {
        if (p100_attn_dec_enabled())
            attn_chunk_q8_p100_kernel<G><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                                   scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
        else
            attn_chunk_kernel<1, G><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                               scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    } else
        attn_chunk_kernel<0, G><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    attn_merge_kernel<G><<<dim3((unsigned) s.n_head, (unsigned) n_q), HD, 0, st>>>(part_acc, part_m, part_l, steps,
                                                                                   n_chunks, attn, stride);
}

template <int G>
void launch_decode_attn_step_g(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                               int64_t cap, const QsaShapes& s, float* scratch, float* attn, int kv_mode,
                               cudaStream_t st) {
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv);
    if (kv_mode == 3)
        attn_chunk_kernel<3, G><<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks);
    else if (kv_mode == 2)
        attn_chunk_kernel<2, G><<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks);
    else if (kv_mode == 1) {
        if (p100_attn_dec_enabled())
            attn_chunk_q8_p100_kernel<G><<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                                   scale, part_acc, part_m, part_l, n_chunks);
        else
            attn_chunk_kernel<1, G><<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                               scale, part_acc, part_m, part_l, n_chunks);
    } else
        attn_chunk_kernel<0, G><<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks);
    attn_merge_kernel<G><<<(unsigned) s.n_head, HD, 0, st>>>(part_acc, part_m, part_l, step, n_chunks, attn);
}

}  // namespace

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    const int g = (s.n_head_kv > 0 && s.n_head % s.n_head_kv == 0) ? (int) (s.n_head / s.n_head_kv) : 0;
    if (s.head_dim != HD || (g != 12 && g != 6) || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    cudaStream_t st = (cudaStream_t) stream;
    if (g == 6) launch_decode_attn_batch_g<6>(q, pools, ids, steps, cap, s, scratch, attn, n_q, kv_mode, st);
    else launch_decode_attn_batch_g<12>(q, pools, ids, steps, cap, s, scratch, attn, n_q, kv_mode, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    const int g = (s.n_head_kv > 0 && s.n_head % s.n_head_kv == 0) ? (int) (s.n_head / s.n_head_kv) : 0;
    if (s.head_dim != HD || (g != 12 && g != 6) || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    if (g == 6) launch_decode_attn_step_g<6>(q, pools, ids, step, cap, s, scratch, attn, kv_mode, st);
    else launch_decode_attn_step_g<12>(q, pools, ids, step, cap, s, scratch, attn, kv_mode, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
