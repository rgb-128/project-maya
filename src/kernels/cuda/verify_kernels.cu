// src/kernels/cuda/verify_kernels.cu - see include/strata/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

__global__ void __launch_bounds__(S) gdn_conv_l2_multi_kernel(const float* __restrict__ hist,
                                                              const float* __restrict__ qkv,
                                                              const float* __restrict__ w, float* __restrict__ h,
                                                              int C, int qk_heads, float eps, int t_begin) {
    __shared__ float part[S / 32];
    const int t = t_begin + blockIdx.y;
    const int c = blockIdx.x * S + threadIdx.x;
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    if ((int) blockIdx.x < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

__global__ void gdn_conv_commit_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int C,
                                       const int32_t* __restrict__ n_keep) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

template<int NT>
__global__ void __launch_bounds__(128) gdn_ab_multi_kernel_nt(const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                                              const uint16_t* __restrict__ wb,
                                                              const float* __restrict__ dt,
                                                              const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                              float* __restrict__ beta, int n, int h_v, int T) {
    // 1 block (4 warps = 128 threads) per output row -> 64 blocks on 56 SMs (was 8 blocks!).
    const int row = (int) blockIdx.x;
    if (row >= 2 * h_v) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) r * n);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    for (int j = tid; j < n / 8; j += 128) {
        const uint4 wv = __ldg(w4 + j);
        const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
        const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
        const float w4f = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
        const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if constexpr (NT > 4) { if (t >= T) break; }
            const float* xt = x + (size_t) t * n;
            const float4 xa = __ldg(reinterpret_cast<const float4*>(xt + j * 8));
            const float4 xb = __ldg(reinterpret_cast<const float4*>(xt + j * 8 + 4));
            float a = acc[t];
            a = fmaf(w0, xa.x, a); a = fmaf(w1, xa.y, a);
            a = fmaf(w2, xa.z, a); a = fmaf(w3, xa.w, a);
            a = fmaf(w4f, xb.x, a); a = fmaf(w5, xb.y, a);
            a = fmaf(w6, xb.z, a); a = fmaf(w7, xb.w, a);
            acc[t] = a;
        }
    }
    __shared__ float ws[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if constexpr (NT > 4) { if (t >= T) break; }
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) ws[t][warp] = a;
    }
    __syncthreads();
    if (tid < T) {
        const float a = ws[tid][0] + ws[tid][1] + ws[tid][2] + ws[tid][3];
        if (is_beta) {
            beta[(size_t) tid * h_v + r] = 1.0f / (1.0f + __expf(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : log1pf(__expf(v));
            gate[(size_t) tid * h_v + r] = sp * ssm_a[r];
        }
    }
}

__global__ void __launch_bounds__(S * RG) gdn_step_norm_multi_kernel(float* __restrict__ state,
                                                                     const float* __restrict__ hbuf, int C,
                                                                     const float* __restrict__ gate,
                                                                     const float* __restrict__ beta,
                                                                     const float* __restrict__ z,
                                                                     const float* __restrict__ gamma, float eps,
                                                                     float* __restrict__ y, int h_k, int h_v, int T,
                                                                     const int32_t* __restrict__ n_keep, int t_out_begin) {
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int tid = rg * S + col;
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();                // the previous token is done with sk/sq/red/wsum
        if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        if (t < t_out_begin) continue;   // a replayed token: its state update is needed, its output is not
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            y[(size_t) t * value_dim + head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
        }
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

// GP100 TP-sharded GDN step (h_v <= 24): 4 column blocks per head (48 blocks for h_v=12 -> 4x more SMs!)
// + rg = tid & 3 in SAME WARP (warp-shuffle reduction for kv and o -> ZERO red[][] smem barriers!).
__global__ void __launch_bounds__(128) gdn_step_cols4_kernel(float* __restrict__ state,
                                                             const float* __restrict__ hbuf, int C,
                                                             const float* __restrict__ gate,
                                                             const float* __restrict__ beta,
                                                             float* __restrict__ oc_out, int h_k, int h_v, int T,
                                                             const int32_t* __restrict__ n_keep, int t_out_begin) {
    __shared__ float sk[S], sq[S];
    const int head = (int) blockIdx.x >> 2;
    const int cb = (int) blockIdx.x & 3;
    const int tid = threadIdx.x;
    const int rg = tid & 3;           // 0..3 in same warp!
    const int c = tid >> 2;           // 0..31
    const int col = cb * 32 + c;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = __ldg(base + r * row_stride);
    const float inv_s = rsqrtf((float) S);
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();
        sk[tid] = __ldg(ht + qk + qh * S + tid);
        sq[tid] = __ldg(ht + qh * S + tid);
        __syncthreads();
        const float g = __expf(__ldg(gate + (size_t) t * h_v + head));
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float delta = (__ldg(ht + 2 * qk + head * S + col) - g * kv) * __ldg(beta + (size_t) t * h_v + head);
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            if (t >= t_out_begin) o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        if (t >= t_out_begin) {
            o += __shfl_xor_sync(0xffffffffu, o, 1);
            o += __shfl_xor_sync(0xffffffffu, o, 2);
            if (rg == 0) oc_out[(size_t) t * value_dim + head * S + col] = o * inv_s;
        }
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

struct VkQ81Block {
    __half2 ds;
    int8_t qs[32];
};

__global__ void __launch_bounds__(S) gdn_step_norm_out_kernel(float* __restrict__ y, const float* __restrict__ z,
                                                              const float* __restrict__ gamma, float eps,
                                                              int h_v, int t_out_begin,
                                                              VkQ81Block* __restrict__ y_q8) {
    __shared__ float ws[4];
    const int t = t_out_begin + (int) blockIdx.y;
    const int head = (int) blockIdx.x;
    const int col = threadIdx.x;
    const size_t at = (size_t) t * h_v * S + (size_t) head * S + col;
    const float oc = y[at];
    float sp = oc * oc;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
    if ((col & 31) == 0) ws[col >> 5] = sp;
    __syncthreads();
    const float ss = ws[0] + ws[1] + ws[2] + ws[3];
    const float zz = __ldg(z + at);
    const float xi = oc * rsqrtf(ss / (float) S + eps) * __ldg(gamma + col) * (1.0f / (1.0f + __expf(-zz)));
    if (y_q8 != nullptr) {
        float amax = fabsf(xi), sum = xi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            sum += __shfl_xor_sync(0xffffffffu, sum, o);
        }
        const float d = fminf(amax / 127.0f, 65504.0f);
        const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(xi / d)));
        VkQ81Block* bq = y_q8 + (size_t) t * (h_v * 4) + (size_t) head * 4 + (col >> 5);
        bq->qs[col & 31] = q;
        if ((col & 31) == 0) bq->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
    } else {
        y[at] = xi;
    }
}

__global__ void embedding_gather_dev_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                                            const float* __restrict__ offsets, const int32_t* __restrict__ tokens,
                                            int64_t n, int code_bits, int code_bias, int group_elems,
                                            unsigned long long row_codes, unsigned long long row_groups,
                                            float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long token = (unsigned long long) tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = __fmul_rn((float) (code + code_bias), sc[group]);
    out[(size_t) t * n + i] = __fadd_rn(product, of ? of[group] : 0.0f);
}

__global__ void broadcast_streams_kernel(const float* __restrict__ x, float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

__global__ void copy_indexed_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t stride,
                                    const int32_t* __restrict__ index, int64_t n) {
    const int idx = *index;
    if (idx < 0) return;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[(size_t) idx * stride + i];
}

__global__ void fetch_blobs_kernel(const unsigned long long* __restrict__ src, const int32_t* __restrict__ n,
                                   uint4* __restrict__ dst, long long per) {
    const long long total = (long long) *n * per;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long) gridDim.x * blockDim.x) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const uint4*) src[k])[off];
    }
}

__global__ void rebase_ptrs_kernel(unsigned long long* ptr, const int32_t* n, unsigned long long base, long long bytes) {
    const int k = threadIdx.x;
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

__global__ void add_streams_broadcast_kernel(const float* __restrict__ h, const float* __restrict__ e,
                                             float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

__global__ void ident_hits_kernel(const int32_t* __restrict__ ids, int n, int32_t* __restrict__ slot,
                                  int32_t* __restrict__ dst, int32_t* __restrict__ count) {
    const int i = threadIdx.x;
    if (i < n) { slot[i] = ids[i]; dst[i] = i; }
    if (i == 0) *count = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template<typename E>
__global__ void gather_rows_kernel(const E* __restrict__ src, long long row_e, const int32_t* __restrict__ ids,
                                   long long n, E* __restrict__ dst) {
    const long long total = n * row_e;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

__global__ void map_ids_kernel(int32_t* ids, const int32_t* __restrict__ table, int n) {
    const int i = threadIdx.x;
    if (i < n) ids[i] = table[ids[i]];
}

__global__ void row_top_prob_kernel(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ ids,
                                    float* __restrict__ probs) {
    __shared__ float part[32];
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
    if ((n_vocab & 3) == 0 && (reinterpret_cast<uintptr_t>(l) & 15) == 0) {
        const float4* l4 = reinterpret_cast<const float4*>(l);
        const int n4 = n_vocab >> 2;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = __ldg(l4 + i);
            s += __expf(v.x - m) + __expf(v.y - m) + __expf(v.z - m) + __expf(v.w - m);
        }
    } else {
        for (int i = threadIdx.x; i < n_vocab; i += blockDim.x) s += __expf(__ldg(l + i) - m);
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) tot += part[w];
        probs[t] = 1.0f / tot;
    }
}

__global__ void __launch_bounds__(1024) mtp_greedy_prob_map_kernel(
    const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ dvocab,
    int32_t* __restrict__ out_ids, float* __restrict__ probs) {
    __shared__ float sm[32];
    __shared__ float ss[32];
    __shared__ int32_t sid[32];
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const float* l = logits + (size_t) t * n_vocab;

    float m = -3.402823466e+38f;
    float s = 0.0f;
    int32_t best_id = -1;
    auto upd = [&](float v, int32_t idx) {
        if (best_id < 0) {
            m = v; best_id = idx; s = 1.0f;
        } else if (v > m || (v == m && idx < best_id)) {
            s = s * __expf(m - v) + 1.0f;
            m = v; best_id = idx;
        } else {
            s += __expf(v - m);
        }
    };

    if ((n_vocab & 3) == 0 && (reinterpret_cast<uintptr_t>(l) & 15) == 0) {
        const float4* l4 = reinterpret_cast<const float4*>(l);
        const int n4 = n_vocab >> 2;
        for (int i = tid; i < n4; i += 1024) {
            const float4 v = __ldg(l4 + i);
            const int32_t base = i << 2;
            upd(v.x, base + 0);
            upd(v.y, base + 1);
            upd(v.z, base + 2);
            upd(v.w, base + 3);
        }
    } else {
        for (int i = tid; i < n_vocab; i += 1024) upd(__ldg(l + i), i);
    }

    auto merge_pair = [](float& am, int32_t& aid, float& as, float bm, int32_t bid, float bs) {
        if (bid < 0) return;
        if (aid < 0) { am = bm; aid = bid; as = bs; return; }
        if (bm > am || (bm == am && bid < aid)) {
            as = as * __expf(am - bm) + bs;
            am = bm; aid = bid;
        } else {
            as = as + bs * __expf(bm - am);
        }
    };
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float om = __shfl_xor_sync(0xffffffffu, m, o);
        const int32_t oid = __shfl_xor_sync(0xffffffffu, best_id, o);
        const float os = __shfl_xor_sync(0xffffffffu, s, o);
        merge_pair(m, best_id, s, om, oid, os);
    }
    if ((tid & 31) == 0) {
        const int w = tid >> 5;
        sm[w] = m; sid[w] = best_id; ss[w] = s;
    }
    __syncthreads();
    if (tid < 32) {
        m = sm[tid]; best_id = sid[tid]; s = ss[tid];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float om = __shfl_xor_sync(0xffffffffu, m, o);
            const int32_t oid = __shfl_xor_sync(0xffffffffu, best_id, o);
            const float os = __shfl_xor_sync(0xffffffffu, s, o);
            merge_pair(m, best_id, s, om, oid, os);
        }
        if (tid == 0) {
            out_ids[t] = (dvocab != nullptr) ? dvocab[best_id] : best_id;
            probs[t] = 1.0f / s;
        }
    }
}

__global__ void mtp_select_kernel(const float* __restrict__ R_src, int64_t stride, const int32_t* __restrict__ ids,
                                  const int32_t* __restrict__ row_dev, float* __restrict__ R_dst,
                                  int32_t* __restrict__ tok_dst, int32_t* out, int j, const float* probs, float* out_p,
                                  uint32_t* done_flag, uint32_t done_val) {
    const int row = *row_dev;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += (int64_t) gridDim.x * blockDim.x)
        R_dst[i] = R_src[(size_t) row * stride + i];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
        if (done_flag != nullptr) {
            __threadfence_system();
            *(volatile uint32_t*) done_flag = done_val;
            __threadfence_system();
        }
    }
}

__global__ void dense_steps_kernel(const int32_t* __restrict__ cells, int n, int32_t* __restrict__ steps) {
    const int i = threadIdx.x;
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    fetch_blobs_kernel<<<48 * 8, 256, 0, (cudaStream_t) stream>>>(src, n, (uint4*) dst, (long long) (blob_bytes / 16));
    check("fetch_blobs");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    rebase_ptrs_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ptr, n, (unsigned long long) base, (long long) blob_bytes);
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    add_streams_broadcast_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                                   (cudaStream_t) stream>>>(h, e, R, n_embd, hc);
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    ident_hits_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(ids, n, slot, dst, count);
    check("ident_hits");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p,
                uint32_t* done_flag, uint32_t done_val) {
    mtp_select_kernel<<<16, 256, 0, (cudaStream_t) stream>>>(R_src, R_stride, ids, row_dev, R_dst, tok_dst, out, j,
                                                             probs, out_p, done_flag, done_val);
    check("mtp_select");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    if (row_bytes % 16 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint4*) src, row_bytes / 16, ids, n, (uint4*) dst);
    else if (row_bytes % 4 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint32_t*) src, row_bytes / 4, ids, n, (uint32_t*) dst);
    else
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>(src, row_bytes, ids, n, dst);
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    map_ids_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(ids, table, n);
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    row_top_prob_kernel<<<n_rows, 1024, 0, (cudaStream_t) stream>>>(logits, n_vocab, ids, probs);
    check("row_top_prob");
}

void mtp_greedy_prob_map(const float* logits, int n_rows, int n_vocab, const int32_t* dvocab,
                         int32_t* out_ids, float* probs, void* stream) {
    mtp_greedy_prob_map_kernel<<<n_rows, 1024, 0, (cudaStream_t) stream>>>(logits, n_vocab, dvocab, out_ids, probs);
    check("mtp_greedy_prob_map");
}

namespace {
__global__ void window_ids_kernel(int32_t* steps, int window, int32_t* ids, long long stride) {
    const int q = blockIdx.y;
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x)
        ids[q * stride + j] = start + j;
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) st[3] = width;
}
__global__ void handoff_pack_kernel(float* __restrict__ hand_out, const float* __restrict__ R,
                                    const float* __restrict__ bo, const float* __restrict__ inj2,
                                    int hc_n4, int n4, int hc, int hb) {
    const int t = (int) blockIdx.y;
    const int tid = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    const int stride = (int) gridDim.x * (int) blockDim.x;
    float* ho = hand_out + (size_t) t * hb;
    float4* ho4 = reinterpret_cast<float4*>(ho);
    const float4* r4 = reinterpret_cast<const float4*>(R + (size_t) t * (hc_n4 * 4));
    const float4* b4 = reinterpret_cast<const float4*>(bo + (size_t) t * (n4 * 4));
    const int tot4 = hc_n4 + n4;
    for (int i = tid; i < tot4; i += stride) {
        ho4[i] = (i < hc_n4) ? r4[i] : b4[i - hc_n4];
    }
    if (tid < hc) {
        ho[tot4 * 4 + tid] = inj2[(size_t) t * hc + tid];
    }
}

__global__ void handoff_unpack_kernel(float* __restrict__ R, float* __restrict__ bo, float* __restrict__ inj2,
                                      const volatile float* __restrict__ hand_in,
                                      int hc_n4, int n4, int hc, int hb) {
    const int t = (int) blockIdx.y;
    const int tid = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    const int stride = (int) gridDim.x * (int) blockDim.x;
    const float* hi = const_cast<const float*>(hand_in) + (size_t) t * hb;
    const float4* hi4 = reinterpret_cast<const float4*>(hi);
    float4* r4 = reinterpret_cast<float4*>(R + (size_t) t * (hc_n4 * 4));
    float4* b4 = reinterpret_cast<float4*>(bo + (size_t) t * (n4 * 4));
    const int tot4 = hc_n4 + n4;
    for (int i = tid; i < tot4; i += stride) {
#if defined(__CUDA_ARCH__)
        const float4 v = __ldcv(hi4 + i);
#else
        const float4 v = hi4[i];
#endif
        if (i < hc_n4) r4[i] = v;
        else b4[i - hc_n4] = v;
    }
    if (tid < hc) {
#if defined(__CUDA_ARCH__)
        inj2[(size_t) t * hc + tid] = __ldcv(hi + tot4 * 4 + tid);
#else
        inj2[(size_t) t * hc + tid] = hi[tot4 * 4 + tid];
#endif
    }
}
}  // namespace

void handoff_pack(float* hand_out, const float* R, const float* bo, const float* inj2,
                  int64_t hc_n, int64_t n, int64_t hc, int64_t hb, int n_tok, void* stream) {
    const int hc_n4 = (int) (hc_n / 4), n4 = (int) (n / 4);
    const unsigned gx = (unsigned) ((hc_n4 + n4 + 255) / 256);
    handoff_pack_kernel<<<dim3(gx < 16u ? gx : 16u, (unsigned) n_tok), 256, 0, (cudaStream_t) stream>>>(
        hand_out, R, bo, inj2, hc_n4, n4, (int) hc, (int) hb);
    check("handoff_pack");
}

void handoff_unpack(float* R, float* bo, float* inj2, const float* hand_in,
                    int64_t hc_n, int64_t n, int64_t hc, int64_t hb, int n_tok, void* stream) {
    const int hc_n4 = (int) (hc_n / 4), n4 = (int) (n / 4);
    const unsigned gx = (unsigned) ((hc_n4 + n4 + 255) / 256);
    handoff_unpack_kernel<<<dim3(gx < 16u ? gx : 16u, (unsigned) n_tok), 256, 0, (cudaStream_t) stream>>>(
        R, bo, inj2, (const volatile float*) hand_in, hc_n4, n4, (int) hc, (int) hb);
    check("handoff_unpack");
}

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    window_ids_kernel<<<dim3(8, (unsigned) n), 256, 0, (cudaStream_t) stream>>>(steps, window, ids, (long long) ids_stride);
    check("window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    dense_steps_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(cells, n, steps);
    check("dense_steps");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    gdn_conv_l2_multi_kernel<<<dim3((unsigned) (channels / S), (unsigned) n_tok), S, 0, (cudaStream_t) stream>>>(
        history, qkv, conv_w, h, channels, qk_heads, eps, t_begin);
    check("gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    gdn_conv_commit_kernel<<<(unsigned) ((channels + 255) / 256), 256, 0, (cudaStream_t) stream>>>(history, qkv,
                                                                                                 channels, n_keep);
    check("gdn_conv_commit");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    const unsigned grid = (unsigned) (2 * h_v);
    cudaStream_t st = (cudaStream_t) stream;
    switch (n_tok) {
        case 1: gdn_ab_multi_kernel_nt<1><<<grid, 128, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 2: gdn_ab_multi_kernel_nt<2><<<grid, 128, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 3: gdn_ab_multi_kernel_nt<3><<<grid, 128, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 4: gdn_ab_multi_kernel_nt<4><<<grid, 128, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 5: gdn_ab_multi_kernel_nt<5><<<grid, 128, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 6: gdn_ab_multi_kernel_nt<6><<<grid, 128, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 7: gdn_ab_multi_kernel_nt<7><<<grid, 128, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        default: gdn_ab_multi_kernel_nt<kVerifyMaxT><<<grid, 128, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
    }
    check("gdn_ab_multi");
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin, void* y_q8) {
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    if (h_v <= 24) {
        gdn_step_cols4_kernel<<<(unsigned) (h_v * 4), 128, 0, st>>>(
            state, h, conv_channels, gate, beta, y, h_k, h_v, n_tok, n_keep, t_out_begin);
        const int n_out = n_tok - t_out_begin;
        if (n_out > 0) {
            gdn_step_norm_out_kernel<<<dim3((unsigned) h_v, (unsigned) n_out), S, 0, st>>>(
                y, z, gamma, eps, h_v, t_out_begin, reinterpret_cast<VkQ81Block*>(y_q8));
        }
    } else {
        gdn_step_norm_multi_kernel<<<(unsigned) h_v, dim3(S, RG), 0, st>>>(
            state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, n_keep, t_out_begin);
    }
    check("gdn_step_norm_multi");
}

namespace {
__global__ void wait_flag_ge_kernel(const volatile uint32_t* flag, uint32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    while (*flag < value) {}   // Pascal (sm_6x) has no __nanosleep: a plain spin
#else
    while (*flag < value) __nanosleep(100);
#endif
    __threadfence_system();
}
}  // namespace

namespace {
__global__ void resident_plan_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                     int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                     long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                     uint32_t ring, uint32_t* miss_flag) {
    // Warp-parallel (32 threads): handles up to 128 entries (kVerifyMaxT * 10 = 80).
    // Parallel res[e] lookup + shared-mem dedup replaces the old 1-thread serial loop (~60 us -> ~3 us!).
    __shared__ int32_t s_ids[128];
    __shared__ int32_t s_slot[128];
    __shared__ int32_t s_first[128];
    __shared__ int32_t s_gidx[128];
    __shared__ int32_t s_gstart[129];
    __shared__ int32_t s_rank[128];
    __shared__ int32_t s_missed;
    const int tid = threadIdx.x;
    if (tid == 0) s_missed = 0;
    __syncwarp();
    int local_miss = 0;
    for (int i = tid; i < n; i += 32) {
        const int32_t e = ids[i];
        s_ids[i] = e;
        if (e < 0) {
            s_slot[i] = -2;  // skipped/dropped
        } else if (e >= n_expert) {
            s_slot[i] = -1;
            local_miss = 1;
        } else {
            const int32_t sl = res[e];
            s_slot[i] = sl;
            if (sl < 0) local_miss = 1;
        }
    }
    if (__any_sync(0xffffffffu, local_miss != 0)) {
        if (tid == 0) {
            pl[0] = 0; pl[1] = 0; pl[2] = 0; pl[4] = 0;
            *skip = 0;
            if (miss_flag != nullptr) {
                __threadfence_system();
                *(volatile uint32_t*) miss_flag = 1u;
            }
        }
        return;
    }
    // Find first occurrence, within-group rank, and group count in parallel across the warp
    __shared__ int32_t s_cnt[128];
    for (int i = tid; i < n; i += 32) {
        const int32_t e = s_ids[i];
        if (e < 0) { s_first[i] = -1; s_rank[i] = 0; s_cnt[i] = 0; continue; }
        int first = i, rank = 0, cnt = 0;
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == e) {
                ++cnt;
                if (j < i) {
                    if (first == i) first = j;
                    ++rank;
                }
            }
        }
        s_first[i] = first;
        s_rank[i] = rank;
        s_cnt[i] = (first == i) ? cnt : 0;
    }
    __syncwarp();
    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;
    if (tid == 0) {
        int groups = 0, entries = 0;
        for (int i = 0; i < n; ++i) {
            const int cnt = s_cnt[i];
            if (cnt == 0) continue;
            s_gidx[i] = groups;
            s_gstart[groups] = entries;
            start[groups] = entries;
            entries += cnt;
            ++groups;
        }
        s_gstart[groups] = entries;
        start[groups] = entries;
        start2[0] = entries;
        counts[0] = groups;
        counts[1] = entries;
        counts[2] = 0;
    }
    __syncwarp();
    // Parallel ptr + dst + tok writes!
    for (int i = tid; i < n; i += 32) {
        const int f = s_first[i];
        if (f < 0) continue;
        const int gi = s_gidx[f];
        if (f == i) {
            const int32_t slot = s_slot[i];
            ptr[gi] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        }
        const int at = s_gstart[gi] + s_rank[i];
        dst[at] = i;
        tok[at] = i / k;
    }
    __syncwarp();
    if (tid == 0) {
        if (miss_flag == nullptr) __threadfence();
        *skip = ring;
    }
}

__device__ __forceinline__ float vr_warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    return v;
}
__device__ __forceinline__ float vr_warp_max(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m, 32));
    return v;
}

// Fused exact top-10 router (bit-identical to native_router.cu) + resident_plan in ONE single block of n_tok*32 threads.
__global__ void fused_router_top10_resident_plan_kernel(
    const float* __restrict__ logits, int n_tok, int32_t* __restrict__ ids, float* __restrict__ weights,
    const int32_t* __restrict__ res, const uint8_t* cache_base, const unsigned long long* slot_off,
    long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip, uint32_t ring, uint32_t* miss_flag) {
    __shared__ int32_t s_ids[128];
    __shared__ int32_t s_slot[128];
    __shared__ int32_t s_first[128];
    __shared__ int32_t s_gidx[128];
    __shared__ int32_t s_gstart[129];
    __shared__ int32_t s_rank[128];
    __shared__ int32_t s_cnt[128];
    __shared__ int32_t s_missed;

    const int tid = threadIdx.x;
    const int tok = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) s_missed = 0;
    __syncthreads();

    if (tok < n_tok) {
        const float* lg = logits + (size_t) tok * 512;
        int32_t* id_out = ids + (size_t) tok * 10;
        float* w_out = weights + (size_t) tok * 10;
        float values[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) values[i] = lg[lane + i * 32];
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < 16; ++i) maximum = fmaxf(maximum, values[i]);
        maximum = vr_warp_max(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] = expf(values[i] - maximum);
            sum += values[i];
        }
        const float reciprocal = 1.0f / vr_warp_sum(sum);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] *= reciprocal;
            if (__isnanf(values[i])) values[i] = -3.402823466e+38f;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        int local_miss = 0;
        for (int rank = 0; rank < 10; ++rank) {
            float best = values[0];
            int expert = lane;
#pragma unroll
            for (int i = 1; i < 16; ++i) {
                if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
            }
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
                const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
                if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
            }
            if ((expert & 31) == lane) {
                values[expert / 32] = -INFINITY;
                id_out[rank] = expert;
                const int eidx = tok * 10 + rank;
                s_ids[eidx] = expert;
                const int32_t sl = res[expert];
                s_slot[eidx] = sl;
                if (sl < 0) local_miss = 1;
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        if (__any_sync(0xffffffffu, local_miss != 0) && lane == 0) atomicExch(&s_missed, 1);
        selected_sum = fmaxf(vr_warp_sum(selected_sum), 6.103515625e-5f);
        const float inv_sum = 1.0f / selected_sum;
        if (lane < 10) w_out[lane] = selected * inv_sum;
    }
    __syncthreads();

    if (s_missed != 0) {
        if (tid == 0) {
            pl[0] = 0; pl[1] = 0; pl[2] = 0; pl[4] = 0;
            *skip = 0;
            if (miss_flag != nullptr) {
                __threadfence_system();
                *(volatile uint32_t*) miss_flag = 1u;
            }
        }
        return;
    }

    const int n = n_tok * 10;
    const int n_threads = n_tok * 32;
    for (int i = tid; i < n; i += n_threads) {
        const int32_t e = s_ids[i];
        int first = i, rank = 0, cnt = 0;
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == e) {
                ++cnt;
                if (j < i) {
                    if (first == i) first = j;
                    ++rank;
                }
            }
        }
        s_first[i] = first;
        s_rank[i] = rank;
        s_cnt[i] = (first == i) ? cnt : 0;
    }
    __syncthreads();

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok_out = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;
    if (tid == 0) {
        int groups = 0, entries = 0;
        for (int i = 0; i < n; ++i) {
            const int cnt = s_cnt[i];
            if (cnt == 0) continue;
            s_gidx[i] = groups;
            s_gstart[groups] = entries;
            start[groups] = entries;
            entries += cnt;
            ++groups;
        }
        s_gstart[groups] = entries;
        start[groups] = entries;
        start2[0] = entries;
        counts[0] = groups;
        counts[1] = entries;
        counts[2] = 0;
    }
    __syncthreads();

    for (int i = tid; i < n; i += n_threads) {
        const int f = s_first[i];
        const int gi = s_gidx[f];
        if (f == i) {
            const int32_t slot = s_slot[i];
            ptr[gi] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        }
        const int at = s_gstart[gi] + s_rank[i];
        dst[at] = i;
        tok_out[at] = i / 10;
    }
    __syncthreads();
    if (tid == 0) {
        if (miss_flag == nullptr) __threadfence();
        *skip = ring;
    }
}
__global__ void wait_flag_ge_or_kernel(const volatile uint32_t* flag, uint32_t value, const volatile uint32_t* skip) {
    if (*skip == value) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    while (*flag < value) {}   // Pascal (sm_6x) has no __nanosleep: a plain spin
#else
    while (*flag < value) __nanosleep(100);
#endif
    __threadfence_system();
}
__global__ void copy_i32_unless_kernel(int32_t* __restrict__ dst, const volatile int32_t* src, int n,
                                       const uint32_t* skip, uint32_t value) {
    if (*skip == value) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
__global__ void copy_or_zero_kernel(float4* __restrict__ dst, const volatile float4* src, long long n4,
                                    const uint32_t* skip, uint32_t value) {
    const bool zero = *skip == value;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long) gridDim.x * blockDim.x)
        dst[i] = zero ? make_float4(0.f, 0.f, 0.f, 0.f) : const_cast<const float4*>(src)[i];
}
__global__ void copy_f32_unless_kernel(float4* __restrict__ dst, const volatile float4* src, long long n4,
                                       const uint32_t* skip, uint32_t value) {
    if (*skip == value) return;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long) gridDim.x * blockDim.x)
        dst[i] = const_cast<const float4*>(src)[i];
}
}  // namespace

void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* miss_flag) {
    resident_plan_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, n_entries, k, res_layer, n_expert, cache_base,
                                                              slot_off, blob, plan, capx, skip, ring, miss_flag);
    check("resident_plan");
}
void fused_router_top10_resident_plan(const float* logits, int n_tok, int32_t* ids, float* weights,
                                      const int32_t* res_layer, const uint8_t* cache_base,
                                      const unsigned long long* slot_off, long long blob, int32_t* plan,
                                      long long capx, uint32_t* skip, uint32_t ring, void* stream,
                                      uint32_t* miss_flag) {
    fused_router_top10_resident_plan_kernel<<<1, (unsigned) (n_tok * 32), 0, (cudaStream_t) stream>>>(
        logits, n_tok, ids, weights, res_layer, cache_base, slot_off, blob, plan, capx, skip, ring, miss_flag);
    check("fused_router_top10_resident_plan");
}
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    wait_flag_ge_or_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value, skip);
    check("wait_flag_ge_or");
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    copy_i32_unless_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(dst, (const volatile int32_t*) src, (int) n, skip, value);
    check("copy_i32_from_mapped_unless");
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    copy_or_zero_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src, n4, skip,
                                                                    value);
    check("copy_or_zero_from_mapped");
}
void copy_from_mapped_unless(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                             void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    copy_f32_unless_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src, n4,
                                                                      skip, value);
    check("copy_from_mapped_unless");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    wait_flag_ge_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value);
    check("wait_flag_ge");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    embedding_gather_dev_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) n_tok), 256, 0,
                                  (cudaStream_t) stream>>>(codes, scales, offsets, tokens, n, code_bits, code_bias,
                                                           group_elems, row_codes, row_groups, out);
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    broadcast_streams_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                               (cudaStream_t) stream>>>(x, R, n_embd, hc);
    check("broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    copy_indexed_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(dst, src, stride, index, n);
    check("copy_indexed");
}

// a GPU timestamp (ns, %globaltimer) into buf[i] - the verify window's stage profiler
namespace { __global__ void gpu_stamp_kernel(unsigned long long* buf, int i) {
    unsigned long long t;
#if defined(__HIPCC__)
    t = wall_clock64() * 10ull;   // gfx11: a constant 100 MHz counter, in ns
#else
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
#endif
    buf[i] = t;
} }
void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    gpu_stamp_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(buf, i);
}

}  // namespace strata::kernels
