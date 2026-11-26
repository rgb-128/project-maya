#include "strata/kernels/tp_allreduce.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE (1u << 1)
#endif
#endif

namespace strata::kernels {
namespace {

__device__ __forceinline__ uint32_t tp_ld_sys_u32(const uint32_t* p) {
    // Must be a true C++ volatile load (ld.volatile.global -> LDG.E.CV on sm_60);
    // CUDA 12.4 ptxas dead-code-eliminates empty spin loops around inline asm ld.global.cv!
    return *(const volatile uint32_t*) p;
}

__device__ __forceinline__ uint2 tp_ld_cv_u2(const uint2* p) {
#if defined(__HIPCC__)
    const volatile uint32_t* q = reinterpret_cast<const volatile uint32_t*>(p);
    return make_uint2(q[0], q[1]);
#else
    return __ldcv(p);
#endif
}

__device__ __forceinline__ float4 tp_ld_cv_f4(const float4* p) {
#if defined(__HIPCC__)
    const volatile float* q = reinterpret_cast<const volatile float*>(p);
    return make_float4(q[0], q[1], q[2], q[3]);
#else
    return __ldcv(p);
#endif
}

__device__ __forceinline__ uint2 tp_f4_to_h4(float4 v) {
    const float m = 65504.0f;
    const __half2 a = __floats2half2_rn(fminf(fmaxf(v.x, -m), m), fminf(fmaxf(v.y, -m), m));
    const __half2 b = __floats2half2_rn(fminf(fmaxf(v.z, -m), m), fminf(fmaxf(v.w, -m), m));
    return make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
}

__device__ __forceinline__ float4 tp_h4_to_f4(uint2 u) {
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ float tp_warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    return v;
}

__device__ __forceinline__ float tp_warp_max(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m, 32));
    return v;
}

__device__ __forceinline__ uint4 tp_ld_cv_u4(const uint4* p) {
#if defined(__HIPCC__)
    const volatile uint32_t* q = reinterpret_cast<const volatile uint32_t*>(p);
    return make_uint4(q[0], q[1], q[2], q[3]);
#else
    return __ldcv(p);
#endif
}

__device__ __forceinline__ uint4 tp_f4x2_to_h8(float4 a, float4 b) {
    const uint2 u0 = tp_f4_to_h4(a);
    const uint2 u1 = tp_f4_to_h4(b);
    return make_uint4(u0.x, u0.y, u1.x, u1.y);
}

__device__ __forceinline__ void tp_h8_to_f4x2(uint4 u, float4& a, float4& b) {
    a = tp_h4_to_f4(make_uint2(u.x, u.y));
    b = tp_h4_to_f4(make_uint2(u.z, u.w));
}

template <int NGPU>
__global__ void __launch_bounds__(256) tp_ar_host_ag_f16_u4_kernel(
    float4* __restrict__ local_x,
    uint4* __restrict__ h_rows,
    uint32_t* __restrict__ h_flags,
    int64_t stride8,
    int rank,
    uint32_t seq,
    int n8) {
    const int slot = (int) (seq & 1u);
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int per = (n8 + gridDim.x - 1) / gridDim.x;
    const int lo = bid * per;
    const int hi = min(n8, lo + per);
    auto row = [&](int r) { return h_rows + ((size_t) r * 2 + (size_t) slot) * (size_t) stride8; };
    auto flg = [&](int r) { return h_flags + (((size_t) r * 2 + (size_t) slot) * kTpMaxBlocks + (size_t) bid) * 16; };

    uint4* mine = row(rank);
    for (int i = lo + tid; i < hi; i += blockDim.x) {
        mine[i] = tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]);
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) flg(rank) = seq;
        __threadfence_system();
        if (seq > 0u) {
#pragma unroll
            for (int r = 0; r < NGPU; ++r) {
                if (r == rank) continue;
                while ((int) (tp_ld_sys_u32(flg(r)) - seq) < 0) {}
            }
            __threadfence_system();
        }
    }
    __syncthreads();
    if (seq == 0u) return;
    for (int i = lo + tid; i < hi; i += blockDim.x) {
        float4 acc0, acc1;
        if (rank == 0) tp_h8_to_f4x2(tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]), acc0, acc1);
        else tp_h8_to_f4x2(tp_ld_cv_u4(row(0) + i), acc0, acc1);
#pragma unroll
        for (int r = 1; r < NGPU; ++r) {
            float4 p0, p1;
            if (r == rank) tp_h8_to_f4x2(tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]), p0, p1);
            else tp_h8_to_f4x2(tp_ld_cv_u4(row(r) + i), p0, p1);
            acc0.x = __fadd_rn(acc0.x, p0.x); acc0.y = __fadd_rn(acc0.y, p0.y);
            acc0.z = __fadd_rn(acc0.z, p0.z); acc0.w = __fadd_rn(acc0.w, p0.w);
            acc1.x = __fadd_rn(acc1.x, p1.x); acc1.y = __fadd_rn(acc1.y, p1.y);
            acc1.z = __fadd_rn(acc1.z, p1.z); acc1.w = __fadd_rn(acc1.w, p1.w);
        }
        local_x[2 * i] = acc0;
        local_x[2 * i + 1] = acc1;
    }
}

// NUMA-Split 4-rank Flat AG (Decode, Host-Flag-Only, 0 QPI writes, Single-TX Pure-RAM DDIO default, STACK:0!)
template <bool USE_P2P>
__global__ void __launch_bounds__(256) tp_ar_host_ag_f16_u4_split4_kernel(
    float4* __restrict__ local_x,
    uint4* __restrict__ r0, uint4* __restrict__ r1, uint4* __restrict__ r2, uint4* __restrict__ r3,
    uint32_t* __restrict__ f0, uint32_t* __restrict__ f1, uint32_t* __restrict__ f2, uint32_t* __restrict__ f3,
    uint4* __restrict__ p2p_me, uint4* __restrict__ p2p_peer, int64_t p2p_stride8,
    int64_t stride8, int rank, uint32_t seq, int n8) {
    auto row_ptr = [&](int r) -> uint4* { return r == 0 ? r0 : (r == 1 ? r1 : (r == 2 ? r2 : r3)); };
    auto flg_ptr = [&](int r) -> uint32_t* { return r == 0 ? f0 : (r == 1 ? f1 : (r == 2 ? f2 : f3)); };
    const int slot = (int) (seq & 1u);
    const int peer = rank ^ 1;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int per = (n8 + gridDim.x - 1) / gridDim.x;
    const int lo = bid * per;
    const int hi = min(n8, lo + per);
    const size_t soff = (size_t) slot * (size_t) stride8;
    const size_t p_soff = (size_t) slot * (size_t) p2p_stride8;
    const size_t foff = ((size_t) slot * kTpMaxBlocks + (size_t) bid) * 16;

    uint4* mine = row_ptr(rank) + soff;
    uint4* pp = (USE_P2P && p2p_peer != nullptr) ? (p2p_peer + p_soff) : nullptr;
    uint4 saved_pk[4];
    int n_saved = 0;
    for (int i = lo + tid; i < hi; i += blockDim.x) {
        const uint4 pk = tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]);
        if constexpr (USE_P2P) {
            if (pp != nullptr) pp[i] = pk;
        }
        mine[i] = pk;
        if (n_saved < 4) {
            if (n_saved == 0) saved_pk[0] = pk;
            else if (n_saved == 1) saved_pk[1] = pk;
            else if (n_saved == 2) saved_pk[2] = pk;
            else saved_pk[3] = pk;
        }
        ++n_saved;
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) (flg_ptr(rank) + foff) = seq;
        __threadfence_system();
        if (seq > 0u) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (r == rank) continue;
                while ((int) (tp_ld_sys_u32(flg_ptr(r) + foff) - seq) < 0) {}
            }
            __threadfence_system();
        }
    }
    __syncthreads();
    if (seq == 0u) return;
    const uint4* pm = (USE_P2P && p2p_me != nullptr) ? (p2p_me + p_soff) : nullptr;
    int iter = 0;
    for (int i = lo + tid; i < hi; i += blockDim.x, ++iter) {
        uint4 my_u;
        if (iter == 0) my_u = saved_pk[0];
        else if (iter == 1) my_u = saved_pk[1];
        else if (iter == 2) my_u = saved_pk[2];
        else if (iter == 3) my_u = saved_pk[3];
        else my_u = tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]);
        float4 acc0, acc1;
        if (rank == 0) tp_h8_to_f4x2(my_u, acc0, acc1);
        else if (USE_P2P && 0 == peer && pm != nullptr) tp_h8_to_f4x2(tp_ld_cv_u4(pm + i), acc0, acc1);
        else tp_h8_to_f4x2(tp_ld_cv_u4(r0 + soff + i), acc0, acc1);
#pragma unroll
        for (int r = 1; r < 4; ++r) {
            float4 p0, p1;
            if (r == rank) tp_h8_to_f4x2(my_u, p0, p1);
            else if (USE_P2P && r == peer && pm != nullptr) tp_h8_to_f4x2(tp_ld_cv_u4(pm + i), p0, p1);
            else tp_h8_to_f4x2(tp_ld_cv_u4(row_ptr(r) + soff + i), p0, p1);
            acc0.x = __fadd_rn(acc0.x, p0.x); acc0.y = __fadd_rn(acc0.y, p0.y);
            acc0.z = __fadd_rn(acc0.z, p0.z); acc0.w = __fadd_rn(acc0.w, p0.w);
            acc1.x = __fadd_rn(acc1.x, p1.x); acc1.y = __fadd_rn(acc1.y, p1.y);
            acc1.z = __fadd_rn(acc1.z, p1.z); acc1.w = __fadd_rn(acc1.w, p1.w);
        }
        local_x[2 * i] = acc0;
        local_x[2 * i + 1] = acc1;
    }
}

// Hybrid Intra-Socket BAR P2P + Cross-Socket NUMA-Local Hierarchical AllReduce (Prefill large tensors, 1.69x faster!).
// Uses half-sized VRAM P2P buffers (21 MiB/GPU @ T=2048), 0 DRAM/0 QPI in Stages 1 & 4, and 2.75x less QPI overall!
__global__ void __launch_bounds__(256) tp_ar_p2p_hier_f16_u4_kernel(
    float4* __restrict__ local_x,
    uint4* __restrict__ my_p2p_in,  uint4* __restrict__ peer_p2p_in,
    uint4* __restrict__ my_p2p_fin, uint4* __restrict__ peer_p2p_fin,
    uint32_t* __restrict__ my_p2p_flg, uint32_t* __restrict__ peer_p2p_flg,
    uint4* __restrict__ my_h_sum, uint4* __restrict__ rem_h_sum,
    uint32_t* __restrict__ my_h_flg, uint32_t* __restrict__ rem_h_flg,
    int64_t stride8, int64_t half_stride8, int rank, uint32_t seq, int n8) {
    const int lrank = rank & 1;
    const int slot = (int) (seq & 1u);
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int per = (n8 + gridDim.x - 1) / gridDim.x;
    const int lo = bid * per;
    const int hi = min(n8, lo + per);
    const int h_per = (per + 1) / 2;
    const int my_hlo = min(hi, lo + lrank * h_per);
    const int my_hhi = min(hi, my_hlo + h_per);
    const int ot_hlo = min(hi, lo + (1 - lrank) * h_per);
    const int ot_hhi = min(hi, ot_hlo + h_per);

    auto pflg = [&](uint32_t* base, int ph) { return base + (((ph * 2 + slot) * kTpMaxBlocks) + bid) * 16; };
    auto hflg = [&](uint32_t* base, int lr) { return base + (((lr * 2 + slot) * kTpMaxBlocks) + bid) * 16; };
    const size_t soff = (size_t) slot * (size_t) stride8;
    const size_t p_off = (size_t) slot * (size_t) half_stride8 + (size_t) bid * (size_t) h_per;

    if (seq == 0u) {
        if (tid == 0) {
            *(volatile uint32_t*) pflg(my_p2p_flg, 0) = 0u;
            *(volatile uint32_t*) pflg(my_p2p_flg, 1) = 0u;
            *(volatile uint32_t*) hflg(my_h_flg, lrank) = 0u;
            __threadfence_system();
        }
        return;
    }

    // Stage 1 (Intra-socket BAR P2P write, 4x unrolled pipelined PCIe bursts): push peer's half into peer VRAM
    uint4* p_in = peer_p2p_in + p_off;
    int i = ot_hlo + tid;
    for (; i + 768 < ot_hhi; i += 1024) {
        const uint4 v0 = tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]);
        const uint4 v1 = tp_f4x2_to_h8(local_x[2 * (i + 256)], local_x[2 * (i + 256) + 1]);
        const uint4 v2 = tp_f4x2_to_h8(local_x[2 * (i + 512)], local_x[2 * (i + 512) + 1]);
        const uint4 v3 = tp_f4x2_to_h8(local_x[2 * (i + 768)], local_x[2 * (i + 768) + 1]);
        p_in[i - ot_hlo] = v0;
        p_in[i + 256 - ot_hlo] = v1;
        p_in[i + 512 - ot_hlo] = v2;
        p_in[i + 768 - ot_hlo] = v3;
    }
    for (; i < ot_hhi; i += 256) {
        p_in[i - ot_hlo] = tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]);
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) pflg(peer_p2p_flg, 0) = seq;
        __threadfence_system();
        while ((int) (tp_ld_sys_u32(pflg(my_p2p_flg, 0)) - seq) < 0) {}
        __threadfence_system();
    }
    __syncthreads();

    // Stage 2 (HBM2 intra-socket reduce -> write LOCAL NUMA RAM my_h_sum + cache in HBM2 m_in, 4x unrolled)
    uint4* m_in = my_p2p_in + p_off;
    uint4* h_loc = my_h_sum + soff;
    auto add_u4_f32 = [](uint4 pa, uint4 pb) -> uint4 {
        float4 a0, a1, b0, b1;
        tp_h8_to_f4x2(pa, a0, a1);
        tp_h8_to_f4x2(pb, b0, b1);
        a0.x = __fadd_rn(a0.x, b0.x); a0.y = __fadd_rn(a0.y, b0.y);
        a0.z = __fadd_rn(a0.z, b0.z); a0.w = __fadd_rn(a0.w, b0.w);
        a1.x = __fadd_rn(a1.x, b1.x); a1.y = __fadd_rn(a1.y, b1.y);
        a1.z = __fadd_rn(a1.z, b1.z); a1.w = __fadd_rn(a1.w, b1.w);
        return tp_f4x2_to_h8(a0, a1);
    };
    i = my_hlo + tid;
    for (; i + 768 < my_hhi; i += 1024) {
        const uint4 p0 = add_u4_f32(tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]), tp_ld_cv_u4(m_in + (i - my_hlo)));
        const uint4 p1 = add_u4_f32(tp_f4x2_to_h8(local_x[2 * (i + 256)], local_x[2 * (i + 256) + 1]), tp_ld_cv_u4(m_in + (i + 256 - my_hlo)));
        const uint4 p2 = add_u4_f32(tp_f4x2_to_h8(local_x[2 * (i + 512)], local_x[2 * (i + 512) + 1]), tp_ld_cv_u4(m_in + (i + 512 - my_hlo)));
        const uint4 p3 = add_u4_f32(tp_f4x2_to_h8(local_x[2 * (i + 768)], local_x[2 * (i + 768) + 1]), tp_ld_cv_u4(m_in + (i + 768 - my_hlo)));
        h_loc[i] = p0; h_loc[i + 256] = p1; h_loc[i + 512] = p2; h_loc[i + 768] = p3;
        m_in[i - my_hlo] = p0; m_in[i + 256 - my_hlo] = p1; m_in[i + 512 - my_hlo] = p2; m_in[i + 768 - my_hlo] = p3;
    }
    for (; i < my_hhi; i += 256) {
        const uint4 pk = add_u4_f32(tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]), tp_ld_cv_u4(m_in + (i - my_hlo)));
        h_loc[i] = pk;
        m_in[i - my_hlo] = pk;
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) hflg(my_h_flg, lrank) = seq;
        __threadfence_system();
        while ((int) (tp_ld_sys_u32(hflg(rem_h_flg, lrank)) - seq) < 0) {}
        __threadfence_system();
    }
    __syncthreads();

    // Stage 3 (4x unrolled pipelined cross-socket QPI read of h_rem -> combine with HBM2 m_in -> BAR P2P push final!)
    uint4* h_rem = rem_h_sum + soff;
    uint4* p_fin = peer_p2p_fin + p_off;
    i = my_hlo + tid;
    for (; i + 768 < my_hhi; i += 1024) {
        const uint4 r0 = tp_ld_cv_u4(h_rem + i);
        const uint4 r1 = tp_ld_cv_u4(h_rem + i + 256);
        const uint4 r2 = tp_ld_cv_u4(h_rem + i + 512);
        const uint4 r3 = tp_ld_cv_u4(h_rem + i + 768);
        const uint4 pk0 = add_u4_f32(m_in[i - my_hlo], r0);
        const uint4 pk1 = add_u4_f32(m_in[i + 256 - my_hlo], r1);
        const uint4 pk2 = add_u4_f32(m_in[i + 512 - my_hlo], r2);
        const uint4 pk3 = add_u4_f32(m_in[i + 768 - my_hlo], r3);
        p_fin[i - my_hlo] = pk0;
        p_fin[i + 256 - my_hlo] = pk1;
        p_fin[i + 512 - my_hlo] = pk2;
        p_fin[i + 768 - my_hlo] = pk3;
        float4 a0, a1;
        tp_h8_to_f4x2(pk0, a0, a1); local_x[2 * i] = a0; local_x[2 * i + 1] = a1;
        tp_h8_to_f4x2(pk1, a0, a1); local_x[2 * (i + 256)] = a0; local_x[2 * (i + 256) + 1] = a1;
        tp_h8_to_f4x2(pk2, a0, a1); local_x[2 * (i + 512)] = a0; local_x[2 * (i + 512) + 1] = a1;
        tp_h8_to_f4x2(pk3, a0, a1); local_x[2 * (i + 768)] = a0; local_x[2 * (i + 768) + 1] = a1;
    }
    for (; i < my_hhi; i += 256) {
        const uint4 pk = add_u4_f32(m_in[i - my_hlo], tp_ld_cv_u4(h_rem + i));
        p_fin[i - my_hlo] = pk;
        float4 a0, a1;
        tp_h8_to_f4x2(pk, a0, a1);
        local_x[2 * i] = a0;
        local_x[2 * i + 1] = a1;
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) pflg(peer_p2p_flg, 1) = seq;
        __threadfence_system();
        while ((int) (tp_ld_sys_u32(pflg(my_p2p_flg, 1)) - seq) < 0) {}
        __threadfence_system();
    }
    __syncthreads();

    // Stage 4 (Local HBM2 unpack of peer's final half — 0 PCIe, 0 DRAM, 0 QPI!)
    uint4* m_fin = my_p2p_fin + p_off;
    for (int j = ot_hlo + tid; j < ot_hhi; j += 256) {
        float4 a0, a1;
        tp_h8_to_f4x2(tp_ld_cv_u4(m_fin + (j - ot_hlo)), a0, a1);
        local_x[2 * j] = a0;
        local_x[2 * j + 1] = a1;
    }
}

__device__ __forceinline__ uint4 tp_f16_to_bf16_u4(uint4 v) {
    auto cvt2 = [](uint32_t u) -> uint32_t {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&u));
        uint32_t a = __float_as_uint(f.x); a += 0x7fffu + ((a >> 16) & 1u);
        uint32_t b = __float_as_uint(f.y); b += 0x7fffu + ((b >> 16) & 1u);
        return (a >> 16) | (b & 0xffff0000u);
    };
    return make_uint4(cvt2(v.x), cvt2(v.y), cvt2(v.z), cvt2(v.w));
}

// 4-GPU Sequence-Parallel HC-read AllGather (Hybrid Intra-Socket P2P + Cross-Socket NUMA, 4x unrolled):
// Reuses d_p2p_in/fin, d_p2p_flg, h_sock_sum, h_sock_flg with strictly monotonic `seq` (ZERO extra VRAM!).
__global__ void __launch_bounds__(256) tp_ag_p2p_hc_f16_kernel(
    uint4* __restrict__ mixed_h, uint4* __restrict__ mixed_bf, uint4* __restrict__ inj4,
    uint4* __restrict__ my_p2p_in,  uint4* __restrict__ peer_p2p_in,
    uint4* __restrict__ my_p2p_fin, uint4* __restrict__ peer_p2p_fin,
    uint32_t* __restrict__ my_p2p_flg, uint32_t* __restrict__ peer_p2p_flg,
    uint4* __restrict__ my_h_sum, const uint4* __restrict__ rem_h_sum,
    uint32_t* __restrict__ my_h_flg, const uint32_t* __restrict__ rem_h_flg,
    int64_t stride8, int64_t half_stride8, int rank, uint32_t seq, int q8, int t_loc) {
    const int lrank = rank & 1;
    const int peer = rank ^ 1;
    const int rem_me = rank ^ 2;
    const int rem_peer = peer ^ 2;
    const int slot = (int) (seq & 1u);
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int nb = gridDim.x;
    auto pflg = [&](uint32_t* base, int ph) { return base + (((ph * 2 + slot) * kTpMaxBlocks) + bid) * 16; };
    auto hflg = [&](uint32_t* base, int lr) { return base + (((lr * 2 + slot) * kTpMaxBlocks) + bid) * 16; };
    if (seq == 0u) {
        if (tid == 0) {
            *(volatile uint32_t*) pflg(my_p2p_flg, 0) = 0u;
            *(volatile uint32_t*) pflg(my_p2p_flg, 1) = 0u;
            *(volatile uint32_t*) hflg(my_h_flg, lrank) = 0u;
            __threadfence_system();
        }
        return;
    }
    const size_t p_base = (size_t) slot * (size_t) half_stride8;
    const size_t half_h = (size_t) (stride8 >> 1);
    const size_t h_base = (size_t) slot * (size_t) stride8 + (size_t) lrank * half_h;
    const int per = (q8 + nb - 1) / nb;
    const int lo = bid * per;
    const int hi = min(q8, lo + per);

    // Phase 1: push own quarter to peer BAR + local NUMA RAM simultaneously!
    const uint4* mine = mixed_h + (size_t) rank * q8;
    uint4* p_in_peer = peer_p2p_in + p_base;
    uint4* h_loc = my_h_sum + h_base;
    int i = lo + tid;
    for (; i + 768 < hi; i += 1024) {
        const uint4 v0 = mine[i], v1 = mine[i + 256], v2 = mine[i + 512], v3 = mine[i + 768];
        p_in_peer[i] = v0; p_in_peer[i + 256] = v1; p_in_peer[i + 512] = v2; p_in_peer[i + 768] = v3;
        h_loc[i] = v0; h_loc[i + 256] = v1; h_loc[i + 512] = v2; h_loc[i + 768] = v3;
    }
    for (; i < hi; i += 256) { const uint4 v = mine[i]; p_in_peer[i] = v; h_loc[i] = v; }
    if (bid == 0 && inj4 != nullptr) {
        const uint4* mine_inj = inj4 + (size_t) rank * t_loc;
        for (int j = tid; j < t_loc; j += 256) {
            const uint4 v = mine_inj[j];
            p_in_peer[q8 + j] = v;
            h_loc[q8 + j] = v;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) pflg(peer_p2p_flg, 0) = seq;
        *(volatile uint32_t*) hflg(my_h_flg, lrank) = seq;
        __threadfence_system();
        while ((int) (tp_ld_sys_u32(pflg(my_p2p_flg, 0)) - seq) < 0) {}
        __threadfence_system();
    }
    __syncthreads();

    // Phase 2a: unpack intra-socket peer's quarter from local HBM2 (mixed_h + mixed_bf + inj)
    const uint4* m_in = my_p2p_in + p_base;
    uint4* dst_ph = mixed_h + (size_t) peer * q8;
    uint4* dst_pb = mixed_bf + (size_t) peer * q8;
    for (int j = lo + tid; j < hi; j += 256) {
        const uint4 v = tp_ld_cv_u4(m_in + j);
        dst_ph[j] = v;
        dst_pb[j] = tp_f16_to_bf16_u4(v);
    }
    if (bid == 0 && inj4 != nullptr) {
        uint4* dst_pi = inj4 + (size_t) peer * t_loc;
        for (int j = tid; j < t_loc; j += 256) dst_pi[j] = tp_ld_cv_u4(m_in + q8 + j);
    }

    // Phase 2b: wait for remote partner (rank^2) -> read ONCE over QPI, write local + P2P relay to peer!
    if (tid == 0) {
        while ((int) (tp_ld_sys_u32(hflg(const_cast<uint32_t*>(rem_h_flg), lrank)) - seq) < 0) {}
        __threadfence_system();
    }
    __syncthreads();
    const uint4* h_rem = rem_h_sum + h_base;
    uint4* p_fin_peer = peer_p2p_fin + p_base;
    uint4* dst_rh = mixed_h + (size_t) rem_me * q8;
    uint4* dst_rb = mixed_bf + (size_t) rem_me * q8;
    i = lo + tid;
    for (; i + 768 < hi; i += 1024) {
        const uint4 v0 = tp_ld_cv_u4(h_rem + i);
        const uint4 v1 = tp_ld_cv_u4(h_rem + i + 256);
        const uint4 v2 = tp_ld_cv_u4(h_rem + i + 512);
        const uint4 v3 = tp_ld_cv_u4(h_rem + i + 768);
        p_fin_peer[i] = v0; p_fin_peer[i + 256] = v1; p_fin_peer[i + 512] = v2; p_fin_peer[i + 768] = v3;
        dst_rh[i] = v0; dst_rh[i + 256] = v1; dst_rh[i + 512] = v2; dst_rh[i + 768] = v3;
        dst_rb[i] = tp_f16_to_bf16_u4(v0); dst_rb[i + 256] = tp_f16_to_bf16_u4(v1);
        dst_rb[i + 512] = tp_f16_to_bf16_u4(v2); dst_rb[i + 768] = tp_f16_to_bf16_u4(v3);
    }
    for (; i < hi; i += 256) {
        const uint4 v = tp_ld_cv_u4(h_rem + i);
        p_fin_peer[i] = v;
        dst_rh[i] = v;
        dst_rb[i] = tp_f16_to_bf16_u4(v);
    }
    if (bid == 0 && inj4 != nullptr) {
        uint4* dst_ri = inj4 + (size_t) rem_me * t_loc;
        for (int j = tid; j < t_loc; j += 256) {
            const uint4 v = tp_ld_cv_u4(h_rem + q8 + j);
            p_fin_peer[q8 + j] = v;
            dst_ri[j] = v;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) pflg(peer_p2p_flg, 1) = seq;
        __threadfence_system();
        while ((int) (tp_ld_sys_u32(pflg(my_p2p_flg, 1)) - seq) < 0) {}
        __threadfence_system();
    }
    __syncthreads();

    // Phase 3: unpack peer's relayed remote quarter from local HBM2 (0 PCIe, 0 QPI!)
    const uint4* m_fin = my_p2p_fin + p_base;
    uint4* dst_rph = mixed_h + (size_t) rem_peer * q8;
    uint4* dst_rpb = mixed_bf + (size_t) rem_peer * q8;
    for (int j = lo + tid; j < hi; j += 256) {
        const uint4 v = tp_ld_cv_u4(m_fin + j);
        dst_rph[j] = v;
        dst_rpb[j] = tp_f16_to_bf16_u4(v);
    }
    if (bid == 0 && inj4 != nullptr) {
        uint4* dst_rpi = inj4 + (size_t) rem_peer * t_loc;
        for (int j = tid; j < t_loc; j += 256) dst_rpi[j] = tp_ld_cv_u4(m_fin + q8 + j);
    }
}

// RAM-Only Hierarchical 2-Socket NUMA-Local AllReduce (fallback when P2P unavailable; still 1.29x faster than flat RS+AG!).
__global__ void __launch_bounds__(256) tp_ar_numa_hier_f16_u4_kernel(
    float4* __restrict__ local_x,
    uint4* __restrict__ my_r0, uint4* __restrict__ my_r1,
    uint4* __restrict__ my_s, uint4* __restrict__ rem_s, uint4* __restrict__ my_fin,
    uint32_t* __restrict__ my_f, uint32_t* __restrict__ rem_f,
    int64_t stride8, int rank, uint32_t seq, int n8) {
    const int lrank = rank & 1;
    const int slot = (int) (seq & 1u);
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int per = (n8 + gridDim.x - 1) / gridDim.x;
    const int lo = bid * per;
    const int hi = min(n8, lo + per);
    const int h_per = (hi - lo + 1) / 2;
    const int my_hlo = min(hi, lo + lrank * h_per);
    const int my_hhi = min(hi, my_hlo + h_per);
    const int ot_hlo = min(hi, lo + (1 - lrank) * h_per);
    const int ot_hhi = min(hi, ot_hlo + h_per);
    auto flg = [&](uint32_t* base, int ph, int lr) {
        return base + (((((size_t) ph * 2 + lr) * 2 + slot) * kTpMaxBlocks) + bid) * 16;
    };
    const size_t soff = (size_t) slot * (size_t) stride8;
    if (seq == 0u) {
        if (tid == 0) {
            *(volatile uint32_t*) flg(my_f, 0, lrank) = 0u;
            *(volatile uint32_t*) flg(my_f, 1, lrank) = 0u;
            *(volatile uint32_t*) flg(my_f, 2, lrank) = 0u;
            __threadfence_system();
        }
        return;
    }
    uint4* mine = (lrank == 0 ? my_r0 : my_r1) + soff;
    uint4* peer = (lrank == 0 ? my_r1 : my_r0) + soff;
    for (int i = lo + tid; i < hi; i += blockDim.x) mine[i] = tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]);
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) flg(my_f, 0, lrank) = seq;
        __threadfence_system();
        while ((int) (tp_ld_sys_u32(flg(my_f, 0, 1 - lrank)) - seq) < 0) {}
        __threadfence_system();
    }
    __syncthreads();
    uint4* s_loc = my_s + soff;
    for (int i = my_hlo + tid; i < my_hhi; i += blockDim.x) {
        float4 a0, a1, b0, b1;
        tp_h8_to_f4x2(tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]), a0, a1);
        tp_h8_to_f4x2(tp_ld_cv_u4(peer + i), b0, b1);
        a0.x = __fadd_rn(a0.x, b0.x); a0.y = __fadd_rn(a0.y, b0.y);
        a0.z = __fadd_rn(a0.z, b0.z); a0.w = __fadd_rn(a0.w, b0.w);
        a1.x = __fadd_rn(a1.x, b1.x); a1.y = __fadd_rn(a1.y, b1.y);
        a1.z = __fadd_rn(a1.z, b1.z); a1.w = __fadd_rn(a1.w, b1.w);
        s_loc[i] = tp_f4x2_to_h8(a0, a1);
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) flg(my_f, 1, lrank) = seq;
        __threadfence_system();
        while ((int) (tp_ld_sys_u32(flg(rem_f, 1, lrank)) - seq) < 0) {}
        __threadfence_system();
    }
    __syncthreads();
    uint4* s_rem = rem_s + soff;
    uint4* fin_loc = my_fin + soff;
    for (int i = my_hlo + tid; i < my_hhi; i += blockDim.x) {
        float4 a0, a1, b0, b1;
        tp_h8_to_f4x2(tp_ld_cv_u4(s_loc + i), a0, a1);
        tp_h8_to_f4x2(tp_ld_cv_u4(s_rem + i), b0, b1);
        a0.x = __fadd_rn(a0.x, b0.x); a0.y = __fadd_rn(a0.y, b0.y);
        a0.z = __fadd_rn(a0.z, b0.z); a0.w = __fadd_rn(a0.w, b0.w);
        a1.x = __fadd_rn(a1.x, b1.x); a1.y = __fadd_rn(a1.y, b1.y);
        a1.z = __fadd_rn(a1.z, b1.z); a1.w = __fadd_rn(a1.w, b1.w);
        const uint4 pk = tp_f4x2_to_h8(a0, a1);
        fin_loc[i] = pk;
        tp_h8_to_f4x2(pk, a0, a1);
        local_x[2 * i] = a0;
        local_x[2 * i + 1] = a1;
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) flg(my_f, 2, lrank) = seq;
        __threadfence_system();
        while ((int) (tp_ld_sys_u32(flg(my_f, 2, 1 - lrank)) - seq) < 0) {}
        __threadfence_system();
    }
    __syncthreads();
    for (int i = ot_hlo + tid; i < ot_hhi; i += blockDim.x) {
        float4 a0, a1;
        tp_h8_to_f4x2(tp_ld_cv_u4(fin_loc + i), a0, a1);
        local_x[2 * i] = a0;
        local_x[2 * i + 1] = a1;
    }
}

__global__ void tp_p2p_selftest_write_kernel(uint32_t* dst, uint32_t seed, int n) {
    const int i = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = seed ^ (uint32_t) (i * 2654435761u);
}

__global__ void tp_p2p_selftest_verify_kernel(const uint32_t* buf, uint32_t seed, int n, int* bad) {
    const int i = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const uint32_t exp = seed ^ (uint32_t) (i * 2654435761u);
        if (buf[i] != exp) atomicAdd(bad, 1);
    }
}

// 2-phase GPU-only uint4 Reduce-Scatter + All-Gather for large prefill tensors (ZERO host syncs, 1.5x less PCIe bytes!).
template <int NGPU>
__global__ void __launch_bounds__(256) tp_ar_host_rsag_f16_u4_kernel(
    float4* __restrict__ local_x,
    uint4* __restrict__ h_rows,
    uint4* __restrict__ h_sum,
    uint32_t* __restrict__ h_flags1,
    uint32_t* __restrict__ h_flags2,
    int64_t stride8,
    int rank,
    uint32_t seq,
    int n8) {
    const int slot = (int) (seq & 1u);
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int per = (n8 + gridDim.x - 1) / gridDim.x;
    const int lo = bid * per;
    const int hi = min(n8, lo + per);
    auto row = [&](int r) { return h_rows + ((size_t) r * 2 + (size_t) slot) * (size_t) stride8; };
    auto flg1 = [&](int r) { return h_flags1 + (((size_t) r * 2 + (size_t) slot) * kTpMaxBlocks + (size_t) bid) * 16; };
    auto flg2 = [&](int r) { return h_flags2 + (((size_t) r * 2 + (size_t) slot) * kTpMaxBlocks + (size_t) bid) * 16; };
    uint4* sum_slot = h_sum + (size_t) slot * (size_t) stride8;

    // Phase 1: write own slice in 16-byte uint4 bursts
    uint4* mine = row(rank);
    for (int i = lo + tid; i < hi; i += blockDim.x) {
        mine[i] = tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]);
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) flg1(rank) = seq;
        __threadfence_system();
        if (seq > 0u) {
#pragma unroll
            for (int r = 0; r < NGPU; ++r) {
                if (r == rank) continue;
                while ((int) (tp_ld_sys_u32(flg1(r)) - seq) < 0) {}
            }
            __threadfence_system();
        }
    }
    __syncthreads();
    if (seq == 0u) return;

    // Phase 2: reduce ONLY quarter `rank` of [lo, hi) across all peers, write FP16 result to sum_slot + own local_x
    const int q_per = (hi - lo + NGPU - 1) / NGPU;
    const int my_qlo = min(hi, lo + rank * q_per);
    const int my_qhi = min(hi, my_qlo + q_per);
    for (int i = my_qlo + tid; i < my_qhi; i += blockDim.x) {
        float4 acc0, acc1;
        if (rank == 0) tp_h8_to_f4x2(tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]), acc0, acc1);
        else tp_h8_to_f4x2(tp_ld_cv_u4(row(0) + i), acc0, acc1);
#pragma unroll
        for (int r = 1; r < NGPU; ++r) {
            float4 p0, p1;
            if (r == rank) tp_h8_to_f4x2(tp_f4x2_to_h8(local_x[2 * i], local_x[2 * i + 1]), p0, p1);
            else tp_h8_to_f4x2(tp_ld_cv_u4(row(r) + i), p0, p1);
            acc0.x = __fadd_rn(acc0.x, p0.x); acc0.y = __fadd_rn(acc0.y, p0.y);
            acc0.z = __fadd_rn(acc0.z, p0.z); acc0.w = __fadd_rn(acc0.w, p0.w);
            acc1.x = __fadd_rn(acc1.x, p1.x); acc1.y = __fadd_rn(acc1.y, p1.y);
            acc1.z = __fadd_rn(acc1.z, p1.z); acc1.w = __fadd_rn(acc1.w, p1.w);
        }
        const uint4 packed = tp_f4x2_to_h8(acc0, acc1);
        sum_slot[i] = packed;
        // Unpack packed so every rank produces bit-identical values!
        tp_h8_to_f4x2(packed, acc0, acc1);
        local_x[2 * i] = acc0;
        local_x[2 * i + 1] = acc1;
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) flg2(rank) = seq;
        __threadfence_system();
#pragma unroll
        for (int r = 0; r < NGPU; ++r) {
            if (r == rank) continue;
            while ((int) (tp_ld_sys_u32(flg2(r)) - seq) < 0) {}
        }
        __threadfence_system();
    }
    __syncthreads();

    // Phase 3: read ONLY the other NGPU-1 reduced quarters from sum_slot in 16-byte uint4 bursts!
#pragma unroll
    for (int r = 0; r < NGPU; ++r) {
        if (r == rank) continue;
        const int p_lo = min(hi, lo + r * q_per);
        const int p_hi = min(hi, p_lo + q_per);
        for (int i = p_lo + tid; i < p_hi; i += blockDim.x) {
            float4 a0, a1;
            tp_h8_to_f4x2(tp_ld_cv_u4(sum_slot + i), a0, a1);
            local_x[2 * i] = a0;
            local_x[2 * i + 1] = a1;
        }
    }
}

template <int NGPU>
__global__ void __launch_bounds__(256) tp_ar_host_ag_f16_kernel(
    float4* __restrict__ local_x,
    uint2* __restrict__ h_rows,
    uint32_t* __restrict__ h_flags,
    int64_t stride4,
    int rank,
    uint32_t seq,
    int n4) {
    const int slot = (int) (seq & 1u);
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int per = (n4 + gridDim.x - 1) / gridDim.x;
    const int lo = bid * per;
    const int hi = min(n4, lo + per);
    auto row = [&](int r) { return h_rows + ((size_t) r * 2 + (size_t) slot) * (size_t) stride4; };
    auto flg = [&](int r) { return h_flags + (((size_t) r * 2 + (size_t) slot) * kTpMaxBlocks + (size_t) bid) * 16; };

    uint2* mine = row(rank);
    for (int i = lo + tid; i < hi; i += blockDim.x) {
        mine[i] = tp_f4_to_h4(local_x[i]);
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) flg(rank) = seq;
        __threadfence_system();
        if (seq > 0u) {
#pragma unroll
            for (int r = 0; r < NGPU; ++r) {
                if (r == rank) continue;
                while ((int) (tp_ld_sys_u32(flg(r)) - seq) < 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                    __nanosleep(50);
#endif
                }
            }
            __threadfence_system();
        }
    }
    __syncthreads();
    if (seq == 0u) return;
    // Deterministic rank 0..NGPU-1 summation in FP32 (every rank produces bit-identical output!)
    for (int i = lo + tid; i < hi; i += blockDim.x) {
        float4 acc = (rank == 0) ? tp_h4_to_f4(tp_f4_to_h4(local_x[i])) : tp_h4_to_f4(tp_ld_cv_u2(row(0) + i));
#pragma unroll
        for (int r = 1; r < NGPU; ++r) {
            const float4 p = (r == rank) ? tp_h4_to_f4(tp_f4_to_h4(local_x[i])) : tp_h4_to_f4(tp_ld_cv_u2(row(r) + i));
            acc.x = __fadd_rn(acc.x, p.x);
            acc.y = __fadd_rn(acc.y, p.y);
            acc.z = __fadd_rn(acc.z, p.z);
            acc.w = __fadd_rn(acc.w, p.w);
        }
        local_x[i] = acc;
    }
}

__global__ void __launch_bounds__(256) tp_bcast_f32_kernel(
    float4* __restrict__ local_x,
    float4* __restrict__ h_buf,
    uint32_t* __restrict__ h_flags,
    int64_t stride4,
    int rank,
    uint32_t seq,
    int n4) {
    const int slot = (int) (seq & 1u);
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int per = (n4 + gridDim.x - 1) / gridDim.x;
    const int lo = bid * per;
    const int hi = min(n4, lo + per);
    float4* buf = h_buf + (size_t) slot * (size_t) stride4;
    uint32_t* flg = h_flags + ((size_t) slot * kTpMaxBlocks + (size_t) bid) * 16;
    if (rank == 0) {
        for (int i = lo + tid; i < hi; i += blockDim.x) buf[i] = local_x[i];
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            *(volatile uint32_t*) flg = seq;
            __threadfence_system();
        }
    } else {
        if (tid == 0 && seq > 0u) {
            while ((int) (tp_ld_sys_u32(flg) - seq) < 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                __nanosleep(50);
#endif
            }
            __threadfence_system();
        }
        __syncthreads();
        if (seq > 0u) {
            for (int i = lo + tid; i < hi; i += blockDim.x) local_x[i] = tp_ld_cv_f4(buf + i);
        }
    }
}

// Fused exact top-10 router + per-GPU EP=TP local VRAM plan + Rank-0 global miss detector (+ optional fused doorbell).
__global__ void __launch_bounds__(256) tp_ep_route_and_plan_kernel(
    const float* __restrict__ logits, int n_tok,
    int32_t* __restrict__ ids, float* __restrict__ weights,
    const int32_t* __restrict__ local_res,
    const int8_t* __restrict__ global_owner,
    const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
    int32_t* __restrict__ pl, long long capx, uint32_t* skip, uint32_t ring, int is_rank0,
    const float* __restrict__ mixed_x, int n_embd,
    float* __restrict__ h_x, int32_t* __restrict__ h_ids, float* __restrict__ h_w,
    uint32_t* __restrict__ h_seq, uint32_t* __restrict__ h_miss_l, uint32_t layer_idx) {
    __shared__ int32_t s_ids[128];
    __shared__ int32_t s_slot[128];
    __shared__ int32_t s_first[128];
    __shared__ int32_t s_gidx[128];
    __shared__ int32_t s_gstart[129];
    __shared__ int32_t s_rank[128];
    __shared__ int32_t s_cnt[128];
    __shared__ int32_t s_global_miss;

    const int tid = threadIdx.x;
    const int tok = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) s_global_miss = 0;
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
        maximum = tp_warp_max(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] = __expf(values[i] - maximum);
            sum += values[i];
        }
        const float reciprocal = 1.0f / tp_warp_sum(sum);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] *= reciprocal;
            if (__isnanf(values[i])) values[i] = -3.402823466e+38f;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        int local_gmiss = 0;
        for (int rk = 0; rk < 10; ++rk) {
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
                id_out[rk] = expert;
                const int eidx = tok * 10 + rk;
                const int32_t sl = local_res[expert];
                s_ids[eidx] = (sl >= 0) ? expert : -1;
                s_slot[eidx] = sl;
                if (is_rank0 && global_owner != nullptr && global_owner[expert] < 0) local_gmiss = 1;
                selected_sum += best;
            }
            if (rk == lane) selected = best;
        }
        if (is_rank0 && __any_sync(0xffffffffu, local_gmiss != 0) && lane == 0) atomicExch(&s_global_miss, 1);
        selected_sum = fmaxf(tp_warp_sum(selected_sum), 6.103515625e-5f);
        const float inv_sum = 1.0f / selected_sum;
        if (lane < 10) w_out[lane] = selected * inv_sum;
    }
    __syncthreads();

    // Group ONLY the experts resident on THIS GPU (s_ids[i] >= 0) using 4-warp parallel prefix scan (n <= 80 <= 128, ZERO tid==0 serial loop!)
    const int n = n_tok * 10;
    const int n_threads = (int) blockDim.x;
    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok_out = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    __shared__ int32_t s_wh[4];
    __shared__ int32_t s_wc[4];
    if (tid < 128) {
        const int32_t e = (tid < n) ? s_ids[tid] : -1;
        int first = tid, rk = 0, cnt = 0;
        if (e >= 0) {
            for (int j = 0; j < n; ++j) {
                if (s_ids[j] == e) {
                    ++cnt;
                    if (j < tid) {
                        if (first == tid) first = j;
                        ++rk;
                    }
                }
            }
        } else {
            first = -1;
        }
        const int is_head = (e >= 0 && first == tid) ? 1 : 0;
        const int my_cnt = is_head ? cnt : 0;
        s_first[tid] = first;
        s_rank[tid] = rk;
        int gh = is_head, gc = my_cnt;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int yh = __shfl_up_sync(0xffffffffu, gh, o);
            const int yc = __shfl_up_sync(0xffffffffu, gc, o);
            if (lane >= o) { gh += yh; gc += yc; }
        }
        if (lane == 31) {
            s_wh[tok] = gh;
            s_wc[tok] = gc;
        }
        __syncthreads();
        int bh = 0, bc = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < tok) { bh += s_wh[w]; bc += s_wc[w]; }
        }
        gh += bh;
        gc += bc;
        const int gi = gh - is_head;
        const int gs = gc - my_cnt;
        if (is_head) {
            s_gidx[tid] = gi;
            s_gstart[gi] = gs;
            start[gi] = gs;
        }
        if (tid == 127) {
            s_gstart[gh] = gc;
            start[gh] = gc;
            start2[0] = gc;
            counts[0] = gh;
            counts[1] = gc;
            counts[2] = 0;
            if (skip != nullptr) {
                *skip = (is_rank0 && s_global_miss != 0) ? 0u : ring;
            }
        }
    } else {
        __syncthreads();
    }
    __syncthreads();

    for (int i = tid; i < n; i += n_threads) {
        const int f = s_first[i];
        if (f < 0) continue;
        const int gi = s_gidx[f];
        if (f == i) {
            const int32_t slot = s_slot[i];
            ptr[gi] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        }
        const int at = s_gstart[gi] + s_rank[i];
        dst[at] = i;
        tok_out[at] = i / 10;
    }

    // Fused conditional Rank-0 doorbell publish (zero extra launch gap before waking CPU miss pool!)
    if (is_rank0 && s_global_miss != 0 && h_x != nullptr) {
        const int n4 = (n_tok * n_embd) >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(mixed_x);
        float4* hx4 = reinterpret_cast<float4*>(h_x);
        for (int i = tid; i < n4; i += n_threads) hx4[i] = x4[i];
        for (int i = tid; i < n; i += n_threads) {
            h_ids[i] = ids[i];
            h_w[i] = weights[i];
        }
        if (tid == 0) *(volatile uint32_t*) h_miss_l = layer_idx;
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            *(volatile uint32_t*) h_seq = *(volatile uint32_t*) h_seq + 1u;
            __threadfence_system();
        }
    }
}

template <int NGPU>
__global__ void __launch_bounds__(256) tp_moe_combine_miss_ar_f16_u4_kernel(
    const float* __restrict__ hit_out,
    const int32_t* __restrict__ p_dst,
    const int32_t* __restrict__ p_counts,
    const float* __restrict__ weights,
    const float* __restrict__ shared,
    float4* __restrict__ bo,
    const float4* __restrict__ h_ymiss_sum,
    const uint32_t* __restrict__ skip,
    const uint32_t* __restrict__ h_flag,
    uint32_t ring,
    int n_embd, int k,
    uint4* __restrict__ h_rows,
    uint32_t* __restrict__ h_flags,
    int64_t stride8,
    int rank,
    uint32_t seq,
    int n8) {
    const int slot = (int) (seq & 1u);
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int per = (n8 + gridDim.x - 1) / gridDim.x;
    const int lo = bid * per;
    const int hi = min(n8, lo + per);
    auto row = [&](int r) { return h_rows + ((size_t) r * 2 + (size_t) slot) * (size_t) stride8; };
    auto flg = [&](int r) { return h_flags + (((size_t) r * 2 + (size_t) slot) * kTpMaxBlocks + (size_t) bid) * 16; };

    if (seq == 0u) {
        uint4* mine = row(rank);
        for (int i = lo + tid; i < hi; i += blockDim.x) mine[i] = make_uint4(0u, 0u, 0u, 0u);
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            *(volatile uint32_t*) flg(rank) = 0u;
            __threadfence_system();
        }
        return;
    }

    __shared__ int32_t s_dst[80];
    __shared__ float s_w[80];
    __shared__ int32_t s_ne;
    __shared__ int32_t s_has_miss;
    if (tid == 0) {
        const int ne = p_counts[1];
        s_ne = (ne < 80) ? ne : 80;
        s_has_miss = (rank == 0 && skip != nullptr && *skip != ring) ? 1 : 0;
    }
    __syncthreads();
    const int ne = s_ne;
    if (tid < ne) {
        const int32_t sl = p_dst[tid];
        s_dst[tid] = sl;
        s_w[tid] = weights[sl];
    }
    if (s_has_miss && tid == 0) {
        while ((int) (tp_ld_sys_u32(h_flag) - ring) < 0) {}
        __threadfence_system();
    }
    __syncthreads();

    const int n_embd8 = n_embd >> 3;
    const int has_miss = s_has_miss;
    uint4* mine = row(rank);
    // Keep packed own-rank FP16 in registers (max 3 iters for T<=8 -> STACK:0!)
    uint4 s0 = make_uint4(0u, 0u, 0u, 0u), s1 = s0, s2 = s0;
    int scnt = 0;
    for (int i = lo + tid; i < hi; i += blockDim.x, ++scnt) {
        const int t = i / n_embd8;
        const int d8 = i - t * n_embd8;
        const int sl_lo = t * k;
        const int sl_hi = sl_lo + k;
        const float4* sh4 = reinterpret_cast<const float4*>(shared + (size_t) t * n_embd) + d8 * 2;
        float4 a0 = sh4[0];
        float4 a1 = sh4[1];
        for (int m = 0; m < ne; ++m) {
            const int sl = s_dst[m];
            if (sl >= sl_lo && sl < sl_hi) {
                const float wm = s_w[m];
                const float4* h4 = reinterpret_cast<const float4*>(hit_out + (size_t) sl * n_embd) + d8 * 2;
                const float4 v0 = h4[0];
                const float4 v1 = h4[1];
                a0.x = fmaf(wm, v0.x, a0.x); a0.y = fmaf(wm, v0.y, a0.y);
                a0.z = fmaf(wm, v0.z, a0.z); a0.w = fmaf(wm, v0.w, a0.w);
                a1.x = fmaf(wm, v1.x, a1.x); a1.y = fmaf(wm, v1.y, a1.y);
                a1.z = fmaf(wm, v1.z, a1.z); a1.w = fmaf(wm, v1.w, a1.w);
            }
        }
        if (has_miss) {
            const float4 m0 = tp_ld_cv_f4(h_ymiss_sum + 2 * i);
            const float4 m1 = tp_ld_cv_f4(h_ymiss_sum + 2 * i + 1);
            a0.x += m0.x; a0.y += m0.y; a0.z += m0.z; a0.w += m0.w;
            a1.x += m1.x; a1.y += m1.y; a1.z += m1.z; a1.w += m1.w;
        }
        const uint4 packed = tp_f4x2_to_h8(a0, a1);
        mine[i] = packed;
        if (scnt == 0) s0 = packed;
        else if (scnt == 1) s1 = packed;
        else if (scnt == 2) s2 = packed;
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) flg(rank) = seq;
        __threadfence_system();
#pragma unroll
        for (int r = 0; r < NGPU; ++r) {
            if (r == rank) continue;
            while ((int) (tp_ld_sys_u32(flg(r)) - seq) < 0) {}
        }
        __threadfence_system();
    }
    __syncthreads();
    int idx_s = 0;
    for (int i = lo + tid; i < hi; i += blockDim.x, ++idx_s) {
        const uint4 my_u = (idx_s == 0) ? s0 : (idx_s == 1 ? s1 : (idx_s == 2 ? s2 : tp_ld_cv_u4(mine + i)));
        float4 acc0, acc1;
        tp_h8_to_f4x2((rank == 0) ? my_u : tp_ld_cv_u4(row(0) + i), acc0, acc1);
#pragma unroll
        for (int r = 1; r < NGPU; ++r) {
            float4 p0, p1;
            tp_h8_to_f4x2((r == rank) ? my_u : tp_ld_cv_u4(row(r) + i), p0, p1);
            acc0.x = __fadd_rn(acc0.x, p0.x); acc0.y = __fadd_rn(acc0.y, p0.y);
            acc0.z = __fadd_rn(acc0.z, p0.z); acc0.w = __fadd_rn(acc0.w, p0.w);
            acc1.x = __fadd_rn(acc1.x, p1.x); acc1.y = __fadd_rn(acc1.y, p1.y);
            acc1.z = __fadd_rn(acc1.z, p1.z); acc1.w = __fadd_rn(acc1.w, p1.w);
        }
        bo[2 * i] = acc0;
        bo[2 * i + 1] = acc1;
    }
}

// NUMA-Split 4-rank Fused MoE combine + Rank-0 miss wait/add + FP16 AR (Single-TX Pure-RAM DDIO default, STACK:0!)
template <bool USE_P2P>
__global__ void __launch_bounds__(256, 2) tp_moe_combine_miss_ar_f16_u4_split4_kernel(
    const float* __restrict__ hit_out,
    const int32_t* __restrict__ p_dst,
    const int32_t* __restrict__ p_counts,
    const float* __restrict__ weights,
    const float* __restrict__ shared,
    float4* __restrict__ bo,
    const float4* __restrict__ h_ymiss_sum,
    const uint32_t* __restrict__ skip,
    const uint32_t* __restrict__ h_flag,
    uint32_t ring,
    int n_embd, int k,
    uint4* __restrict__ r0, uint4* __restrict__ r1, uint4* __restrict__ r2, uint4* __restrict__ r3,
    uint32_t* __restrict__ f0, uint32_t* __restrict__ f1, uint32_t* __restrict__ f2, uint32_t* __restrict__ f3,
    uint4* __restrict__ p2p_me, uint4* __restrict__ p2p_peer, int64_t p2p_stride8,
    int64_t stride8, int rank, uint32_t seq, int n8) {
    auto row_ptr = [&](int r) -> uint4* { return r == 0 ? r0 : (r == 1 ? r1 : (r == 2 ? r2 : r3)); };
    auto flg_ptr = [&](int r) -> uint32_t* { return r == 0 ? f0 : (r == 1 ? f1 : (r == 2 ? f2 : f3)); };
    const int slot = (int) (seq & 1u);
    const int peer = rank ^ 1;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int per = (n8 + gridDim.x - 1) / gridDim.x;
    const int lo = bid * per;
    const int hi = min(n8, lo + per);
    const size_t soff = (size_t) slot * (size_t) stride8;
    const size_t p_soff = (size_t) slot * (size_t) p2p_stride8;
    const size_t foff = ((size_t) slot * kTpMaxBlocks + (size_t) bid) * 16;

    if (seq == 0u) {
        uint4* mine = row_ptr(rank) + soff;
        for (int i = lo + tid; i < hi; i += blockDim.x) mine[i] = make_uint4(0u, 0u, 0u, 0u);
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            *(volatile uint32_t*) (flg_ptr(rank) + foff) = 0u;
            __threadfence_system();
        }
        return;
    }

    __shared__ int32_t s_tok_cnt[8];
    __shared__ int32_t s_tok_sl[8][10];
    __shared__ float s_tok_w[8][10];
    __shared__ int32_t s_ne;
    __shared__ int32_t s_has_miss;
    if (tid == 0) {
        const int ne = p_counts[1];
        s_ne = (ne < 80) ? ne : 80;
        s_has_miss = (rank == 0 && skip != nullptr && *skip != ring) ? 1 : 0;
    }
    if (tid < 8) s_tok_cnt[tid] = 0;
    __syncthreads();
    const int ne = s_ne;
    // Pre-bucket local hit entries by token t in SMEM so each thread iterates only 2-3 exact hits (instead of 20-80!)
    if (tid < ne) {
        const int32_t sl = p_dst[tid];
        const int t = sl / k;
        if (t < 8) {
            const int pos = atomicAdd(&s_tok_cnt[t], 1);
            if (pos < 10) {
                s_tok_sl[t][pos] = sl;
                s_tok_w[t][pos] = weights[sl];
            }
        }
    }
    __syncthreads();

    const int n_embd8 = n_embd >> 3;
    const int has_miss = s_has_miss;
    uint4* mine = row_ptr(rank) + soff;
    uint4* pp = (USE_P2P && p2p_peer != nullptr) ? (p2p_peer + p_soff) : nullptr;
    uint4 s0 = make_uint4(0u, 0u, 0u, 0u), s1 = s0, s2 = s0, s3 = s0;
    int scnt = 0;
    // Overlap Rank 0's local shared+hit_out combine WITH CPU miss compute before spinning on h_flag!
    for (int i = lo + tid; i < hi; i += blockDim.x, ++scnt) {
        const int t = i / n_embd8;
        const int d8 = i - t * n_embd8;
        const float4* sh4 = reinterpret_cast<const float4*>(shared + (size_t) t * n_embd) + d8 * 2;
        float4 a0 = sh4[0];
        float4 a1 = sh4[1];
        const int tc = s_tok_cnt[t];
        for (int m = 0; m < tc; ++m) {
            const int sl = s_tok_sl[t][m];
            const float wm = s_tok_w[t][m];
            const float4* h4 = reinterpret_cast<const float4*>(hit_out + (size_t) sl * n_embd) + d8 * 2;
            const float4 v0 = h4[0];
            const float4 v1 = h4[1];
            a0.x = fmaf(wm, v0.x, a0.x); a0.y = fmaf(wm, v0.y, a0.y);
            a0.z = fmaf(wm, v0.z, a0.z); a0.w = fmaf(wm, v0.w, a0.w);
            a1.x = fmaf(wm, v1.x, a1.x); a1.y = fmaf(wm, v1.y, a1.y);
            a1.z = fmaf(wm, v1.z, a1.z); a1.w = fmaf(wm, v1.w, a1.w);
        }
        if (!has_miss) {
            const uint4 packed = tp_f4x2_to_h8(a0, a1);
            if constexpr (USE_P2P) {
                if (pp != nullptr) pp[i] = packed;
            }
            mine[i] = packed;
            if (scnt == 0) s0 = packed;
            else if (scnt == 1) s1 = packed;
            else if (scnt == 2) s2 = packed;
            else if (scnt == 3) s3 = packed;
        } else {
            bo[2 * i] = a0;
            bo[2 * i + 1] = a1;
        }
    }
    if (has_miss) {
        __syncthreads();
        if (tid == 0) {
            while ((int) (tp_ld_sys_u32(h_flag) - ring) < 0) {}
            __threadfence_system();
        }
        __syncthreads();
        scnt = 0;
        for (int i = lo + tid; i < hi; i += blockDim.x, ++scnt) {
            float4 a0 = bo[2 * i];
            float4 a1 = bo[2 * i + 1];
            const float4 m0 = tp_ld_cv_f4(h_ymiss_sum + 2 * i);
            const float4 m1 = tp_ld_cv_f4(h_ymiss_sum + 2 * i + 1);
            a0.x += m0.x; a0.y += m0.y; a0.z += m0.z; a0.w += m0.w;
            a1.x += m1.x; a1.y += m1.y; a1.z += m1.z; a1.w += m1.w;
            const uint4 packed = tp_f4x2_to_h8(a0, a1);
            if constexpr (USE_P2P) {
                if (pp != nullptr) pp[i] = packed;
            }
            mine[i] = packed;
            if (scnt == 0) s0 = packed;
            else if (scnt == 1) s1 = packed;
            else if (scnt == 2) s2 = packed;
            else if (scnt == 3) s3 = packed;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) (flg_ptr(rank) + foff) = seq;
        __threadfence_system();
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (r == rank) continue;
            while ((int) (tp_ld_sys_u32(flg_ptr(r) + foff) - seq) < 0) {}
        }
        __threadfence_system();
    }
    __syncthreads();
    const uint4* pm = (USE_P2P && p2p_me != nullptr) ? (p2p_me + p_soff) : nullptr;
    int idx_s = 0;
    for (int i = lo + tid; i < hi; i += blockDim.x, ++idx_s) {
        const uint4 my_u = (idx_s == 0) ? s0 : (idx_s == 1 ? s1 : (idx_s == 2 ? s2 : (idx_s == 3 ? s3 : tp_ld_cv_u4(mine + i))));
        float4 acc0, acc1;
        if (rank == 0) tp_h8_to_f4x2(my_u, acc0, acc1);
        else if (USE_P2P && 0 == peer && pm != nullptr) tp_h8_to_f4x2(tp_ld_cv_u4(pm + i), acc0, acc1);
        else tp_h8_to_f4x2(tp_ld_cv_u4(r0 + soff + i), acc0, acc1);
#pragma unroll
        for (int r = 1; r < 4; ++r) {
            float4 p0, p1;
            if (r == rank) tp_h8_to_f4x2(my_u, p0, p1);
            else if (USE_P2P && r == peer && pm != nullptr) tp_h8_to_f4x2(tp_ld_cv_u4(pm + i), p0, p1);
            else tp_h8_to_f4x2(tp_ld_cv_u4(row_ptr(r) + soff + i), p0, p1);
            acc0.x = __fadd_rn(acc0.x, p0.x); acc0.y = __fadd_rn(acc0.y, p0.y);
            acc0.z = __fadd_rn(acc0.z, p0.z); acc0.w = __fadd_rn(acc0.w, p0.w);
            acc1.x = __fadd_rn(acc1.x, p1.x); acc1.y = __fadd_rn(acc1.y, p1.y);
            acc1.z = __fadd_rn(acc1.z, p1.z); acc1.w = __fadd_rn(acc1.w, p1.w);
        }
        bo[2 * i] = acc0;
        bo[2 * i + 1] = acc1;
    }
}

__global__ void __launch_bounds__(256) tp_moe_combine_partial_kernel(
    const float* __restrict__ hit_out,
    const int32_t* __restrict__ p_dst,
    const int32_t* __restrict__ p_counts,
    const float* __restrict__ weights,
    const float* __restrict__ shared,
    float* __restrict__ bo,
    int n_embd, int k) {
    const int t = (int) blockIdx.y;
    const int d = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;

    const int n_entries = p_counts[1];
    const int slot_lo = t * k;
    const int slot_hi = slot_lo + k;
    float acc = shared[(size_t) t * n_embd + d];
    for (int m = 0; m < n_entries; ++m) {
        const int slot = p_dst[m];
        if (slot >= slot_lo && slot < slot_hi) {
            acc = fmaf(weights[slot], hit_out[(size_t) slot * n_embd + d], acc);
        }
    }
    bo[(size_t) t * n_embd + d] = acc;
}

__global__ void __launch_bounds__(256) tp_add_miss_sum_unless_kernel(
    float4* __restrict__ bo,
    const float4* __restrict__ m_ymiss_sum,
    const uint32_t* __restrict__ skip,
    uint32_t ring,
    int n4) {
    if (*skip == ring) return;
    for (int i = (int) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int) gridDim.x * blockDim.x) {
        const float4 a = bo[i];
        const float4 b = tp_ld_cv_f4(m_ymiss_sum + i);
        bo[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

__global__ void __launch_bounds__(256) tp_lmhead_local_argmax_kernel(
    const float* __restrict__ logits, int n_local, int row_offset,
    const int32_t* __restrict__ skip_if_pos,
    float* __restrict__ h_argmax_v, int32_t* __restrict__ h_argmax_i,
    int rank, int n_tok) {
    if (skip_if_pos != nullptr && *(const volatile int32_t*) skip_if_pos > 0) return;
    __shared__ float sm[8];
    __shared__ int32_t si[8];
    const int t = (int) blockIdx.x;
    if (t >= n_tok) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* row = logits + (size_t) t * n_local;
    float best_v = -FLT_MAX;
    int32_t best_i = 0;
    for (int i = tid; i < n_local; i += 256) {
        const float v = __ldg(row + i);
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, m, 32);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, m, 32);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) { sm[warp] = best_v; si[warp] = best_i; }
    __syncthreads();
    if (warp == 0) {
        float bv = (lane < 8) ? sm[lane] : -FLT_MAX;
        int32_t bi = (lane < 8) ? si[lane] : 0;
#pragma unroll
        for (int m = 4; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, m, 32);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, m, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            h_argmax_v[(size_t) rank * kTpMaxWin + t] = bv;
            h_argmax_i[(size_t) rank * kTpMaxWin + t] = bi + row_offset;
        }
    }
}

__global__ void tp_lmhead_merge_argmax_kernel(
    const float* __restrict__ h_argmax_v,
    const int32_t* __restrict__ h_argmax_i,
    uint32_t* __restrict__ h_flags,
    int tp, int rank, uint32_t seq, int n_tok,
    int32_t* __restrict__ out_ids) {
    if (threadIdx.x != 0) return;
    __threadfence_system();
    *(volatile uint32_t*) (h_flags + (size_t) rank * 16) = seq;
    __threadfence_system();
    if (rank != 0 || seq == 0u) return;
    for (int r = 1; r < tp; ++r) {
        while ((int) (tp_ld_sys_u32(h_flags + (size_t) r * 16) - seq) < 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(50);
#endif
        }
    }
    __threadfence_system();
    for (int t = 0; t < n_tok; ++t) {
        float bv = ((const volatile float*) h_argmax_v)[t];
        int32_t bi = ((const volatile int32_t*) h_argmax_i)[t];
        for (int r = 1; r < tp; ++r) {
            const float ov = ((const volatile float*) h_argmax_v)[(size_t) r * kTpMaxWin + t];
            const int32_t oi = ((const volatile int32_t*) h_argmax_i)[(size_t) r * kTpMaxWin + t];
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        out_ids[t] = bi;
    }
    __threadfence_system();
}

static constexpr int kTopkStageBlocks = 32;

__global__ void __launch_bounds__(256) tp_f32_to_f16_bulk_kernel(
    const float4* __restrict__ src, uint2* __restrict__ dst, int n4) {
    const int i = (int) blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = (int) gridDim.x * blockDim.x;
    for (int j = i; j < n4; j += stride) {
        dst[j] = tp_f4_to_h4(src[j]);
    }
}

__global__ void __launch_bounds__(256) tp_f16_to_f32_bulk_kernel(
    const uint2* __restrict__ src, float4* __restrict__ dst, int n4) {
    const int i = (int) blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = (int) gridDim.x * blockDim.x;
    for (int j = i; j < n4; j += stride) {
        dst[j] = tp_h4_to_f4(src[j]);
    }
}

// Stage 1: 32 blocks per token, 1 coalesced global pass into registers (<=16 elems/thread for n_local <= 131072),
// extracting top-k_eff per block with exact tie-breaking. When `k_ptr != nullptr`, reads dynamic k (0 = instant skip).
__global__ void __launch_bounds__(256) tp_lmhead_topk_stage1_kernel(
    const float* __restrict__ logits, int n_local, int k_fixed, const int32_t* __restrict__ k_ptr,
    float* __restrict__ stage_v, int32_t* __restrict__ stage_i) {
    const int k_eff = (k_ptr != nullptr) ? *(const volatile int32_t*) k_ptr : k_fixed;
    if (k_eff <= 0) return;
    __shared__ float sm[8];
    __shared__ int32_t si[8];
    const int bid = (int) blockIdx.x;
    const int t = (int) blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int per = (n_local + kTopkStageBlocks - 1) / kTopkStageBlocks;
    const int lo = bid * per;
    const int hi = min(n_local, lo + per);
    const float* row = logits + (size_t) t * n_local;
    float* ov_out = stage_v + ((size_t) t * kTopkStageBlocks + (size_t) bid) * (size_t) k_eff;
    int32_t* oi_out = stage_i + ((size_t) t * kTopkStageBlocks + (size_t) bid) * (size_t) k_eff;

    // Fast register-cached path when each thread holds <= 16 elements (always true for TP=2/4 on vocab <= 262K).
    // Constant-indexed #pragma unroll keeps rv[]/ri[] strictly in registers (STACK:0 vs STACK:128!).
    if (hi - lo <= 256 * 16) {
        float rv[16];
        int32_t ri[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int idx = lo + tid + j * 256;
            rv[j] = (idx < hi) ? __ldg(row + idx) : -FLT_MAX;
            ri[j] = (idx < hi) ? idx : -1;
        }
        for (int rk = 0; rk < k_eff; ++rk) {
            float best_v = -FLT_MAX;
            int32_t best_i = 0;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float v = rv[j];
                const int32_t ii = ri[j];
                if (v > best_v || (v == best_v && ii < best_i)) {
                    best_v = v;
                    best_i = ii;
                }
            }
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, best_v, m, 32);
                const int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, m, 32);
                if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
            }
            if (lane == 0) { sm[warp] = best_v; si[warp] = best_i; }
            __syncthreads();
            if (warp == 0) {
                float bv = (lane < 8) ? sm[lane] : -FLT_MAX;
                int32_t bi = (lane < 8) ? si[lane] : 0;
#pragma unroll
                for (int m = 4; m > 0; m >>= 1) {
                    const float ov = __shfl_xor_sync(0xffffffffu, bv, m, 32);
                    const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, m, 32);
                    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                }
                if (lane == 0) {
                    sm[0] = bv;
                    si[0] = bi;
                    ov_out[rk] = bv;
                    oi_out[rk] = bi;
                }
            }
            __syncthreads();
            const int32_t win_i = si[0];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                if (ri[j] == win_i) rv[j] = -FLT_MAX;
            }
        }
        return;
    }

    // Fallback for huge vocabularies
    float prev_v = FLT_MAX;
    int32_t prev_i = -1;
    for (int rk = 0; rk < k_eff; ++rk) {
        float best_v = -FLT_MAX;
        int32_t best_i = 0;
        for (int idx = lo + tid; idx < hi; idx += 256) {
            const float v = __ldg(row + idx);
            if (v < prev_v || (v == prev_v && idx > prev_i)) {
                if (v > best_v || (v == best_v && idx < best_i)) { best_v = v; best_i = idx; }
            }
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best_v, m, 32);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, m, 32);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) { sm[warp] = best_v; si[warp] = best_i; }
        __syncthreads();
        if (warp == 0) {
            float bv = (lane < 8) ? sm[lane] : -FLT_MAX;
            int32_t bi = (lane < 8) ? si[lane] : 0;
#pragma unroll
            for (int m = 4; m > 0; m >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, m, 32);
                const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, m, 32);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { sm[0] = bv; si[0] = bi; ov_out[rk] = bv; oi_out[rk] = bi; }
        }
        __syncthreads();
        prev_v = sm[0];
        prev_i = si[0];
    }
}

// Stage 2: merge 32 * k_eff (<= 2048 -> <= 8 elems/thread in registers!) into final top-k_eff in mapped host RAM.
__global__ void __launch_bounds__(256) tp_lmhead_topk_stage2_kernel(
    const float* __restrict__ stage_v, const int32_t* __restrict__ stage_i,
    int row_offset, int k_fixed, const int32_t* __restrict__ k_ptr,
    float* __restrict__ h_topk_v, int32_t* __restrict__ h_topk_i,
    int tp, int rank) {
    const int k_eff = (k_ptr != nullptr) ? *(const volatile int32_t*) k_ptr : k_fixed;
    if (k_eff <= 0) return;
    __shared__ float sm[8];
    __shared__ int32_t si[8];
    const int t = (int) blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n_cand = kTopkStageBlocks * k_eff;
    const float* sv = stage_v + (size_t) t * (size_t) n_cand;
    const int32_t* si_in = stage_i + (size_t) t * (size_t) n_cand;
    float* out_v = h_topk_v + ((size_t) t * tp + (size_t) rank) * (size_t) k_eff;
    int32_t* out_i = h_topk_i + ((size_t) t * tp + (size_t) rank) * (size_t) k_eff;

    // Constant-indexed #pragma unroll keeps rv[]/ri[] strictly in registers (STACK:0 vs STACK:64!).
    float rv[8];
    int32_t ri[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int idx = tid + j * 256;
        rv[j] = (idx < n_cand) ? sv[idx] : -FLT_MAX;
        ri[j] = (idx < n_cand) ? si_in[idx] : -1;
    }
    for (int rk = 0; rk < k_eff; ++rk) {
        float best_v = -FLT_MAX;
        int32_t best_i = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float v = rv[j];
            const int32_t ii = ri[j];
            if (v > best_v || (v == best_v && ii < best_i)) {
                best_v = v;
                best_i = ii;
            }
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best_v, m, 32);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, m, 32);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) { sm[warp] = best_v; si[warp] = best_i; }
        __syncthreads();
        if (warp == 0) {
            float bv = (lane < 8) ? sm[lane] : -FLT_MAX;
            int32_t bi = (lane < 8) ? si[lane] : 0;
#pragma unroll
            for (int m = 4; m > 0; m >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, m, 32);
                const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, m, 32);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) {
                sm[0] = bv;
                si[0] = bi;
                out_v[rk] = bv;
                out_i[rk] = bi + row_offset;
            }
        }
        __syncthreads();
        const int32_t win_i = si[0];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (ri[j] == win_i) rv[j] = -FLT_MAX;
        }
    }
}

__global__ void __launch_bounds__(256) tp_lmhead_scatter_logits_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    int n_local4, int row_off4, int vocab4) {
    const int t = (int) blockIdx.y;
    const int i = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local4) return;
    dst[(size_t) t * vocab4 + row_off4 + i] = src[(size_t) t * n_local4 + i];
}

void check_tp(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "tp_allreduce %s: %s\n", what, cudaGetErrorString(e));
    }
}

int detect_gpu_numa_node(int dev) {
#if defined(__linux__)
    char bus[32] = {};
    if (cudaDeviceGetPCIBusId(bus, (int) sizeof(bus), dev) == cudaSuccess) {
        for (char* p = bus; *p; ++p) if (*p >= 'A' && *p <= 'Z') *p = (char) (*p - 'A' + 'a');
        char npath[96];
        std::snprintf(npath, sizeof(npath), "/sys/bus/pci/devices/%s/numa_node", bus);
        if (std::FILE* nf = std::fopen(npath, "r")) {
            int nd = 0;
            if (std::fscanf(nf, "%d", &nd) == 1 && nd >= 0) {
                std::fclose(nf);
                return nd;
            }
            std::fclose(nf);
        }
    }
#else
    (void) dev;
#endif
    return 0;
}

void* alloc_numa_mapped_portable(size_t bytes, int node) {
#if defined(__linux__) && defined(SYS_mbind)
    const size_t align = bytes >= 4096 ? 4096 : 64;
    const size_t alloc_b = (bytes + align - 1) & ~(align - 1);
    void* p = nullptr;
    if (posix_memalign(&p, align, alloc_b) != 0 || p == nullptr) return nullptr;
    unsigned long mask = 1ul << (unsigned) node;
    (void) ::syscall(SYS_mbind, p, alloc_b, MPOL_BIND, &mask, 64ul, MPOL_MF_MOVE);
    std::memset(p, 0, alloc_b);
    if (cudaHostRegister(p, alloc_b, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        std::free(p);
        return nullptr;
    }
    return p;
#else
    (void) node;
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return nullptr;
    std::memset(p, 0, bytes);
    return p;
#endif
}

void free_numa_mapped_portable(void* p) {
    if (!p) return;
#if defined(__linux__) && defined(SYS_mbind)
    cudaHostUnregister(p);
    std::free(p);
#else
    cudaFreeHost(p);
#endif
}

bool verify_p2p_pair_bitexact(int a, int b) {
    int can_ab = 0, can_ba = 0;
    if (cudaDeviceCanAccessPeer(&can_ab, a, b) != cudaSuccess || !can_ab) return false;
    if (cudaDeviceCanAccessPeer(&can_ba, b, a) != cudaSuccess || !can_ba) return false;
    cudaSetDevice(a);
    cudaError_t e1 = cudaDeviceEnablePeerAccess(b, 0);
    if (e1 != cudaSuccess && e1 != cudaErrorPeerAccessAlreadyEnabled) { (void) cudaGetLastError(); return false; }
    if (e1 == cudaErrorPeerAccessAlreadyEnabled) (void) cudaGetLastError();
    cudaSetDevice(b);
    cudaError_t e2 = cudaDeviceEnablePeerAccess(a, 0);
    if (e2 != cudaSuccess && e2 != cudaErrorPeerAccessAlreadyEnabled) { (void) cudaGetLastError(); return false; }
    if (e2 == cudaErrorPeerAccessAlreadyEnabled) (void) cudaGetLastError();

    // Strict 4 MiB bidirectional direct BAR P2P write + verify self-test (catches IOMMU DMA-FQ / ACS drops!)
    constexpr int kN = 1024 * 1024;
    uint32_t *da = nullptr, *db = nullptr;
    int *bad_a = nullptr, *bad_b = nullptr;
    cudaSetDevice(a);
    bool ok = cudaMalloc(&da, (size_t) kN * 4) == cudaSuccess && cudaMalloc(&bad_a, 4) == cudaSuccess;
    cudaSetDevice(b);
    ok = ok && cudaMalloc(&db, (size_t) kN * 4) == cudaSuccess && cudaMalloc(&bad_b, 4) == cudaSuccess;
    if (ok) {
        cudaSetDevice(a); cudaMemset(da, 0, (size_t) kN * 4); cudaMemset(bad_a, 0, 4); cudaDeviceSynchronize();
        cudaSetDevice(b); cudaMemset(db, 0, (size_t) kN * 4); cudaMemset(bad_b, 0, 4); cudaDeviceSynchronize();
        cudaSetDevice(a);
        tp_p2p_selftest_write_kernel<<<(kN + 255) / 256, 256>>>(db, 0xCAFEBABEu, kN);
        cudaDeviceSynchronize();
        cudaSetDevice(b);
        tp_p2p_selftest_verify_kernel<<<(kN + 255) / 256, 256>>>(db, 0xCAFEBABEu, kN, bad_b);
        int hb = -1;
        cudaMemcpy(&hb, bad_b, 4, cudaMemcpyDeviceToHost);
        tp_p2p_selftest_write_kernel<<<(kN + 255) / 256, 256>>>(da, 0x12345678u, kN);
        cudaDeviceSynchronize();
        cudaSetDevice(a);
        tp_p2p_selftest_verify_kernel<<<(kN + 255) / 256, 256>>>(da, 0x12345678u, kN, bad_a);
        int ha = -1;
        cudaMemcpy(&ha, bad_a, 4, cudaMemcpyDeviceToHost);
        ok = (ha == 0 && hb == 0 && cudaGetLastError() == cudaSuccess);
    }
    cudaSetDevice(a); if (da) cudaFree(da); if (bad_a) cudaFree(bad_a);
    cudaSetDevice(b); if (db) cudaFree(db); if (bad_b) cudaFree(bad_b);
    (void) cudaGetLastError();
    return ok;
}

}  // namespace

bool tp_comm_init(TpComm& c, int tp, const int* devs, int64_t max_ar_floats, int64_t max_bcast_floats,
                  int64_t vocab_total) {
    tp_comm_free(c);
    if (tp != 2 && tp != 4) return false;
    c.tp = tp;
    for (int r = 0; r < tp; ++r) {
        c.devs[r] = devs[r];
        c.dev_numa[r] = detect_gpu_numa_node(devs[r]);
    }
    c.ar_stride4 = ((max_ar_floats + 7) / 8) * 2;  // even float4 count -> aligned for uint4 (8 FP16s)
    c.bcast_cap = (max_bcast_floats + 3) & ~3ll;
    c.vocab_total = vocab_total;
    const int64_t stride8 = c.ar_stride4 / 2;
    c.p2p_half_stride8 = (stride8 / 2) + kTpMaxBlocks;

    const unsigned flags = cudaHostAllocMapped | cudaHostAllocPortable;
    bool ok = cudaHostAlloc(&c.h_ar_rows, (size_t) tp * 2 * (size_t) c.ar_stride4 * sizeof(uint2), flags) == cudaSuccess &&
              cudaHostAlloc(&c.h_ar_sum, 2 * (size_t) c.ar_stride4 * sizeof(uint2), flags) == cudaSuccess &&
              cudaHostAlloc((void**) &c.h_ar_flags, (size_t) tp * 2 * kTpMaxBlocks * 16 * sizeof(uint32_t), flags) == cudaSuccess &&
              cudaHostAlloc((void**) &c.h_ar_flags2, (size_t) tp * 2 * kTpMaxBlocks * 16 * sizeof(uint32_t), flags) == cudaSuccess &&
              cudaHostAlloc((void**) &c.h_bcast_buf, 2 * (size_t) c.bcast_cap * sizeof(float), flags) == cudaSuccess &&
              cudaHostAlloc((void**) &c.h_bcast_flags, 2 * kTpMaxBlocks * 16 * sizeof(uint32_t), flags) == cudaSuccess &&
              cudaHostAlloc((void**) &c.h_argmax_v, (size_t) tp * kTpMaxWin * sizeof(float), flags) == cudaSuccess &&
              cudaHostAlloc((void**) &c.h_argmax_i, (size_t) tp * kTpMaxWin * sizeof(int32_t), flags) == cudaSuccess &&
              cudaHostAlloc((void**) &c.h_argmax_flags, (size_t) tp * 16 * sizeof(uint32_t), flags) == cudaSuccess &&
              cudaHostAlloc((void**) &c.h_topk_v, (size_t) kTpMaxWin * (size_t) tp * 64 * sizeof(float), flags) == cudaSuccess &&
              cudaHostAlloc((void**) &c.h_topk_i, (size_t) kTpMaxWin * (size_t) tp * 64 * sizeof(int32_t), flags) == cudaSuccess &&
              cudaHostAlloc((void**) &c.h_topk_k, 16 * sizeof(int32_t), flags) == cudaSuccess &&
              cudaHostAlloc((void**) &c.h_err, 16 * sizeof(uint32_t), flags) == cudaSuccess;
    if (ok && vocab_total > 0) {
        ok = cudaHostAlloc((void**) &c.h_logits, (size_t) kTpMaxWin * (size_t) vocab_total * sizeof(float), flags) == cudaSuccess;
    }
    if (!ok) {
        tp_comm_free(c);
        return false;
    }

    // 2-Socket NUMA-Split + Hybrid Intra-Socket P2P AllReduce setup for TP=4
    const char* env_numa = std::getenv("MAYA_TP_NUMA");
    const bool allow_numa = (env_numa == nullptr || std::atoi(env_numa) != 0);
    if (allow_numa && tp == 4 && c.dev_numa[0] == c.dev_numa[1] && c.dev_numa[2] == c.dev_numa[3] &&
        c.dev_numa[0] != c.dev_numa[2]) {
        const size_t row_b = (size_t) 2 * (size_t) c.ar_stride4 * sizeof(uint2);
        const size_t flg_b = (size_t) 2 * kTpMaxBlocks * 16 * sizeof(uint32_t);
        const size_t sflg_b = (size_t) 3 * 2 * 2 * kTpMaxBlocks * 16 * sizeof(uint32_t);
        bool nok = true;
        for (int r = 0; r < 4 && nok; ++r) {
            c.h_ar_row[r] = alloc_numa_mapped_portable(row_b, c.dev_numa[r]);
            c.h_ar_flg[r] = (uint32_t*) alloc_numa_mapped_portable(flg_b, c.dev_numa[r]);
            nok = (c.h_ar_row[r] != nullptr && c.h_ar_flg[r] != nullptr);
        }
        const int sock_nodes[2] = {c.dev_numa[0], c.dev_numa[2]};
        for (int s = 0; s < 2 && nok; ++s) {
            c.h_sock_sum[s] = alloc_numa_mapped_portable(row_b, sock_nodes[s]);
            c.h_sock_fin[s] = alloc_numa_mapped_portable(row_b, sock_nodes[s]);
            c.h_sock_flg[s] = (uint32_t*) alloc_numa_mapped_portable(sflg_b, sock_nodes[s]);
            nok = (c.h_sock_sum[s] != nullptr && c.h_sock_fin[s] != nullptr && c.h_sock_flg[s] != nullptr);
        }
        if (nok) {
            c.numa_split = true;
            c.h_ar_numa_alloc = true;
        }
    }

    int saved_dev = 0;
    cudaGetDevice(&saved_dev);
    const char* env_p2p = std::getenv("MAYA_TP_P2P");
    const bool allow_p2p = (env_p2p == nullptr || std::atoi(env_p2p) != 0);
    if (allow_p2p && c.numa_split && tp == 4) {
        if (verify_p2p_pair_bitexact(c.devs[0], c.devs[1]) && verify_p2p_pair_bitexact(c.devs[2], c.devs[3])) {
            bool pok = true;
            const size_t p2p_b = (size_t) 2 * (size_t) c.p2p_half_stride8 * sizeof(uint4);
            const size_t pflg_b = (size_t) 2 * 2 * kTpMaxBlocks * 16 * sizeof(uint32_t);
            for (int r = 0; r < 4 && pok; ++r) {
                cudaSetDevice(c.devs[r]);
                pok = cudaMalloc(&c.d_p2p_in[r], p2p_b) == cudaSuccess &&
                      cudaMalloc(&c.d_p2p_fin[r], p2p_b) == cudaSuccess &&
                      cudaMalloc(&c.d_p2p_flg[r], pflg_b) == cudaSuccess;
                if (pok) cudaMemset(c.d_p2p_flg[r], 0, pflg_b);
            }
            if (pok) {
                c.p2p_intra = true;
            } else {
                for (int r = 0; r < 4; ++r) {
                    cudaSetDevice(c.devs[r]);
                    if (c.d_p2p_in[r]) { cudaFree(c.d_p2p_in[r]); c.d_p2p_in[r] = nullptr; }
                    if (c.d_p2p_fin[r]) { cudaFree(c.d_p2p_fin[r]); c.d_p2p_fin[r] = nullptr; }
                    if (c.d_p2p_flg[r]) { cudaFree(c.d_p2p_flg[r]); c.d_p2p_flg[r] = nullptr; }
                }
                (void) cudaGetLastError();
            }
        }
    }
    if (tp == 4) {
        std::fprintf(stderr, "maya tp-comm: TP=4 NUMA-split=%s (N%d:G%d,G%d / N%d:G%d,G%d), intra-socket P2P=%s\n",
                     c.numa_split ? "ON" : "OFF", c.dev_numa[0], c.devs[0], c.devs[1],
                     c.dev_numa[2], c.devs[2], c.devs[3],
                     c.p2p_intra ? "ON (bit-exact verified)" : "OFF (NUMA RAM hier)");
    }

    tp_comm_reset_flags(c, /*reset_p2p=*/true);

    // Preload AND run a seq=0 eager warmup of every spinning collective kernel on EVERY rank!
    // Prevents CUDA 12.2+ lazy-loading deadlock when Graph 0 starts spinning before Graph 1..3 launch.
    for (int r = 0; r < tp; ++r) {
        cudaSetDevice(c.devs[r]);
        const size_t topk_stage_n = (size_t) kTpMaxWin * (size_t) kTopkStageBlocks * 64u;
        if (cudaMalloc(&c.d_topk_v[r], topk_stage_n * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&c.d_topk_i[r], topk_stage_n * sizeof(int32_t)) != cudaSuccess) {
            cudaSetDevice(saved_dev);
            tp_comm_free(c);
            return false;
        }
        float* dummy = nullptr;
        if (cudaMalloc(&dummy, 1024) != cudaSuccess) { cudaSetDevice(saved_dev); tp_comm_free(c); return false; }
        cudaMemset(dummy, 0, 1024);
        if (tp == 2) {
            tp_ar_host_ag_f16_kernel<2><<<1, 256>>>((float4*) dummy, (uint2*) c.h_ar_rows, c.h_ar_flags, c.ar_stride4, r, 0u, 16);
            tp_ar_host_ag_f16_u4_kernel<2><<<1, 256>>>((float4*) dummy, (uint4*) c.h_ar_rows, c.h_ar_flags, stride8, r, 0u, 8);
            tp_ar_host_rsag_f16_u4_kernel<2><<<1, 256>>>((float4*) dummy, (uint4*) c.h_ar_rows, (uint4*) c.h_ar_sum, c.h_ar_flags, c.h_ar_flags2, stride8, r, 0u, 8);
            tp_moe_combine_miss_ar_f16_u4_kernel<2><<<1, 256>>>(nullptr, nullptr, nullptr, nullptr, nullptr, (float4*) dummy, nullptr, nullptr, nullptr, 0u, 2560, 10, (uint4*) c.h_ar_rows, c.h_ar_flags, stride8, r, 0u, 8);
        } else {
            tp_ar_host_ag_f16_kernel<4><<<1, 256>>>((float4*) dummy, (uint2*) c.h_ar_rows, c.h_ar_flags, c.ar_stride4, r, 0u, 16);
            tp_ar_host_ag_f16_u4_kernel<4><<<1, 256>>>((float4*) dummy, (uint4*) c.h_ar_rows, c.h_ar_flags, stride8, r, 0u, 8);
            tp_ar_host_rsag_f16_u4_kernel<4><<<1, 256>>>((float4*) dummy, (uint4*) c.h_ar_rows, (uint4*) c.h_ar_sum, c.h_ar_flags, c.h_ar_flags2, stride8, r, 0u, 8);
            tp_moe_combine_miss_ar_f16_u4_kernel<4><<<1, 256>>>(nullptr, nullptr, nullptr, nullptr, nullptr, (float4*) dummy, nullptr, nullptr, nullptr, 0u, 2560, 10, (uint4*) c.h_ar_rows, c.h_ar_flags, stride8, r, 0u, 8);
            if (c.numa_split) {
                const int sock = r >> 1;
                const int peer = r ^ 1;
                tp_ar_host_ag_f16_u4_split4_kernel<false><<<1, 256>>>(
                    (float4*) dummy,
                    (uint4*) c.h_ar_row[0], (uint4*) c.h_ar_row[1], (uint4*) c.h_ar_row[2], (uint4*) c.h_ar_row[3],
                    c.h_ar_flg[0], c.h_ar_flg[1], c.h_ar_flg[2], c.h_ar_flg[3],
                    nullptr, nullptr, 0,
                    stride8, r, 0u, 8);
                tp_moe_combine_miss_ar_f16_u4_split4_kernel<false><<<1, 256>>>(
                    nullptr, nullptr, nullptr, nullptr, nullptr, (float4*) dummy, nullptr, nullptr, nullptr, 0u, 2560, 10,
                    (uint4*) c.h_ar_row[0], (uint4*) c.h_ar_row[1], (uint4*) c.h_ar_row[2], (uint4*) c.h_ar_row[3],
                    c.h_ar_flg[0], c.h_ar_flg[1], c.h_ar_flg[2], c.h_ar_flg[3],
                    nullptr, nullptr, 0,
                    stride8, r, 0u, 8);
                if (c.p2p_intra) {
                    tp_ar_host_ag_f16_u4_split4_kernel<true><<<1, 256>>>(
                        (float4*) dummy,
                        (uint4*) c.h_ar_row[0], (uint4*) c.h_ar_row[1], (uint4*) c.h_ar_row[2], (uint4*) c.h_ar_row[3],
                        c.h_ar_flg[0], c.h_ar_flg[1], c.h_ar_flg[2], c.h_ar_flg[3],
                        (uint4*) c.d_p2p_in[r], (uint4*) c.d_p2p_in[peer], c.p2p_half_stride8,
                        stride8, r, 0u, 8);
                    tp_moe_combine_miss_ar_f16_u4_split4_kernel<true><<<1, 256>>>(
                        nullptr, nullptr, nullptr, nullptr, nullptr, (float4*) dummy, nullptr, nullptr, nullptr, 0u, 2560, 10,
                        (uint4*) c.h_ar_row[0], (uint4*) c.h_ar_row[1], (uint4*) c.h_ar_row[2], (uint4*) c.h_ar_row[3],
                        c.h_ar_flg[0], c.h_ar_flg[1], c.h_ar_flg[2], c.h_ar_flg[3],
                        (uint4*) c.d_p2p_in[r], (uint4*) c.d_p2p_in[peer], c.p2p_half_stride8,
                        stride8, r, 0u, 8);
                }
                tp_ar_numa_hier_f16_u4_kernel<<<1, 256>>>(
                    (float4*) dummy,
                    (uint4*) c.h_ar_row[sock * 2], (uint4*) c.h_ar_row[sock * 2 + 1],
                    (uint4*) c.h_sock_sum[sock], (uint4*) c.h_sock_sum[1 - sock], (uint4*) c.h_sock_fin[sock],
                    c.h_sock_flg[sock], c.h_sock_flg[1 - sock], stride8, r, 0u, 8);
                if (c.p2p_intra) {
                    tp_ar_p2p_hier_f16_u4_kernel<<<1, 256>>>(
                        (float4*) dummy,
                        (uint4*) c.d_p2p_in[r], (uint4*) c.d_p2p_in[peer],
                        (uint4*) c.d_p2p_fin[r], (uint4*) c.d_p2p_fin[peer],
                        c.d_p2p_flg[r], c.d_p2p_flg[peer],
                        (uint4*) c.h_sock_sum[sock], (uint4*) c.h_sock_sum[1 - sock],
                        c.h_sock_flg[sock], c.h_sock_flg[1 - sock], stride8, c.p2p_half_stride8, r, 0u, 8);
                    tp_ag_p2p_hc_f16_kernel<<<1, 256>>>(
                        (uint4*) dummy, (uint4*) dummy, (uint4*) dummy,
                        (uint4*) c.d_p2p_in[r], (uint4*) c.d_p2p_in[peer],
                        (uint4*) c.d_p2p_fin[r], (uint4*) c.d_p2p_fin[peer],
                        c.d_p2p_flg[r], c.d_p2p_flg[peer],
                        (uint4*) c.h_sock_sum[sock], (uint4*) c.h_sock_sum[1 - sock],
                        c.h_sock_flg[sock], c.h_sock_flg[1 - sock], stride8, c.p2p_half_stride8, r, 0u, 8, 1);
                }
            }
        }
        tp_bcast_f32_kernel<<<1, 256>>>((float4*) dummy, (float4*) c.h_bcast_buf, c.h_bcast_flags, c.bcast_cap / 4, r, 0u, 16);
        tp_lmhead_merge_argmax_kernel<<<1, 1>>>(c.h_argmax_v, c.h_argmax_i, c.h_argmax_flags, tp, r, 0u, 1, (int32_t*) dummy);
        cudaDeviceSynchronize();
        cudaFree(dummy);
    }
    cudaSetDevice(saved_dev);
    tp_comm_reset_flags(c, /*reset_p2p=*/true);
    return true;
}

void tp_comm_free(TpComm& c) {
    int saved_dev = 0;
    cudaGetDevice(&saved_dev);
    for (int r = 0; r < kTpMaxRanks; ++r) {
        if (c.d_topk_v[r] || c.d_topk_i[r] || c.d_p2p_in[r] || c.d_p2p_fin[r] || c.d_p2p_flg[r]) {
            cudaSetDevice(c.devs[r]);
            if (c.d_topk_v[r]) cudaFree(c.d_topk_v[r]);
            if (c.d_topk_i[r]) cudaFree(c.d_topk_i[r]);
            if (c.d_p2p_in[r]) cudaFree(c.d_p2p_in[r]);
            if (c.d_p2p_fin[r]) cudaFree(c.d_p2p_fin[r]);
            if (c.d_p2p_flg[r]) cudaFree(c.d_p2p_flg[r]);
        }
    }
    cudaSetDevice(saved_dev);
    if (c.h_ar_numa_alloc) {
        for (int r = 0; r < kTpMaxRanks; ++r) {
            free_numa_mapped_portable(c.h_ar_row[r]);
            free_numa_mapped_portable(c.h_ar_flg[r]);
        }
        for (int s = 0; s < 2; ++s) {
            free_numa_mapped_portable(c.h_sock_sum[s]);
            free_numa_mapped_portable(c.h_sock_fin[s]);
            free_numa_mapped_portable(c.h_sock_flg[s]);
        }
    }
    if (c.h_ar_rows) cudaFreeHost(c.h_ar_rows);
    if (c.h_ar_sum) cudaFreeHost(c.h_ar_sum);
    if (c.h_ar_flags) cudaFreeHost(c.h_ar_flags);
    if (c.h_ar_flags2) cudaFreeHost(c.h_ar_flags2);
    if (c.h_bcast_buf) cudaFreeHost(c.h_bcast_buf);
    if (c.h_bcast_flags) cudaFreeHost(c.h_bcast_flags);
    if (c.h_argmax_v) cudaFreeHost(c.h_argmax_v);
    if (c.h_argmax_i) cudaFreeHost(c.h_argmax_i);
    if (c.h_argmax_flags) cudaFreeHost(c.h_argmax_flags);
    if (c.h_topk_v) cudaFreeHost(c.h_topk_v);
    if (c.h_topk_i) cudaFreeHost(c.h_topk_i);
    if (c.h_topk_k) cudaFreeHost(c.h_topk_k);
    if (c.h_logits) cudaFreeHost(c.h_logits);
    if (c.h_err) cudaFreeHost(c.h_err);
    c = TpComm{};
}

void tp_comm_reset_flags(TpComm& c, bool reset_p2p) {
    if (c.h_ar_flags) std::memset(c.h_ar_flags, 0, (size_t) c.tp * 2 * kTpMaxBlocks * 16 * sizeof(uint32_t));
    if (c.h_ar_flags2) std::memset(c.h_ar_flags2, 0, (size_t) c.tp * 2 * kTpMaxBlocks * 16 * sizeof(uint32_t));
    if (c.numa_split) {
        for (int r = 0; r < c.tp; ++r) {
            if (c.h_ar_flg[r]) std::memset(c.h_ar_flg[r], 0, (size_t) 2 * kTpMaxBlocks * 16 * sizeof(uint32_t));
        }
        if (reset_p2p) {
            for (int s = 0; s < 2; ++s) {
                if (c.h_sock_flg[s]) std::memset(c.h_sock_flg[s], 0, (size_t) 3 * 2 * 2 * kTpMaxBlocks * 16 * sizeof(uint32_t));
            }
            if (c.p2p_intra) {
                int saved_dev = 0;
                cudaGetDevice(&saved_dev);
                const size_t pflg_b = (size_t) 2 * 2 * kTpMaxBlocks * 16 * sizeof(uint32_t);
                for (int r = 0; r < c.tp; ++r) {
                    if (c.d_p2p_flg[r]) {
                        cudaSetDevice(c.devs[r]);
                        cudaMemset(c.d_p2p_flg[r], 0, pflg_b);
                    }
                }
                cudaSetDevice(saved_dev);
            }
        }
    }
    if (c.h_bcast_flags) std::memset(c.h_bcast_flags, 0, 2 * kTpMaxBlocks * 16 * sizeof(uint32_t));
    if (c.h_argmax_flags) std::memset(c.h_argmax_flags, 0, (size_t) c.tp * 16 * sizeof(uint32_t));
    if (c.h_topk_k) std::memset(c.h_topk_k, 0, 16 * sizeof(int32_t));
    if (c.h_err) std::memset(c.h_err, 0, 16 * sizeof(uint32_t));
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void tp_allreduce_f16(float* x, int n_floats, const TpComm& c, int rank, uint32_t seq, void* stream) {
    if (n_floats <= 0 || c.tp <= 1) return;
    cudaStream_t st = (cudaStream_t) stream;
    const int64_t stride8 = c.ar_stride4 / 2;
    if ((n_floats & 7) == 0) {
        const int n8 = n_floats / 8;
        if (n_floats > 65536) {
            const int nb = kTpMaxBlocks;
            if (c.tp == 4 && c.numa_split && c.p2p_intra) {
                const int sock = rank >> 1;
                const int peer = rank ^ 1;
                tp_ar_p2p_hier_f16_u4_kernel<<<nb, 256, 0, st>>>(
                    (float4*) x,
                    (uint4*) c.d_p2p_in[rank], (uint4*) c.d_p2p_in[peer],
                    (uint4*) c.d_p2p_fin[rank], (uint4*) c.d_p2p_fin[peer],
                    c.d_p2p_flg[rank], c.d_p2p_flg[peer],
                    (uint4*) c.h_sock_sum[sock], (uint4*) c.h_sock_sum[1 - sock],
                    c.h_sock_flg[sock], c.h_sock_flg[1 - sock],
                    stride8, c.p2p_half_stride8, rank, seq, n8);
                check_tp("tp_allreduce_f16_p2p_hier");
                return;
            }
            if (c.tp == 4 && c.numa_split) {
                const int sock = rank >> 1;
                tp_ar_numa_hier_f16_u4_kernel<<<nb, 256, 0, st>>>(
                    (float4*) x,
                    (uint4*) c.h_ar_row[sock * 2], (uint4*) c.h_ar_row[sock * 2 + 1],
                    (uint4*) c.h_sock_sum[sock], (uint4*) c.h_sock_sum[1 - sock], (uint4*) c.h_sock_fin[sock],
                    c.h_sock_flg[sock], c.h_sock_flg[1 - sock],
                    stride8, rank, seq, n8);
                check_tp("tp_allreduce_f16_numa_hier");
                return;
            }
            if (c.h_ar_sum != nullptr && c.h_ar_flags2 != nullptr) {
                if (c.tp == 2) {
                    tp_ar_host_rsag_f16_u4_kernel<2><<<nb, 256, 0, st>>>(
                        (float4*) x, (uint4*) c.h_ar_rows, (uint4*) c.h_ar_sum, c.h_ar_flags, c.h_ar_flags2,
                        stride8, rank, seq, n8);
                } else {
                    tp_ar_host_rsag_f16_u4_kernel<4><<<nb, 256, 0, st>>>(
                        (float4*) x, (uint4*) c.h_ar_rows, (uint4*) c.h_ar_sum, c.h_ar_flags, c.h_ar_flags2,
                        stride8, rank, seq, n8);
                }
                check_tp("tp_allreduce_f16_rsag");
                return;
            }
        }
        int nb = 1;
        if (n_floats >= 15360) nb = 4;
        else if (n_floats > 2560) nb = 2;
        if (c.tp == 4 && c.numa_split) {
            static const int env_dec_nb = [] {
                const char* v = std::getenv("MAYA_TP_DEC_NB");
                return v ? std::atoi(v) : 0;
            }();
            static const bool env_dec_p2p = [] {
                const char* v = std::getenv("MAYA_TP_DEC_P2P");
                return v != nullptr && std::atoi(v) != 0;
            }();
            // Wave 6: Single-TX Pure-RAM DDIO (21-30% faster on decode 20-40 KB payloads than Dual-TX BAR+RAM!)
            const bool use_p2p_dec = env_dec_p2p && c.p2p_intra && n8 <= c.p2p_half_stride8;
            if (env_dec_nb > 0 && env_dec_nb <= kTpMaxBlocks) nb = env_dec_nb;
            else nb = (n_floats > 15360) ? 4 : (n_floats > 2560 ? 2 : 1);
            const int peer = rank ^ 1;
            if (use_p2p_dec) {
                tp_ar_host_ag_f16_u4_split4_kernel<true><<<nb, 256, 0, st>>>(
                    (float4*) x,
                    (uint4*) c.h_ar_row[0], (uint4*) c.h_ar_row[1], (uint4*) c.h_ar_row[2], (uint4*) c.h_ar_row[3],
                    c.h_ar_flg[0], c.h_ar_flg[1], c.h_ar_flg[2], c.h_ar_flg[3],
                    (uint4*) c.d_p2p_in[rank], (uint4*) c.d_p2p_in[peer], c.p2p_half_stride8,
                    stride8, rank, seq, n8);
            } else {
                tp_ar_host_ag_f16_u4_split4_kernel<false><<<nb, 256, 0, st>>>(
                    (float4*) x,
                    (uint4*) c.h_ar_row[0], (uint4*) c.h_ar_row[1], (uint4*) c.h_ar_row[2], (uint4*) c.h_ar_row[3],
                    c.h_ar_flg[0], c.h_ar_flg[1], c.h_ar_flg[2], c.h_ar_flg[3],
                    nullptr, nullptr, 0,
                    stride8, rank, seq, n8);
            }
            check_tp("tp_allreduce_f16_u4_split4");
            return;
        }
        if (c.tp == 2) {
            tp_ar_host_ag_f16_u4_kernel<2><<<nb, 256, 0, st>>>(
                (float4*) x, (uint4*) c.h_ar_rows, c.h_ar_flags, stride8, rank, seq, n8);
        } else {
            tp_ar_host_ag_f16_u4_kernel<4><<<nb, 256, 0, st>>>(
                (float4*) x, (uint4*) c.h_ar_rows, c.h_ar_flags, stride8, rank, seq, n8);
        }
        check_tp("tp_allreduce_f16_u4");
        return;
    }
    const int n4 = n_floats / 4;
    int nb = 1;
    if (n_floats > 65536) nb = kTpMaxBlocks;
    else if (n_floats >= 15360) nb = 4;
    else if (n_floats > 2560) nb = 2;
    if (c.tp == 2) {
        tp_ar_host_ag_f16_kernel<2><<<nb, 256, 0, st>>>(
            (float4*) x, (uint2*) c.h_ar_rows, c.h_ar_flags, c.ar_stride4, rank, seq, n4);
    } else {
        tp_ar_host_ag_f16_kernel<4><<<nb, 256, 0, st>>>(
            (float4*) x, (uint2*) c.h_ar_rows, c.h_ar_flags, c.ar_stride4, rank, seq, n4);
    }
    check_tp("tp_allreduce_f16");
}

void tp_allgather_hc_f16(uint16_t* mixed_h, uint16_t* mixed_bf, float* inj, int T, int n_embd,
                         const TpComm& c, int rank, uint32_t seq, void* stream) {
    if (T <= 0 || c.tp != 4 || !c.numa_split || !c.p2p_intra) return;
    const int t_loc = T >> 2;
    const int q8 = (t_loc * n_embd) >> 3;
    const int64_t stride8 = c.ar_stride4 / 2;
    const int sock = rank >> 1;
    const int peer = rank ^ 1;
    tp_ag_p2p_hc_f16_kernel<<<kTpMaxBlocks, 256, 0, (cudaStream_t) stream>>>(
        (uint4*) mixed_h, (uint4*) mixed_bf, (uint4*) inj,
        (uint4*) c.d_p2p_in[rank], (uint4*) c.d_p2p_in[peer],
        (uint4*) c.d_p2p_fin[rank], (uint4*) c.d_p2p_fin[peer],
        c.d_p2p_flg[rank], c.d_p2p_flg[peer],
        (uint4*) c.h_sock_sum[sock], (uint4*) c.h_sock_sum[1 - sock],
        c.h_sock_flg[sock], c.h_sock_flg[1 - sock],
        stride8, c.p2p_half_stride8, rank, seq, q8, t_loc);
    check_tp("tp_allgather_hc_f16");
}

void tp_broadcast_f32(float* x, int n_floats, const TpComm& c, int rank, uint32_t seq, void* stream) {
    if (n_floats <= 0 || c.tp <= 1) return;
    const int n4 = n_floats / 4;
    int nb = 1;
    if (n_floats > 65536) nb = kTpMaxBlocks;
    else if (n_floats > 16384) nb = 4;
    tp_bcast_f32_kernel<<<nb, 256, 0, (cudaStream_t) stream>>>(
        (float4*) x, (float4*) c.h_bcast_buf, c.h_bcast_flags, c.bcast_cap / 4, rank, seq, n4);
    check_tp("tp_broadcast_f32");
}

void tp_ep_route_and_plan(const float* logits, int n_tok, int32_t* ids, float* weights,
                          const int32_t* local_res_layer, const int8_t* global_owner_layer,
                          const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                          int32_t* plan, long long capx, uint32_t* skip, uint32_t ring, int rank, void* stream,
                          const float* mixed_x, int n_embd, float* h_x, int32_t* h_ids, float* h_w,
                          uint32_t* h_seq, uint32_t* h_miss_l, uint32_t layer_idx) {
    tp_ep_route_and_plan_kernel<<<1, 256u, 0, (cudaStream_t) stream>>>(
        logits, n_tok, ids, weights, local_res_layer, global_owner_layer, cache_base, slot_off, blob,
        plan, capx, skip, ring, rank == 0 ? 1 : 0,
        mixed_x, n_embd, h_x, h_ids, h_w, h_seq, h_miss_l, layer_idx);
    check_tp("tp_ep_route_and_plan");
}

void tp_moe_combine_miss_allreduce_f16(
    const float* hit_out, const int32_t* p_dst, const int32_t* p_counts,
    const float* weights, const float* shared, float* bo,
    const float* h_ymiss_sum, const uint32_t* skip, const uint32_t* h_flag, uint32_t ring,
    int n_embd, int k, int n_tok, const TpComm& c, int rank, uint32_t seq, void* stream) {
    const int n_floats = n_tok * n_embd;
    const int n8 = n_floats / 8;
    int nb = 1;
    if (n_floats >= 15360) nb = 4;
    else if (n_floats > 2560) nb = 2;
    cudaStream_t st = (cudaStream_t) stream;
    if (c.tp == 4 && c.numa_split) {
        static const int env_dec_nb = [] {
            const char* v = std::getenv("MAYA_TP_DEC_NB");
            return v ? std::atoi(v) : 0;
        }();
        static const bool env_dec_p2p = [] {
            const char* v = std::getenv("MAYA_TP_DEC_P2P");
            return v != nullptr && std::atoi(v) != 0;
        }();
        const bool use_p2p_dec = env_dec_p2p && c.p2p_intra && n8 <= c.p2p_half_stride8;
        if (env_dec_nb > 0 && env_dec_nb <= kTpMaxBlocks) nb = env_dec_nb;
        else nb = (n_floats > 15360) ? 4 : (n_floats > 2560 ? 2 : 1);
        const int peer = rank ^ 1;
        if (use_p2p_dec) {
            tp_moe_combine_miss_ar_f16_u4_split4_kernel<true><<<nb, 256, 0, st>>>(
                hit_out, p_dst, p_counts, weights, shared, (float4*) bo, (const float4*) h_ymiss_sum,
                skip, h_flag, ring, n_embd, k,
                (uint4*) c.h_ar_row[0], (uint4*) c.h_ar_row[1], (uint4*) c.h_ar_row[2], (uint4*) c.h_ar_row[3],
                c.h_ar_flg[0], c.h_ar_flg[1], c.h_ar_flg[2], c.h_ar_flg[3],
                (uint4*) c.d_p2p_in[rank], (uint4*) c.d_p2p_in[peer], c.p2p_half_stride8,
                c.ar_stride4 / 2, rank, seq, n8);
        } else {
            tp_moe_combine_miss_ar_f16_u4_split4_kernel<false><<<nb, 256, 0, st>>>(
                hit_out, p_dst, p_counts, weights, shared, (float4*) bo, (const float4*) h_ymiss_sum,
                skip, h_flag, ring, n_embd, k,
                (uint4*) c.h_ar_row[0], (uint4*) c.h_ar_row[1], (uint4*) c.h_ar_row[2], (uint4*) c.h_ar_row[3],
                c.h_ar_flg[0], c.h_ar_flg[1], c.h_ar_flg[2], c.h_ar_flg[3],
                nullptr, nullptr, 0,
                c.ar_stride4 / 2, rank, seq, n8);
        }
    } else if (c.tp == 2) {
        tp_moe_combine_miss_ar_f16_u4_kernel<2><<<nb, 256, 0, st>>>(
            hit_out, p_dst, p_counts, weights, shared, (float4*) bo, (const float4*) h_ymiss_sum,
            skip, h_flag, ring, n_embd, k, (uint4*) c.h_ar_rows, c.h_ar_flags, c.ar_stride4 / 2, rank, seq, n8);
    } else {
        tp_moe_combine_miss_ar_f16_u4_kernel<4><<<nb, 256, 0, st>>>(
            hit_out, p_dst, p_counts, weights, shared, (float4*) bo, (const float4*) h_ymiss_sum,
            skip, h_flag, ring, n_embd, k, (uint4*) c.h_ar_rows, c.h_ar_flags, c.ar_stride4 / 2, rank, seq, n8);
    }
    check_tp("tp_moe_combine_miss_allreduce_f16");
}

void tp_moe_combine_partial(const float* hit_out, const int32_t* p_dst, const int32_t* p_counts,
                            const float* weights, const float* shared, float* bo,
                            int n_embd, int k, int n_tok, void* stream) {
    const dim3 grid((unsigned) ((n_embd + 255) / 256), (unsigned) n_tok);
    tp_moe_combine_partial_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(
        hit_out, p_dst, p_counts, weights, shared, bo, n_embd, k);
    check_tp("tp_moe_combine_partial");
}

void tp_add_miss_sum_unless(float* bo, const float* m_ymiss_sum, const uint32_t* skip, uint32_t ring,
                            int n_floats, void* stream) {
    const int n4 = n_floats / 4;
    const int blocks = (n4 + 255) / 256;
    tp_add_miss_sum_unless_kernel<<<blocks < 64 ? blocks : 64, 256, 0, (cudaStream_t) stream>>>(
        (float4*) bo, (const float4*) m_ymiss_sum, skip, ring, n4);
    check_tp("tp_add_miss_sum_unless");
}

void tp_lmhead_greedy_merge(const float* local_logits, int n_local, int row_offset, int n_tok,
                            const TpComm& c, int rank, uint32_t seq, int32_t* out_ids_rank0, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    tp_lmhead_local_argmax_kernel<<<(unsigned) n_tok, 256, 0, st>>>(
        local_logits, n_local, row_offset, c.h_topk_k, c.h_argmax_v, c.h_argmax_i, rank, n_tok);
    tp_lmhead_merge_argmax_kernel<<<1, 1, 0, st>>>(
        c.h_argmax_v, c.h_argmax_i, c.h_argmax_flags, c.tp, rank, seq, n_tok, out_ids_rank0);
    check_tp("tp_lmhead_greedy_merge");
}

void tp_f32_to_f16_bulk(const float* src, uint16_t* dst, int n_floats, void* stream) {
    if (n_floats <= 0) return;
    const int n4 = n_floats / 4;
    const int nb = (n4 + 255) / 256;
    tp_f32_to_f16_bulk_kernel<<<nb < 224 ? nb : 224, 256, 0, (cudaStream_t) stream>>>(
        (const float4*) src, (uint2*) dst, n4);
    check_tp("tp_f32_to_f16_bulk");
}

void tp_f16_to_f32_bulk(const uint16_t* src, float* dst, int n_floats, void* stream) {
    if (n_floats <= 0) return;
    const int n4 = n_floats / 4;
    const int nb = (n4 + 255) / 256;
    tp_f16_to_f32_bulk_kernel<<<nb < 224 ? nb : 224, 256, 0, (cudaStream_t) stream>>>(
        (const uint2*) src, (float4*) dst, n4);
    check_tp("tp_f16_to_f32_bulk");
}

void tp_lmhead_local_topk(const float* local_logits, int n_local, int row_offset, int n_tok, int k_eff,
                          const TpComm& c, int rank, void* stream) {
    if (k_eff < 1) k_eff = 1;
    if (k_eff > 64) k_eff = 64;
    cudaStream_t st = (cudaStream_t) stream;
    const dim3 g1((unsigned) kTopkStageBlocks, (unsigned) n_tok);
    tp_lmhead_topk_stage1_kernel<<<g1, 256, 0, st>>>(
        local_logits, n_local, k_eff, nullptr, c.d_topk_v[rank], c.d_topk_i[rank]);
    tp_lmhead_topk_stage2_kernel<<<(unsigned) n_tok, 256, 0, st>>>(
        c.d_topk_v[rank], c.d_topk_i[rank], row_offset, k_eff, nullptr, c.h_topk_v, c.h_topk_i, c.tp, rank);
    check_tp("tp_lmhead_local_topk");
}

void tp_lmhead_local_topk_if(const float* local_logits, int n_local, int row_offset, int n_tok,
                             const TpComm& c, int rank, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    const dim3 g1((unsigned) kTopkStageBlocks, (unsigned) n_tok);
    tp_lmhead_topk_stage1_kernel<<<g1, 256, 0, st>>>(
        local_logits, n_local, 0, c.h_topk_k, c.d_topk_v[rank], c.d_topk_i[rank]);
    tp_lmhead_topk_stage2_kernel<<<(unsigned) n_tok, 256, 0, st>>>(
        c.d_topk_v[rank], c.d_topk_i[rank], row_offset, 0, c.h_topk_k, c.h_topk_v, c.h_topk_i, c.tp, rank);
    check_tp("tp_lmhead_local_topk_if");
}

void tp_lmhead_scatter_logits(const float* local_logits, float* dst_full, int n_local, int row_offset,
                              int vocab_total, int n_tok, void* stream) {
    const int n_local4 = n_local / 4;
    const dim3 grid((unsigned) ((n_local4 + 255) / 256), (unsigned) n_tok);
    tp_lmhead_scatter_logits_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(
        (const float4*) local_logits, (float4*) dst_full, n_local4, row_offset / 4, vocab_total / 4);
    check_tp("tp_lmhead_scatter_logits");
}

}  // namespace strata::kernels
