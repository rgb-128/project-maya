#pragma once

#include <cstdint>
#include <cstddef>

namespace strata::kernels {

static constexpr int kTpMaxRanks = 4;
static constexpr int kTpMaxBlocks = 32;
static constexpr int kTpMaxWin = 8;

/// Shared mapped-host collective communication buffers for multi-GPU Tensor Parallelism (TP=2 or TP=4).
/// Works across PCIe Host Bridges (PHB) and cross-socket QPI/UPI without requiring direct BAR P2P writes.
struct TpComm {
    int tp = 1;
    int devs[kTpMaxRanks] = {0, 1, 2, 3};
    int64_t ar_stride4 = 0;       ///< float4 units per rank per double-buffer slot
    int64_t bcast_cap = 0;        ///< float elements per double-buffer slot
    int64_t vocab_total = 0;      ///< total vocab size for logits gather

    // Mapped pinned portable host buffers (device-visible on all GPUs via UVA)
    void* h_ar_rows = nullptr;    ///< uint2 [tp][2][ar_stride4] (FP16 wire format)
    void* h_ar_sum = nullptr;     ///< uint2 [2][ar_stride4] (FP16 reduced output for 2-phase RS+AG)
    uint32_t* h_ar_flags = nullptr; ///< [tp][2][kTpMaxBlocks][16] (cacheline-padded)
    uint32_t* h_ar_flags2 = nullptr;///< [tp][2][kTpMaxBlocks][16] (Phase-2 RS+AG flags)
    float* h_bcast_buf = nullptr; ///< [2][bcast_cap]
    uint32_t* h_bcast_flags = nullptr; ///< [2][kTpMaxBlocks][16]
    float* h_argmax_v = nullptr;  ///< [tp][kTpMaxWin]
    int32_t* h_argmax_i = nullptr;///< [tp][kTpMaxWin]
    uint32_t* h_argmax_flags = nullptr; ///< [tp][16]
    float* h_topk_v = nullptr;    ///< [kTpMaxWin][tp][64]
    int32_t* h_topk_i = nullptr;  ///< [kTpMaxWin][tp][64]
    int32_t* h_topk_k = nullptr;  ///< [16] dynamic top-k control read inside captured CUDA graph (0 = skip)
    float* d_topk_v[kTpMaxRanks] = {};   ///< per-GPU Stage-1 top-k scratch [kTpMaxWin * 32 * 64]
    int32_t* d_topk_i[kTpMaxRanks] = {}; ///< per-GPU Stage-1 top-k scratch [kTpMaxWin * 32 * 64]
    float* h_logits = nullptr;    ///< [kTpMaxWin][vocab_total] (allocated lazily or at init)
    uint32_t* h_err = nullptr;    ///< [16] sticky timeout/error bitmask

    // 2-Socket NUMA-local & Hybrid Intra-Socket P2P AllReduce buffers (zero-QPI local writes + 1.69x faster prefill AR!)
    int dev_numa[kTpMaxRanks] = {0, 0, 0, 0};
    bool numa_split = false;      ///< true when TP=4 spans 2 NUMA sockets (0,1 on sock0; 2,3 on sock1)
    bool p2p_intra = false;       ///< true when intra-socket PHB P2P (0<->1, 2<->3) passed bit-exact self-test
    bool h_ar_numa_alloc = false; ///< true when h_ar_row/h_ar_flg use mbind + cudaHostRegister
    void* h_ar_row[kTpMaxRanks] = {};        ///< per-rank NUMA-local FP16 row [2][ar_stride4] uint2
    uint32_t* h_ar_flg[kTpMaxRanks] = {};    ///< per-rank NUMA-local flags [2][kTpMaxBlocks][16]
    void* h_sock_sum[2] = {};                ///< per-socket NUMA-local hierarchical partial sum [2][ar_stride4] uint2
    void* h_sock_fin[2] = {};                ///< per-socket NUMA-local final sum (RAM-only hierarchical fallback)
    uint32_t* h_sock_flg[2] = {};            ///< per-socket NUMA-local flags [3][2][2][kTpMaxBlocks][16]
    int64_t p2p_half_stride8 = 0;            ///< uint4 units per slot in half-size P2P buffers
    void* d_p2p_in[kTpMaxRanks] = {};        ///< per-GPU half-size P2P Stage-1 staging [2][p2p_half_stride8] uint4
    void* d_p2p_fin[kTpMaxRanks] = {};       ///< per-GPU half-size P2P Stage-3 staging [2][p2p_half_stride8] uint4
    uint32_t* d_p2p_flg[kTpMaxRanks] = {};   ///< per-GPU P2P flags [2][2][kTpMaxBlocks][16]

    uint32_t ar_seq_base = 0;     ///< rolling sequence base per launch (if flags not reset)
};

/// Allocates mapped host collective buffers and pre-warms all collective kernels on every rank
/// (preventing CUDA 12.2+ lazy-loading P2P/multi-GPU spin deadlocks).
bool tp_comm_init(TpComm& c, int tp, const int* devs, int64_t max_ar_floats, int64_t max_bcast_floats,
                  int64_t vocab_total);
void tp_comm_free(TpComm& c);

/// Resets per-window flags in host memory before launching synchronized CUDA graphs.
/// Set `reset_p2p=true` before prefill chunks that use large P2P hierarchical AllReduces.
void tp_comm_reset_flags(TpComm& c, bool reset_p2p = false);

/// FP16-wire deterministic AllReduce across `c.tp` ranks (CUDA-graph capturable, zero host sync).
/// `n_floats` must be a multiple of 4. Result is bit-identical on every rank.
void tp_allreduce_f16(float* x, int n_floats, const TpComm& c, int rank, uint32_t seq, void* stream);

/// 4-GPU Sequence-Parallel HC-read AllGather (Hybrid Intra-Socket P2P + Cross-Socket NUMA, single GPU launch):
/// Each rank owns token slice `[rank * (T/4), (rank+1) * (T/4))` of `mixed_h [T, n_embd]` (FP16) and `inj [T, 4]` (FP32).
/// Gathers the other 3 quarters into `mixed_h`, converts incoming quarters FP16->BF16 into `mixed_bf`, and gathers `inj`.
/// Requires `c.tp == 4 && c.numa_split && c.p2p_intra && (T & 3) == 0 && ((T/4 * n_embd) & 7) == 0`.
void tp_allgather_hc_f16(uint16_t* mixed_h, uint16_t* mixed_bf, float* inj, int T, int n_embd,
                         const TpComm& c, int rank, uint32_t seq, void* stream);

/// Fast GPU FP32 <-> FP16 bulk converters (used by prefill full-duplex DMA AllReduce).
void tp_f32_to_f16_bulk(const float* src, uint16_t* dst, int n_floats, void* stream);
void tp_f16_to_f32_bulk(const uint16_t* src, float* dst, int n_floats, void* stream);

/// FP32 broadcast from Rank 0 to all ranks (CUDA-graph capturable, zero host sync).
void tp_broadcast_f32(float* x, int n_floats, const TpComm& c, int rank, uint32_t seq, void* stream);

/// Fused exact top-10 router + per-GPU EP=TP local VRAM plan + Rank-0 global miss detector (+ optional fused doorbell).
/// - Every rank builds its local VRAM plan for `native_expert_grouped` (only experts with `local_res[e] >= 0`).
/// - Rank 0 ALSO checks `global_owner[e]` (>=0 if ANY GPU holds `e`, -1 if CPU miss):
///   sets `*skip = ring` if ZERO global misses exist (skipping host doorbell!), else `*skip = 0`.
/// - When `h_x != nullptr` (Rank 0 fused doorbell), publishes `mixed_x/ids/weights` to host and increments `*h_seq`
///   directly inside the same kernel (eliminating a separate `doorbell_publish_unless` launch).
void tp_ep_route_and_plan(const float* logits, int n_tok, int32_t* ids, float* weights,
                          const int32_t* local_res_layer, const int8_t* global_owner_layer,
                          const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                          int32_t* plan, long long capx, uint32_t* skip, uint32_t ring, int rank, void* stream,
                          const float* mixed_x = nullptr, int n_embd = 0, float* h_x = nullptr,
                          int32_t* h_ids = nullptr, float* h_w = nullptr, uint32_t* h_seq = nullptr,
                          uint32_t* h_miss_l = nullptr, uint32_t layer_idx = 0);

/// Partial MoE combine on Rank `r`:
/// `bo[t, d] = shared[t, d] + sum_{m in [0, p_counts[1]) : slot = p_dst[m], slot/K == t} w[slot] * hit_out[slot, d]`.
/// Only reads the entries this GPU actually computed!
void tp_moe_combine_partial(const float* hit_out, const int32_t* p_dst, const int32_t* p_counts,
                            const float* weights, const float* shared, float* bo,
                            int n_embd, int k, int n_tok, void* stream);

/// Fused MoE partial combine + Rank-0 conditional CPU-miss wait/add + FP16-wire AllReduce into `bo`.
/// Replaces 4 separate launches on Rank 0 (`combine` + `wait_flag` + `add_miss` + `AR2`) and 2 on Ranks 1..3
/// with ONE single kernel launch and zero intermediate `bo` HBM2 round-trips!
void tp_moe_combine_miss_allreduce_f16(
    const float* hit_out, const int32_t* p_dst, const int32_t* p_counts,
    const float* weights, const float* shared, float* bo,
    const float* h_ymiss_sum, const uint32_t* skip, const uint32_t* h_flag, uint32_t ring,
    int n_embd, int k, int n_tok, const TpComm& c, int rank, uint32_t seq, void* stream);

/// Rank 0 only: if `*skip != ring`, adds pre-weighted CPU miss sum `m_ymiss_sum[T * n_embd]` into `bo`.
void tp_add_miss_sum_unless(float* bo, const float* m_ymiss_sum, const uint32_t* skip, uint32_t ring,
                            int n_floats, void* stream);

/// Sharded LM-head greedy argmax inside the CUDA graph:
/// Each rank finds max over its `[row_offset, row_offset + n_local)` logits, writes to `c.h_argmax_*`;
/// Rank 0 merges across `c.tp` ranks and writes global winning token ids into `out_ids[0..n_tok)`.
void tp_lmhead_greedy_merge(const float* local_logits, int n_local, int row_offset, int n_tok,
                            const TpComm& c, int rank, uint32_t seq, int32_t* out_ids_rank0, void* stream);

/// Sharded LM-head exact local top-K extraction (`1 <= k_eff <= 64`) into `c.h_topk_v/i[t][rank][0..k_eff)`.
void tp_lmhead_local_topk(const float* local_logits, int n_local, int row_offset, int n_tok, int k_eff,
                          const TpComm& c, int rank, void* stream);

/// Graph-capturable conditional local top-K: reads `*c.h_topk_k` on device; returns immediately if `<= 0`.
void tp_lmhead_local_topk_if(const float* local_logits, int n_local, int row_offset, int n_tok,
                             const TpComm& c, int rank, void* stream);

/// Copies this rank's `[n_tok][n_local]` logits slice into `dst_full[n_tok][vocab_total]` at column `row_offset`.
void tp_lmhead_scatter_logits(const float* local_logits, float* dst_full, int n_local, int row_offset,
                              int vocab_total, int n_tok, void* stream);

}  // namespace strata::kernels
