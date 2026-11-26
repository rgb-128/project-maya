#pragma once

#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/session.hpp"
#include "strata/core/verify.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/tp_allreduce.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

/// Callback for true global CPU misses in TP+EP mode:
/// computes ONLY experts with global_owner[layer*NE + e] < 0 and writes weighted sum `out_sum[n_tok * n_embd]`.
using TpPoolMissFn = void (*)(void* user, const float* x_f, const int32_t* ids, const float* w,
                              int64_t n_tok, int64_t k, float* out_sum, int64_t layer);

struct TpRankInit {
    int dev = 0;
    const WeightTable* wt = nullptr;
    const ModelGeometry* g_local = nullptr;
    SessionState* ss = nullptr;
    VerifyHits hits;
    const NativeHead* head = nullptr;
    const int8_t* d_global_owner = nullptr;  ///< Rank 0 only: device [n_layers * n_expert], >=0 resident on some GPU, -1 CPU miss
};

/// Parallel multi-GPU Tensor-Parallel + Expert-Parallel (TP=2/4 + EP=2/4) speculative verify engine.
/// Every rank executes 1/TP of every layer concurrently inside synchronized CUDA graphs with 2 FP16 AllReduces/layer.
class TpVerifier {
public:
    TpVerifier() = default;
    ~TpVerifier();
    TpVerifier(const TpVerifier&) = delete;
    TpVerifier& operator=(const TpVerifier&) = delete;

    bool init(const std::vector<TpRankInit>& ranks, const ModelGeometry& g_full,
              strata::kernels::TpComm* comm, int max_t, std::string& err);

    bool run(int T, const int32_t* tokens, int64_t pos0, TpPoolMissFn pool, void* user,
             int32_t* out, std::string& err);

    void set_sampling(const strata::kernels::SamplerParams& sp) { sampling_ = sp; }
    void set_history(const int32_t* history_dev_rank0, int history_len) {
        hist_d_ = history_dev_rank0;
        hist_len_ = history_len;
    }
    void set_head_sampling(bool on) { head_sampling_ = on; }

    bool commit_begin(int n_keep, std::string& err);
    bool commit_wait(std::string& err);
    bool commit(int n_keep, std::string& err);

    /// Rank 0's complete residual R[T][hc][n_embd] on device 0 (identical across all ranks -> MTP reads directly!).
    const float* final_R_all() const { return tp_ > 0 ? r_[0].R : nullptr; }
    bool read_head_logits(int rows, float* host, std::string& err);
    std::string profile_report();

    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
    int64_t windows = 0;

private:
    static constexpr int kProfPer = 32;

    struct RankState {
        int dev = 0;
        const WeightTable* wt = nullptr;
        ModelGeometry g{};
        SessionState* ss = nullptr;
        VerifyHits hits{};
        const NativeHead* head = nullptr;
        const int8_t* d_global_owner = nullptr;

        cudaStream_t cs = nullptr;
        cudaGraphExec_t exec[9] = {};
        cudaGraphExec_t commit_exec = nullptr;

        unsigned long long* slot_off_d = nullptr;
        uint32_t* skip_d = nullptr;
        void* arena = nullptr;

        // Device workspace pointers carved from `arena`
        int32_t *tok = nullptr, *step = nullptr, *pos = nullptr, *commit = nullptr;
        float *ple = nullptr, *emb = nullptr, *R = nullptr, *mixed = nullptr, *bo = nullptr;
        float *inj = nullptr, *inj2 = nullptr, *lo = nullptr, *rs = nullptr, *xn = nullptr;
        uint8_t* xq = nullptr;
        float *qkv_L = nullptr, *h_L = nullptr, *gate_L = nullptr, *beta_L = nullptr;
        float *z = nullptr, *y = nullptr, *y_dummy = nullptr;
        float *qfull = nullptr, *qcur = nullptr, *kcur = nullptr, *vcur = nullptr;
        float *idx_raw_L = nullptr, *qidx = nullptr, *scores = nullptr;
        int32_t* sel = nullptr;
        float *attn = nullptr, *attn32 = nullptr, *attn_scratch = nullptr, *tail_snap = nullptr;
        float *logits = nullptr, *w = nullptr;
        int32_t* ids = nullptr;
        float *shared = nullptr, *hit_out = nullptr;
        int32_t* plan = nullptr;
        uint8_t *nat_xq = nullptr, *hit_scratch = nullptr;
        float *head_mixed = nullptr;
        uint16_t* sh_bf16 = nullptr;
        float *sh_gate = nullptr, *sh_up = nullptr, *sh_g = nullptr;
        float* head_logits_local = nullptr;   ///< [T][head->n_out()]
        float* head_logits_full = nullptr;    ///< Rank 0 only: [T][n_vocab_total] for sampling/dump
        float* topk_logits_d = nullptr;       ///< Rank 0 only: [T][tp * 64] for fast top-k sampling
        float* hist_snap = nullptr;

        int64_t cap = 0, max_blocks = 0, attn_scratch_floats = 0, plan_i32 = 0;

        unsigned long long* prof = nullptr;
        std::vector<unsigned long long> prof_h;
    };

    bool capture_rank(int r, int T, std::string& err);
    bool record_window_rank(int r, int T, cudaStream_t cs, std::string& err);
    bool capture_commit_rank(int r, std::string& err);
    void dispatch_ranks(int cmd, int arg1, int arg2);
    void dispatch_ranks_async(int cmd, int arg1, int arg2);
    void dispatch_ranks_join();

    // Parallel per-GPU graph launcher pool (eliminates 4.6 ms sequential cudaGraphLaunch skew!)
    struct Launcher {
        std::mutex mu;
        std::condition_variable cv;
        std::atomic<uint64_t> epoch{0};
        std::atomic<int> done{0};
        std::atomic<int> sleeping{0};
        bool quit = false;
        bool async_pending = false;
        int cmd = 0;
        int arg1 = 0;
        int arg2 = 0;
        cudaError_t errs[strata::kernels::kTpMaxRanks] = {};
        std::vector<std::thread> threads;
    } launch_;

    int tp_ = 0;
    ModelGeometry g_full_{};
    strata::kernels::TpComm* comm_ = nullptr;
    RankState r_[strata::kernels::kTpMaxRanks];
    int max_t_ = 0;
    int last_t_ = 0;
    int64_t last_pos0_ = 0;
    int32_t last_tokens_[8] = {};
    int64_t n_vocab_ = 0;
    bool commit_pending_ = false;
    bool head_sampling_ = true;
    const int32_t* hist_d_ = nullptr;
    int hist_len_ = 0;
    strata::kernels::SamplerParams sampling_ = [] {
        strata::kernels::SamplerParams s;
        s.greedy = true;
        s.temperature = 0.0f;
        return s;
    }();

    // Shared portable mapped staging buffers (device-visible on ALL ranks via UVA!)
    int32_t* h_tok_ = nullptr;
    int32_t* h_step_ = nullptr;
    int32_t* h_pos_ = nullptr;
    int32_t* h_commit_ = nullptr;
    float* h_ple_ = nullptr;
    int32_t* h_out_ = nullptr;
    // Rank 0 doorbell buffers
    float* h_x_ = nullptr;
    int32_t* h_ids_ = nullptr;
    float* h_w_ = nullptr;
    uint32_t* h_seq_ = nullptr;
    uint32_t* h_flag_ = nullptr;
    float* h_ymiss_sum_ = nullptr;

    bool prof_on_ = false;
    double prof_sum_[2][kProfPer] = {};
    int64_t prof_windows_ = 0;
};

}  // namespace strata::core
