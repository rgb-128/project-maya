#include "strata/core/tp_verify.hpp"

#include "strata/core/on_device.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/platform/cpu_relax.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T>
    T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped_portable(size_t bytes, void** h) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return true;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "tp_verify: " + name + " is not served natively";
        return false;
    }
    return true;
}

}  // namespace

TpVerifier::~TpVerifier() {
    if (!launch_.threads.empty()) {
        {
            std::lock_guard<std::mutex> lk(launch_.mu);
            launch_.quit = true;
            launch_.epoch.fetch_add(1, std::memory_order_release);
        }
        launch_.cv.notify_all();
        for (auto& th : launch_.threads)
            if (th.joinable()) th.join();
        launch_.threads.clear();
    }
    for (int i = 0; i < tp_; ++i) {
        const OnDevice on_dev(r_[i].dev);
        if (r_[i].cs) cudaStreamSynchronize(r_[i].cs);
        for (auto& e : r_[i].exec)
            if (e) cudaGraphExecDestroy(e);
        if (r_[i].commit_exec) cudaGraphExecDestroy(r_[i].commit_exec);
        if (r_[i].cs) cudaStreamDestroy(r_[i].cs);
        if (r_[i].slot_off_d) cudaFree(r_[i].slot_off_d);
        if (r_[i].skip_d) cudaFree(r_[i].skip_d);
        if (r_[i].prof) cudaFree(r_[i].prof);
        if (r_[i].arena) cudaFree(r_[i].arena);
    }
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_,
                     h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_sum_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
}

bool TpVerifier::init(const std::vector<TpRankInit>& ranks, const ModelGeometry& g_full,
                      strata::kernels::TpComm* comm, int max_t, std::string& err) {
    if ((ranks.size() != 2 && ranks.size() != 4) || comm == nullptr || comm->tp != (int) ranks.size()) {
        err = "tp_verify: requires TP=2 or TP=4 with matching TpComm";
        return false;
    }
    if (max_t < 2 || max_t > strata::kernels::kVerifyMaxT || max_t > strata::kernels::cpu::MAXT) {
        err = "tp_verify: invalid max_t";
        return false;
    }
    tp_ = (int) ranks.size();
    g_full_ = g_full;
    comm_ = comm;
    max_t_ = max_t;
    prof_on_ = std::getenv("MAYA_VERIFY_PROFILE") != nullptr;

    const WeightRef* wo = ranks[0].wt->find("output.weight");
    if (wo == nullptr) { err = "tp_verify: output.weight missing"; return false; }
    n_vocab_ = wo->ne1;

    const uint64_t T = (uint64_t) max_t;
    const uint64_t N = (uint64_t) g_full.n_embd;
    const uint64_t K = (uint64_t) ranks[0].ss->k;
    const uint64_t NH_l = (uint64_t) ranks[0].g_local->n_head;
    const uint64_t NKV_l = (uint64_t) ranks[0].g_local->n_head_kv;
    const uint64_t IQ = (uint64_t) g_full.idx_q_heads;

    bool ok = mapped_portable(T * 4, (void**) &h_tok_) &&
              mapped_portable(T * strata::kernels::kStepCount * 4, (void**) &h_step_) &&
              mapped_portable(T * (NH_l + NKV_l + IQ) * 4, (void**) &h_pos_) &&
              mapped_portable((2 + T) * 4 + 16, (void**) &h_commit_) &&
              mapped_portable(T * N * 4, (void**) &h_ple_) &&
              mapped_portable(T * 4 + 16, (void**) &h_out_) &&
              mapped_portable(T * N * 4, (void**) &h_x_) &&
              mapped_portable(T * K * 4, (void**) &h_ids_) &&
              mapped_portable(T * K * 4, (void**) &h_w_) &&
              mapped_portable(64, (void**) &h_seq_) &&
              mapped_portable(64, (void**) &h_flag_) &&
              mapped_portable(T * N * 4, (void**) &h_ymiss_sum_);
    if (!ok) { err = "tp_verify: mapped staging alloc failed"; return false; }

    for (int i = 0; i < tp_; ++i) {
        RankState& rs = r_[i];
        rs.dev = ranks[i].dev;
        rs.wt = ranks[i].wt;
        rs.g = *ranks[i].g_local;
        rs.ss = ranks[i].ss;
        rs.hits = ranks[i].hits;
        rs.head = ranks[i].head;
        rs.d_global_owner = ranks[i].d_global_owner;
        if (!rs.wt || !rs.ss || !rs.head || !rs.head->loaded() || !rs.hits.d_res || !rs.hits.cache_base) {
            err = "tp_verify: incomplete rank " + std::to_string(i);
            return false;
        }
        const OnDevice on_dev(rs.dev);
        const ModelGeometry& g = rs.g;
        const strata::kernels::QsaShapes s = shapes_of(g);
        rs.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
        rs.max_blocks = rs.ss->qsa_states[0].max_cells / s.idx_block + 2;
        rs.attn_scratch_floats = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(rs.cap, s);

        const int64_t cap_e = (int64_t) (T * K);
        const int64_t i32 = 4 + (cap_e + 1) + cap_e + cap_e;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        rs.plan_i32 = ptr_off + 4 * cap_e + (cap_e + 1) + 1;

        const uint64_t HC = (uint64_t) g.hc;
        const uint64_t C = (uint64_t) g.ssm_conv_channels;
        const uint64_t ZV = (uint64_t) g.ssm_value_dim;
        const uint64_t HV = (uint64_t) g.ssm_v_heads;
        const uint64_t NH = (uint64_t) g.n_head;
        const uint64_t HD = (uint64_t) g.head_dim;
        const uint64_t NKV = (uint64_t) g.n_head_kv;
        const uint64_t ID = (uint64_t) g.idx_key_dim;
        const uint64_t nG = (uint64_t) g.n_gdn_layers();
        const uint64_t nQ = (uint64_t) g.n_qsa_layers();
        const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
        const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
        const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

        auto carve = [&](Bump& b) {
            rs.tok = b.take<int32_t>(T);
            rs.step = b.take<int32_t>(T * strata::kernels::kStepCount);
            rs.pos = b.take<int32_t>(T * (NH + NKV + IQ));
            rs.commit = b.take<int32_t>(2 + T);
            rs.ple = b.take<float>(T * N);
            rs.emb = b.take<float>(T * N);
            rs.R = b.take<float>(T * HC * N);
            rs.mixed = b.take<float>(T * N);
            rs.bo = b.take<float>(T * N);
            rs.inj = b.take<float>(T * HC);
            rs.inj2 = b.take<float>(T * HC);
            rs.lo = b.take<float>(T * (uint64_t) g.hc_lr);
            rs.rs = b.take<float>(T * HC);
            rs.xn = b.take<float>(T * HC * N);
            rs.xq = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
            rs.qkv_L = b.take<float>(nG * T * C);
            rs.h_L = b.take<float>(nG * T * C);
            rs.gate_L = b.take<float>(nG * T * HV);
            rs.beta_L = b.take<float>(nG * T * HV);
            rs.z = b.take<float>(T * ZV);
            rs.y = b.take<float>(T * ZV);
            rs.y_dummy = b.take<float>(T * ZV);
            rs.qfull = b.take<float>(T * NH * 2 * HD);
            rs.qcur = b.take<float>(T * NH * HD);
            rs.kcur = b.take<float>(T * NKV * HD);
            rs.vcur = b.take<float>(T * NKV * HD);
            rs.idx_raw_L = b.take<float>(nQ * T * ID);
            rs.qidx = b.take<float>(T * IQ * ID);
            rs.scores = b.take<float>(T * (uint64_t) rs.max_blocks);
            rs.sel = b.take<int32_t>(T * (uint64_t) rs.cap);
            rs.attn = b.take<float>(T * NH * HD);
            rs.attn32 = b.take<float>(T * NH * HD);
            rs.attn_scratch = b.take<float>(T * (uint64_t) rs.attn_scratch_floats);
            rs.tail_snap = b.take<float>(nQ * TS);
            rs.logits = b.take<float>(T * (uint64_t) g.n_expert);
            rs.w = b.take<float>(T * K);
            rs.ids = b.take<int32_t>(T * K);
            rs.shared = b.take<float>(T * N);
            rs.hit_out = b.take<float>(T * K * N);
            rs.plan = b.take<int32_t>((uint64_t) rs.plan_i32 + 16);
            rs.nat_xq = b.take<uint8_t>(T * (N / 32) * 36);
            rs.hit_scratch = b.take<uint8_t>(strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff));
            rs.head_mixed = b.take<float>(T * N);
            rs.sh_bf16 = b.take<uint16_t>(T * N);
            rs.sh_gate = b.take<float>(T * (uint64_t) g.n_ff);
            rs.sh_up = b.take<float>(T * (uint64_t) g.n_ff);
            rs.sh_g = b.take<float>(T + 4);
            rs.head_logits_local = b.take<float>(T * (uint64_t) rs.head->n_out());
            if (i == 0) {
                rs.head_logits_full = b.take<float>(T * (uint64_t) n_vocab_);
                rs.topk_logits_d = b.take<float>(T * (uint64_t) tp_ * 64);
            }
            rs.hist_snap = b.take<float>(T * HS);
        };
        Bump cnt;
        carve(cnt);
        if (cudaMalloc(&rs.arena, cnt.used) != cudaSuccess) {
            err = "tp_verify: rank " + std::to_string(i) + " arena (" + std::to_string(cnt.used >> 20) + " MiB) failed";
            return false;
        }
        cudaMemset(rs.arena, 0, cnt.used);
        Bump real;
        real.base = (uint8_t*) rs.arena;
        carve(real);

        if (cudaMalloc(&rs.skip_d, 64) != cudaSuccess || cudaMemset(rs.skip_d, 0, 64) != cudaSuccess) {
            err = "tp_verify: skip alloc failed";
            return false;
        }
        if (rs.hits.slot_off != nullptr && rs.hits.n_slots > 0) {
            if (cudaMalloc(&rs.slot_off_d, (size_t) rs.hits.n_slots * sizeof(unsigned long long)) != cudaSuccess ||
                cudaMemcpy(rs.slot_off_d, rs.hits.slot_off, (size_t) rs.hits.n_slots * sizeof(unsigned long long),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "tp_verify: slot_off upload failed";
                return false;
            }
        }
        if (cudaStreamCreateWithFlags(&rs.cs, cudaStreamNonBlocking) != cudaSuccess) {
            err = "tp_verify: stream create failed";
            return false;
        }
        if (prof_on_ && i == 0) {
            const size_t np = (size_t) g.n_layers * kProfPer + 4;
            if (cudaMalloc((void**) &rs.prof, np * 8) == cudaSuccess) {
                cudaMemset(rs.prof, 0, np * 8);
                rs.prof_h.assign(np, 0);
            } else {
                prof_on_ = false;
                cudaGetLastError();
            }
        }
        std::fprintf(stderr, "maya tp-verify: rank %d (CUDA%d) up to %d tokens, %.1f MiB device workspace\n",
                     i, rs.dev, max_t, (double) cnt.used / 1048576.0);
    }
    // Spawn persistent launcher threads for ranks 1..tp-1 so all GPUs submit cudaGraphLaunch concurrently!
    // Pin each launcher to a distinct SMT sibling on its own GPU's NUMA socket (never sharing CPU 0 or
    // serializing all 3 ranks onto one core, and never inheriting CPU-0-only affinity!).
    int launch_cpus[4] = {-1, -1, -1, -1};
#if defined(__linux__)
    {
        std::vector<int> smt_n0, smt_n1;
        const std::vector<int> pcores = strata::kernels::cpu::physical_cores(false);
        std::vector<bool> is_phys(256, false);
        for (int c : pcores) if (c >= 0 && c < 256) is_phys[(size_t) c] = true;
        for (int c = 0; c < 128; ++c) {
            if (is_phys[(size_t) c]) continue;
            char path[96];
            std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c);
            if (std::FILE* f = std::fopen(path, "r")) {
                long pkg = -1;
                if (std::fscanf(f, "%ld", &pkg) == 1) {
                    if (pkg == 0) smt_n0.push_back(c);
                    else if (pkg == 1) smt_n1.push_back(c);
                }
                std::fclose(f);
            }
        }
        size_t i0 = 0, i1 = 0;
        for (int r = 1; r < tp_; ++r) {
            const int nd = comm_->dev_numa[r];
            if (nd == 1 && i1 < smt_n1.size()) launch_cpus[r] = smt_n1[i1++];
            else if (i0 < smt_n0.size()) launch_cpus[r] = smt_n0[i0++];
            else if (i1 < smt_n1.size()) launch_cpus[r] = smt_n1[i1++];
        }
    }
#endif
    launch_.quit = false;
    launch_.epoch.store(0, std::memory_order_relaxed);
    for (int r = 1; r < tp_; ++r) {
        const int lcpu = launch_cpus[r];
        launch_.threads.emplace_back([this, r, lcpu]() {
            if (lcpu >= 0) (void) strata::kernels::cpu::pin_current_thread(lcpu);
            cudaSetDevice(r_[r].dev);
            uint64_t seen = 0;
            for (;;) {
                uint64_t cur = launch_.epoch.load(std::memory_order_acquire);
                if (cur == seen) {
                    for (int sp = 0; sp < 200 && cur == seen && !launch_.quit; ++sp) {
                        strata_cpu_pause();
                        cur = launch_.epoch.load(std::memory_order_acquire);
                    }
                    if (cur == seen) {
                        std::unique_lock<std::mutex> lk(launch_.mu);
                        launch_.cv.wait(lk, [&] {
                            return launch_.quit || launch_.epoch.load(std::memory_order_acquire) != seen;
                        });
                        if (launch_.quit) return;
                        cur = launch_.epoch.load(std::memory_order_acquire);
                    }
                }
                if (launch_.quit) return;
                seen = cur;
                const int cmd = launch_.cmd;
                const int a1 = launch_.arg1;
                const int a2 = launch_.arg2;
                cudaError_t ce = cudaSuccess;
                if (cmd == 1) {
                    ce = cudaGraphLaunch(r_[r].exec[a1], r_[r].cs);
                } else if (cmd == 2) {
                    ce = cudaGraphLaunch(r_[r].commit_exec, r_[r].cs);
                } else if (cmd == 3) {
                    strata::kernels::tp_lmhead_local_topk(r_[r].head_logits_local, r_[r].head->n_out(),
                                                          r_[r].head->row_offset(), a1, a2, *comm_, r, r_[r].cs);
                    ce = cudaStreamSynchronize(r_[r].cs);
                }
                launch_.errs[r] = ce;
                launch_.done.fetch_add(1, std::memory_order_release);
            }
        });
    }
    return true;
}

void TpVerifier::dispatch_ranks_join() {
    if (!launch_.async_pending) return;
    while (launch_.done.load(std::memory_order_acquire) < tp_ - 1) strata_cpu_pause();
    launch_.async_pending = false;
}

void TpVerifier::dispatch_ranks_async(int cmd, int arg1, int arg2) {
    dispatch_ranks_join();
    launch_.cmd = cmd;
    launch_.arg1 = arg1;
    launch_.arg2 = arg2;
    launch_.done.store(0, std::memory_order_relaxed);
    for (int r = 0; r < tp_; ++r) launch_.errs[r] = cudaSuccess;
    launch_.async_pending = true;
    {
        std::lock_guard<std::mutex> lk(launch_.mu);
        launch_.epoch.fetch_add(1, std::memory_order_release);
    }
    launch_.cv.notify_all();
    // Rank 0 runs concurrently on the calling thread!
    const OnDevice on_dev(r_[0].dev);
    cudaError_t ce = cudaSuccess;
    if (cmd == 1) {
        ce = cudaGraphLaunch(r_[0].exec[arg1], r_[0].cs);
    } else if (cmd == 2) {
        ce = cudaGraphLaunch(r_[0].commit_exec, r_[0].cs);
    } else if (cmd == 3) {
        strata::kernels::tp_lmhead_local_topk(r_[0].head_logits_local, r_[0].head->n_out(),
                                              r_[0].head->row_offset(), arg1, arg2, *comm_, 0, r_[0].cs);
        ce = cudaStreamSynchronize(r_[0].cs);
    }
    launch_.errs[0] = ce;
}

void TpVerifier::dispatch_ranks(int cmd, int arg1, int arg2) {
    dispatch_ranks_async(cmd, arg1, arg2);
    dispatch_ranks_join();
}

bool TpVerifier::record_window_rank(int r, int T, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    RankState& rs = r_[r];
    const ModelGeometry& g = rs.g;
    const WeightTable& wt = *rs.wt;
    SessionState& ss = *rs.ss;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k;
    const int64_t C = g.ssm_conv_channels, ZV = g.ssm_value_dim, HV = g.ssm_v_heads, HK = g.ssm_k_heads;
    const int64_t NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * (uint64_t) HV * (uint64_t) g.ssm_state_size +
                                (uint64_t) C * (uint64_t) (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = r_[0].ss->ple.ready();
    auto Rt = [&](int t) { return rs.R + (size_t) t * HC * N; };
    auto stamp = [&](int64_t l, int idx) {
        if (prof_on_ && r == 0 && rs.prof) gpu_stamp(rs.prof, (int) (l * kProfPer + idx), cs);
    };

    // ---- Stage inputs from shared portable mapped host memory
    copy_i32_from_mapped(rs.tok, h_tok_, T, cs);
    copy_i32_from_mapped(rs.step, h_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(rs.pos, h_pos_, (int64_t) MT * (NH + NKV + IQ), cs);
    const int32_t* pos_k = rs.pos + MT * NH;
    const int32_t* pos_i = rs.pos + MT * (NH + NKV);
    if (r == 0 && ple_on) copy_from_mapped(rs.ple, h_ple_, (int64_t) T * N, cs);

    // ---- Embeddings (NativeEmbed lives in portable mapped host RAM -> all 4 GPUs gather directly!)
    const NativeEmbed* ne = native_embed();
    if (!ne) { err = "tp_verify: requires NativeEmbed"; return false; }
    ne->gather_dev(rs.tok, T, rs.emb, cs);
    broadcast_streams(rs.emb, rs.R, N, (int) HC, T, cs);

    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    uint32_t ar_seq = 0;
    uint32_t bcast_seq = 0;

    for (int64_t l = 0; l < g.n_layers; ++l) {
        stamp(l, 0);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        bool pending = l > 0 && !cvec().covers(l - 1);
        if (l == 1 && ple_on) {
            if (r == 0) {
                gr_write_batch(rs.R, rs.bo, rs.inj2, gs, rs.R, T, cs);
                if (T > 1) {
                    // Reuse idle buffers on Rank 0: allocate? Wait: qfull is T*3072 (< T*10240!), use xn (T*10240) and hit_out (T*25600)!
                    try {
                        ple_block_batch(T, rs.ple, rs.R, ss.ple.hist, ss.ple.w,
                                        rs.xn, rs.mixed, rs.xq, rs.hist_snap, ss.ple.scratch, cs, rs.hit_out);
                    } catch (const std::exception& e) { err = std::string("tp_verify PLE: ") + e.what(); return false; }
                } else {
                    float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
                    PleOut po; po.normalized = normalized; po.result = Rt(0);
                    try {
                        ple_block(rs.ple, Rt(0), ss.ple.hist, ss.ple.w, po, ss.ple.scratch, cs);
                        ple_history_advance(ss.ple.hist, normalized, cs);
                    } catch (const std::exception& e) { err = std::string("tp_verify PLE1: ") + e.what(); return false; }
                    copy_from_mapped(rs.hist_snap, ss.ple.hist, HS, cs);
                }
            }
            // Broadcast updated R from Rank 0 -> all peers (~12 us once per window!)
            tp_broadcast_f32(rs.R, (int) (T * HC * N), *comm_, r, ++bcast_seq, cs);
            pending = false;
        }

        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out, void* mixed_q8) -> bool {
            FusedGrArgs fa[kFusedGrMaxT];
            const bool fuse_q8 = (mixed_q8 != nullptr && wu[half]->native_type == 108);
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = rs.bo + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                const int dt = wd[half]->native_type, ut = wu[half]->native_type;
                a.w_down_q8 = (dt == 8 || dt == 108) ? wd[half]->native_data : nullptr;
                a.w_up_q8 = (ut == 8 || ut == 108) ? wu[half]->native_data : nullptr;
                a.w_down_q8_soa = (dt == 108);
                a.w_up_q8_soa = (ut == 108);
                a.eps = EPS; a.lo = rs.lo + t * g.hc_lr; a.rs = rs.rs + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = rs.mixed + t * N;
                a.mixed_q8 = (t == 0 && fuse_q8) ? mixed_q8 : nullptr;
            }
            fused_gr_read_multi(fa, T, rs.xn, cs);
            return fuse_q8;
        };

        const bool q8_attn_done = gr_read_group(0, pending, rs.inj2, rs.inj, rs.xq);
        stamp(l, 1);

        // ======================= ATTENTION HALF (TP-SHARDED) =======================
        try {
            if (!is_qsa_layer(g, l)) {
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) gi * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * (uint64_t) HV * (uint64_t) g.ssm_state_size;
                float* qkv = rs.qkv_L + (size_t) gi * MT * C;
                float* hb = rs.h_L + (size_t) gi * MT * C;
                float* gate = rs.gate_L + (size_t) gi * MT * HV;
                float* beta = rs.beta_L + (size_t) gi * MT * HV;
                if (!q8_attn_done) native_quantize_q8_1(rs.mixed, rs.xq, (int) N, T, cs);
                if (wqkv->native_type == kGgmlQ8_0_SOA && wg->native_type == kGgmlQ8_0_SOA) {
                    native_q8_0_soa_mmvq_dual(wqkv->native_data, wg->native_data, rs.xq,
                                              qkv, rs.z, (int) N, (int) C, (int) ZV, T, cs);
                } else {
                    native_mmvq(wqkv->native_type, wqkv->native_data, rs.xq, qkv, (int) N, (int) C, T, cs);
                    native_mmvq(wg->native_type, wg->native_data, rs.xq, rs.z, (int) N, (int) ZV, T, cs);
                }
                stamp(l, 2);
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, T, cs, 0);
                stamp(l, 3);
                gdn_ab_multi(rs.mixed, (const uint16_t*) wa->data, (const uint16_t*) wb->data,
                             (const float*) wdt->data, (const float*) wsa->data, gate, beta, (int) N, (int) HV, T, cs);
                stamp(l, 4);
                const bool fuse_gdn_q8 = (HV <= 24);
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, rs.z, (const float*) wnm->data, EPS,
                                    rs.y, (int) HK, (int) HV, T, nullptr, cs, 0, fuse_gdn_q8 ? rs.xq : nullptr);
                stamp(l, 6);
                if (!fuse_gdn_q8) native_quantize_q8_1(rs.y, rs.xq, (int) ZV, T, cs);
                native_mmvq(wout->native_type, wout->native_data, rs.xq, rs.bo, (int) ZV, (int) N, T, cs);
            } else {
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, cs);
                    native_rope_apply(data, data, rows, cols, (int) s.n_rot, (float) qsa_freq_base(), pos, cs);
                };
                float* idx_raw = rs.idx_raw_L + (size_t) qi * MT * ID;
                if (!q8_attn_done) native_quantize_q8_1(rs.mixed, rs.xq, (int) N, T, cs);
                bf16_gemv_fp32_mmvf_multi(rs.mixed, N, (const uint16_t*) wik->data, idx_raw, ID, N, ID, T, cs);
                stamp(l, 7);
                if (wk->native_type == kGgmlQ8_0_SOA && wv->native_type == kGgmlQ8_0_SOA) {
                    native_q8_0_soa_mmvq_dual(wk->native_data, wv->native_data, rs.xq,
                                              rs.kcur, rs.vcur, (int) N, (int) (NKV * HD), (int) (NKV * HD), T, cs);
                } else {
                    native_mmvq(wk->native_type, wk->native_data, rs.xq, rs.kcur, (int) N, (int) (NKV * HD), T, cs);
                    native_mmvq(wv->native_type, wv->native_data, rs.xq, rs.vcur, (int) N, (int) (NKV * HD), T, cs);
                }
                norm_rope(rs.kcur, wkn, (int) (T * NKV), (int) HD, pos_k);
                stamp(l, 8);
                copy_from_mapped(rs.tail_snap + (size_t) qi * TS, st.idx_tail, TS, cs);
                if (st.kv_int8) {
                    kv_append_q8_multi(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, rs.step,
                                       rs.kcur, rs.vcur, T, s, cs, &st.host);
                } else {
                    for (int t = 0; t < T; ++t) {
                        const int32_t* step_t = rs.step + t * kStepCount;
                        kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t,
                                       rs.kcur + t * NKV * HD, rs.vcur + t * NKV * HD, s, cs, &st.host);
                    }
                }
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                native_qsa_indexer_append_multi(idx_raw, rs.step + kStepPos, kStepCount, T, 0,
                                                (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                (float) qsa_freq_base(), cs);
                stamp(l, 9);
                native_mmvq(wq->native_type, wq->native_data, rs.xq, rs.qfull, (int) N, (int) (NH * 2 * HD), T, cs);
                native_qsa_rms_norm_weighted_strided(rs.qfull, (int) (2 * HD), (const float*) wqn->data,
                                                     rs.qcur, (int) HD, (int) (T * NH), EPS, cs);
                native_rope_apply(rs.qcur, rs.qcur, (int) (T * NH), (int) HD, (int) s.n_rot,
                                  (float) qsa_freq_base(), rs.pos, cs);
                bf16_gemv_fp32_mmvf_multi(rs.mixed, N, (const uint16_t*) wiq->data, rs.qidx, IQ * ID, N, IQ * ID, T, cs);
                norm_rope(rs.qidx, wiqn, (int) (T * IQ), (int) ID, pos_i);
                stamp(l, 10);
                qsa_block_scores(st.idx_pooled, st.idx_dead, rs.qidx, rs.step, T, rs.max_blocks, s, rs.scores, cs);
                qsa_block_topk(rs.scores, rs.step, T, rs.max_blocks, rs.cap, s, rs.sel, cs);
                stamp(l, 11);
                qsa_kv_resolve(st, g, rs.sel, rs.step, T, rs.cap, cs);
                const QsaAttnPools pools = qsa_attn_pools(st);
                qsa_decode_attn_batch(rs.qcur, pools, rs.sel, rs.step, rs.cap, s, rs.attn_scratch, rs.attn, T, cs);
                stamp(l, 13);
                native_qsa_gate_apply_q8_1(rs.attn, rs.qfull, rs.xq, (int) (T * NH), (int) HD, cs);
                stamp(l, 14);
                native_mmvq(wo->native_type, wo->native_data, rs.xq, rs.bo, (int) (NH * HD), (int) N, T, cs);
            }
        } catch (const std::exception& e) {
            err = "tp_verify attn layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 15);
        // AllReduce #1: Attention output across all TP ranks!
        tp_allreduce_f16(rs.bo, (int) (T * N), *comm_, r, ++ar_seq, cs);
        stamp(l, 16);

        // ======================= FFN / MoE HALF (TP+EP SHARDED) =======================
        const bool q8_ffn_done = gr_read_group(1, true, rs.inj, rs.inj2, rs.nat_xq);
        const WeightRef* w_router = need(v, "ffn_gate_inp.weight", err);
        if (!w_router) return false;
        const uint32_t ring = (uint32_t) (l + 1);
        try {
            bf16_gemv_fp32_mmvf_multi(rs.mixed, N, (const uint16_t*) w_router->data, rs.logits, NE, N, NE, T, cs);
            tp_ep_route_and_plan(rs.logits, T, rs.ids, rs.w,
                                 rs.hits.d_res + l * NE,
                                 r == 0 ? (rs.d_global_owner + l * NE) : nullptr,
                                 rs.hits.cache_base, rs.slot_off_d, (long long) rs.hits.blob,
                                 rs.plan, (long long) MT * K, rs.skip_d, ring, r, cs,
                                 r == 0 ? rs.mixed : nullptr, (int) N,
                                 r == 0 ? h_x_ : nullptr, r == 0 ? h_ids_ : nullptr, r == 0 ? h_w_ : nullptr,
                                 r == 0 ? h_seq_ : nullptr, r == 0 ? (h_seq_ + 1) : nullptr, (uint32_t) l);
        } catch (const std::exception& e) {
            err = "tp_verify router layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 17);

        // Sharded Shared Expert (n_ff_local = wsg->ne1 = 160!) + shared activation quantization for routed experts
        const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                        *wsg = need(v, "ffn_gate_shexp.weight", err),
                        *wsu = need(v, "ffn_up_shexp.weight", err),
                        *wsd = need(v, "ffn_down_shexp.weight", err);
        if (!wgi || !wsg || !wsu || !wsd) return false;
        if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) ||
            !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
            !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
            return false;
        const int64_t sh_ff = wsg->ne1;
        NativeSharedWeights nsw;
        nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
        nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
        nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
        nsw.q8_1 = rs.xq;
        if (!q8_ffn_done) quantize_q8_1_rows(rs.mixed, T, N, rs.nat_xq, cs);
        if (!shared_expert_native_bf16()) f32_to_bf16_bulk(rs.mixed, rs.sh_bf16, (int64_t) T * N, cs);
        try {
            shared_expert_multi(T, rs.mixed, rs.sh_bf16, nsw, (const uint16_t*) wgi->data,
                                rs.sh_gate, rs.sh_up, rs.sh_g, rs.shared, N, sh_ff, cs, rs.nat_xq);
        } catch (const std::exception& e) {
            err = "tp_verify shared layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 18);

        // EP=TP Local VRAM Routed Experts (each GPU executes ONLY its ~2.5 owned experts!)
        const int64_t capx = MT * K;
        const int32_t* p_counts = rs.plan;
        const int32_t* p_start = rs.plan + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (rs.plan + ptr_off);
        const auto& lay = strata::kernels::cpu::expert_layout();
        const auto& f = lay.fmt[(size_t) l];
        const NativeExpertLayout L = native_expert_layout(f.gu_type, f.u_type, f.d_type, f.n_embd, f.n_ff);
        if (!native_expert_grouped(L, p_ptr, p_start, p_counts, p_dst, p_tok, (int64_t) T * K, (int64_t) T * K,
                                   rs.nat_xq, rs.hit_scratch, rs.hit_out, cs,
                                   std::numeric_limits<float>::infinity(), rs.hits.cache_soa)) {
            err = "tp_verify: native_expert_grouped failed on layer " + std::to_string(l);
            return false;
        }
        stamp(l, 20);

        // Fused partial combine + Rank-0 conditional CPU-miss wait/add + AllReduce #2 in ONE kernel!
        tp_moe_combine_miss_allreduce_f16(
            rs.hit_out, p_dst, p_counts, rs.w, rs.shared, rs.bo,
            r == 0 ? h_ymiss_sum_ : nullptr, r == 0 ? rs.skip_d : nullptr, r == 0 ? h_flag_ : nullptr, ring,
            (int) N, (int) K, T, *comm_, r, ++ar_seq, cs);
        stamp(l, 25);

        if (cvec().covers(l)) {
            cvec_apply(rs.R, l, T, HC * N, rs.bo, N, rs.inj2, HC, true, cs);
        }
    }

    // ======================= SHARDED LM HEAD (TP-SHARDED) =======================
    stamp(g.n_layers, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"),
                        *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "tp_verify: output_hc_* missing"; return false; }
        const bool apply_last = !cvec().covers(g.n_layers - 1);
        const int dt = hd->native_type, ut = hu->native_type;
        const bool fuse_head_q8 = (ut == 108);
        FusedGrArgs fa[kFusedGrMaxT];
        for (int t = 0; t < T; ++t) {
            FusedGrArgs& a = fa[t];
            a.R = Rt(t); a.R_out = Rt(t); a.apply = apply_last;
            a.bo_prev = rs.bo + t * N; a.inj_prev = rs.inj2 + t * HC;
            a.w_norm = (const float*) hn->data;
            a.w_down = (const uint16_t*) hd->data;
            a.w_up = (const uint16_t*) hu->data;
            a.w_inject = nullptr;
            a.w_down_q8 = (dt == 8 || dt == 108) ? hd->native_data : nullptr;
            a.w_up_q8 = (ut == 8 || ut == 108) ? hu->native_data : nullptr;
            a.w_down_q8_soa = (dt == 108);
            a.w_up_q8_soa = (ut == 108);
            a.eps = EPS; a.lo = rs.lo + t * g.hc_lr; a.rs = rs.rs + t * HC;
            a.inject_out = nullptr; a.mixed = rs.head_mixed + t * N;
            a.mixed_q8 = (t == 0 && fuse_head_q8) ? rs.xq : nullptr;
        }
        fused_gr_read_multi(fa, T, rs.xn, cs);
        if (!fuse_head_q8) native_quantize_q8_1(rs.head_mixed, rs.xq, (int) N, T, cs);
        native_mmvq(rs.head->type(), rs.head->weights(), rs.xq, rs.head_logits_local,
                    (int) N, rs.head->n_out(), T, cs);
        tp_lmhead_local_topk_if(rs.head_logits_local, rs.head->n_out(), rs.head->row_offset(), T,
                                *comm_, r, cs);
        tp_lmhead_greedy_merge(rs.head_logits_local, rs.head->n_out(), rs.head->row_offset(), T,
                               *comm_, r, 1u, r == 0 ? h_out_ : nullptr, cs);
    }
    stamp(g.n_layers, 1);
    if (r == 0) signal_done_flag(h_seq_ + 2, 1u, cs);
    return true;
}

bool TpVerifier::capture_rank(int r, int T, std::string& err) {
    RankState& rs = r_[r];
    if (rs.exec[T] != nullptr) return true;
    const OnDevice on_dev(rs.dev);
    if (cudaStreamBeginCapture(rs.cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "tp_verify: begin capture failed";
        return false;
    }
    std::string rerr;
    const bool ok = record_window_rank(r, T, rs.cs, rerr);
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(rs.cs, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        err = rerr;
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&rs.exec[T], graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("tp_verify: instantiate: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    (void) cudaGraphUpload(rs.exec[T], rs.cs);
    (void) cudaStreamSynchronize(rs.cs);
    if (r == 0) std::fprintf(stderr, "maya tp-verify: captured TP=%d %d-token window graphs\n", tp_, T);
    return true;
}

bool TpVerifier::capture_commit_rank(int r, std::string& err) {
    using namespace strata::kernels;
    RankState& rs = r_[r];
    if (rs.commit_exec != nullptr) return true;
    const OnDevice on_dev(rs.dev);
    const ModelGeometry& g = rs.g;
    SessionState& ss = *rs.ss;
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, HK = g.ssm_k_heads, ID = g.idx_key_dim, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * (uint64_t) HV * (uint64_t) g.ssm_state_size +
                                (uint64_t) C * (uint64_t) (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;

    if (cudaStreamBeginCapture(rs.cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "tp_verify: commit begin capture failed";
        return false;
    }
    copy_i32_from_mapped(rs.commit, h_commit_, 2 + MT, rs.cs);
    int64_t gi = 0, qi = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const LayerView v(*rs.wt, l);
        if (!is_qsa_layer(g, l)) {
            const WeightRef* wnm = need(v, "ssm_norm.weight", err);
            if (!wnm) { cudaGraph_t g0 = nullptr; cudaStreamEndCapture(rs.cs, &g0); if (g0) cudaGraphDestroy(g0); return false; }
            float* state = ss.gdn_state + (size_t) gi * gdn_floats;
            float* conv = state + (uint64_t) g.ssm_state_size * (uint64_t) HV * (uint64_t) g.ssm_state_size;
            gdn_conv_commit(conv, rs.qkv_L + (size_t) gi * MT * C, (int) C, rs.commit, rs.cs);
            gdn_step_norm_multi(state, rs.h_L + (size_t) gi * MT * C, (int) C,
                                rs.gate_L + (size_t) gi * MT * HV, rs.beta_L + (size_t) gi * MT * HV,
                                rs.z, (const float*) wnm->data, EPS, rs.y_dummy, (int) HK, (int) HV,
                                (int) MT, rs.commit, rs.cs);
            ++gi;
        } else {
            const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
            if (!wikn) { cudaGraph_t g0 = nullptr; cudaStreamEndCapture(rs.cs, &g0); if (g0) cudaGraphDestroy(g0); return false; }
            const QsaState& st = ss.qsa_states[qi];
            copy_from_mapped(st.idx_tail, rs.tail_snap + (size_t) qi * TS, TS, rs.cs);
            const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
            native_qsa_indexer_append_multi(rs.idx_raw_L + (size_t) (qi * MT) * ID, rs.commit + 2, 1, max_t_, 0,
                                            (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                            (float) qsa_freq_base(), rs.cs);
            ++qi;
        }
    }
    if (r == 0 && ss.ple.ready()) {
        copy_indexed(ss.ple.hist, rs.hist_snap, HS, rs.commit + 1, HS, rs.cs);
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(rs.cs, &graph);
    if (ce != cudaSuccess || cudaGraphInstantiate(&rs.commit_exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("tp_verify: commit instantiate: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    (void) cudaGraphUpload(rs.commit_exec, rs.cs);
    (void) cudaStreamSynchronize(rs.cs);
    return true;
}

bool TpVerifier::run(int T, const int32_t* tokens, int64_t pos0, TpPoolMissFn pool, void* user,
                     int32_t* out, std::string& err) {
    using namespace strata::kernels;
    if (T < 1 || T > max_t_) { err = "tp_verify: invalid T"; return false; }
    if (pos0 + T > r_[0].ss->qsa_states[0].max_cells) { err = "tp_verify: window runs past context"; return false; }

    for (int r = 0; r < tp_; ++r) {
        if (!capture_rank(r, T, err) || !capture_commit_rank(r, err)) return false;
    }

    // Scoped host-thread pin to physical core 0 (reserved by ExpertPool::physical_cores(true)) ONLY during
    // the active verify window so child threads (xcache, adapt, CUDA driver) NEVER inherit a 1-core mask!
    static const int host_pin_core = [] {
        const std::vector<int> pc = strata::kernels::cpu::physical_cores(false);
        return pc.empty() ? -1 : pc[0];
    }();
    struct ScopedHostPin {
        long long prev = -1;
        explicit ScopedHostPin(int c) { if (c >= 0) prev = strata::kernels::cpu::pin_current_thread(c); }
        ~ScopedHostPin() { if (prev > 0) strata::kernels::cpu::restore_thread_affinity(prev); }
    } scoped_host_pin(host_pin_core);

    const Clock::time_point th0 = Clock::now();
    const ModelGeometry& g0 = r_[0].g;
    const QsaShapes s = shapes_of(g0);
    const int64_t NH = g0.n_head, NKV = g0.n_head_kv, IQ = g0.idx_q_heads;
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        last_tokens_[t] = tokens[t];
        qsa_step_fill(h_step_ + (size_t) t * kStepCount, pos0 + t, s);
        for (int64_t h = 0; h < NH; ++h) h_pos_[t * NH + h] = (int32_t) (pos0 + t);
        int32_t* pk = h_pos_ + (size_t) max_t_ * NH;
        int32_t* pi = pk + (size_t) max_t_ * NKV;
        for (int64_t h = 0; h < NKV; ++h) pk[t * NKV + h] = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < IQ; ++h) pi[t * IQ + h] = (int32_t) (pos0 + t);
    }
    if (r_[0].ss->ple.ready()) {
        uint32_t rows[kVerifyMaxT * PLE_N_HEADS];
        ngram_rows(tokens, r_[0].ss->ple_prev, T, r_[0].ss->ple.consts, rows);
        if (!r_[0].ss->ple.table->gather_batch(rows, T, h_ple_, err)) return false;
    }

    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    const bool fast_topk = head_sampling_ && sampled && hist_d_ == nullptr && sampling_.logit_bias == nullptr;
    const int k_eff = (sampling_.top_k > 0 && sampling_.top_k <= 64) ? sampling_.top_k : 64;

    tp_comm_reset_flags(*comm_);
    if (comm_->h_topk_k) *(volatile int32_t*) comm_->h_topk_k = fast_topk ? k_eff : 0;
    ((volatile uint32_t*) h_seq_)[0] = 0;
    ((volatile uint32_t*) h_seq_)[1] = 0;
    ((volatile uint32_t*) h_seq_)[2] = 0;
    ((volatile uint32_t*) h_seq_)[3] = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = T;
    last_pos0_ = pos0;
    ms_host += ms_since(th0);

    // Launch all TP graphs concurrently across dedicated per-rank host threads!
    // Use async launch so Rank 0 enters the doorbell poll loop immediately while ranks 1..tp-1 finish submit.
    dispatch_ranks_async(1, T, 0);
    if (launch_.errs[0] != cudaSuccess) {
        dispatch_ranks_join();
        err = "tp_verify: launch rank 0: " + std::string(cudaGetErrorString(launch_.errs[0]));
        return false;
    }

    // Host spin-poll for Rank 0 conditional CPU miss doorbells + completion
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const miss_l = h_seq_ + 1;
    volatile uint32_t* const done = h_seq_ + 2;
    volatile uint32_t* const flag = h_flag_;
    uint32_t served = 0;
    uint32_t spins = 0;
    const Clock::time_point tw0 = Clock::now();
    for (;;) {
        if (*seq > served) {
            std::atomic_thread_fence(std::memory_order_acquire);
            const uint32_t l = *miss_l;
            const uint32_t want = l + 1u;
            const Clock::time_point tp0 = Clock::now();
            if (pool != nullptr) {
                pool(user, h_x_, h_ids_, h_w_, T, r_[0].ss->k, h_ymiss_sum_, (int64_t) l);
            } else {
                std::memset(h_ymiss_sum_, 0, (size_t) T * (size_t) g_full_.n_embd * sizeof(float));
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
            strata_store_fence();
            *flag = want;
            ms_pool += ms_since(tp0);
            ++served;
            continue;
        }
        if (*done != 0) break;
        if ((++spins & 16383u) == 0) {
            const OnDevice on_dev(r_[0].dev);
            const cudaError_t q = cudaStreamQuery(r_[0].cs);
            if (q != cudaErrorNotReady && q != cudaSuccess && *done == 0 && *seq <= served) {
                err = std::string("tp_verify: rank 0 stream error: ") + cudaGetErrorString(q);
                return false;
            }
        }
        strata_cpu_pause();
    }
    ms_wait += ms_since(tw0);
    dispatch_ranks_join();
    for (int r = 1; r < tp_; ++r) {
        if (launch_.errs[r] != cudaSuccess) {
            err = "tp_verify: launch rank " + std::to_string(r) + ": " + cudaGetErrorString(launch_.errs[r]);
            return false;
        }
    }

    const bool need_full_logits = head_sampling_ && !fast_topk && (sampled || hist_d_ != nullptr);
    if (prof_on_ || need_full_logits) {
        for (int r = 0; r < tp_; ++r) {
            const OnDevice on_dev(r_[r].dev);
            const cudaError_t se = cudaStreamSynchronize(r_[r].cs);
            if (se != cudaSuccess) {
                err = "tp_verify: sync rank " + std::to_string(r) + ": " + cudaGetErrorString(se);
                return false;
            }
        }
    }

    if (prof_on_ && r_[0].prof != nullptr) {
        const OnDevice on_dev(r_[0].dev);
        cudaMemcpy(r_[0].prof_h.data(), r_[0].prof, r_[0].prof_h.size() * 8, cudaMemcpyDeviceToHost);
        const bool dump_layers = (prof_windows_ == 5 && std::getenv("MAYA_VERIFY_PROFILE_LAYERS") != nullptr);
        for (int64_t l = 0; l < g_full_.n_layers; ++l) {
            const int kind = is_qsa_layer(g_full_, l) ? 1 : 0;
            const unsigned long long* p = r_[0].prof_h.data() + l * kProfPer;
            unsigned long long prev = p[0];
            double hc0 = 0, ar1 = 0, wcpu = 0, ar2 = 0;
            for (int i = 1; i < 26; ++i) {
                if (p[i] > prev) {
                    const double dt = (double) (p[i] - prev);
                    prof_sum_[kind][i] += dt;
                    if (i == 1) hc0 = dt / 1e3;
                    if (i == 16) ar1 = dt / 1e3;
                    if (i == 23) wcpu = dt / 1e3;
                    if (i == 25) ar2 = dt / 1e3;
                    prev = p[i];
                }
            }
            if (dump_layers) {
                std::fprintf(stderr, "  L%02lld (%s): hc0=%.1fus ar1=%.1fus wcpu=%.1fus ar2=%.1fus total=%.1fus\n",
                             (long long) l, kind ? "QSA" : "GDN", hc0, ar1, wcpu, ar2, (double) (prev - p[0]) / 1e3);
            }
        }
        const unsigned long long* ph = r_[0].prof_h.data() + g_full_.n_layers * kProfPer;
        if (ph[1] > ph[0]) prof_sum_[0][26] += (double) (ph[1] - ph[0]);
        ++prof_windows_;
    }

    std::atomic_thread_fence(std::memory_order_acquire);
    if (fast_topk) {
        // Exact 2-microsecond host sampling directly on the in-graph top-k candidates (ZERO extra GPU launch/sync!).
        const int n_cand = tp_ * k_eff;
        const float inv_t = 1.0f / std::max(1e-6f, sampling_.temperature);
        const float min_p = sampling_.min_p;
        const float top_p = sampling_.top_p;
        auto splitmix64 = [](uint64_t x) -> uint64_t {
            x += 0x9e3779b97f4a7c15ULL;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            return x ^ (x >> 31);
        };
        struct Cand { float v; int32_t id; };
        Cand buf[256];
        for (int t = 0; t < T; ++t) {
            const float* tv = comm_->h_topk_v + (size_t) t * (size_t) n_cand;
            const int32_t* ti = comm_->h_topk_i + (size_t) t * (size_t) n_cand;
            for (int i = 0; i < n_cand; ++i) buf[i] = {tv[i], ti[i]};
            std::partial_sort(buf, buf + k_eff, buf + n_cand, [](const Cand& a, const Cand& b) {
                return a.v > b.v || (a.v == b.v && a.id < b.id);
            });
            const float max_v = buf[0].v;
            float probs[64];
            float sum = 0.0f;
            for (int i = 0; i < k_eff; ++i) {
                const float p = std::exp((buf[i].v - max_v) * inv_t);
                probs[i] = p;
                sum += p;
            }
            const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
            int keep = k_eff;
            if (min_p > 0.0f) {
                const float thresh = probs[0] * min_p;
                for (int i = 1; i < keep; ++i) {
                    if (probs[i] < thresh) { keep = i; break; }
                }
            }
            if (top_p > 0.0f && top_p < 1.0f) {
                float cum = 0.0f;
                for (int i = 0; i < keep; ++i) {
                    cum += probs[i] * inv_sum;
                    if (cum >= top_p) { keep = i + 1; break; }
                }
            }
            float kept_sum = 0.0f;
            for (int i = 0; i < keep; ++i) kept_sum += probs[i];
            const uint64_t h = splitmix64(sampling_.seed ^ ((uint64_t) (pos0 + t) * 0x9e3779b97f4a7c15ULL));
            const float u = (float) ((h >> 11) * (1.0 / 9007199254740992.0)) * kept_sum;
            float acc = 0.0f;
            int pick = keep - 1;
            for (int i = 0; i < keep; ++i) {
                acc += probs[i];
                if (u < acc) { pick = i; break; }
            }
            out[t] = buf[pick].id;
        }
    } else if (need_full_logits) {
        for (int r = 0; r < tp_; ++r) {
            const OnDevice on_dev(r_[r].dev);
            tp_lmhead_scatter_logits(r_[r].head_logits_local, comm_->h_logits,
                                     r_[r].head->n_out(), r_[r].head->row_offset(),
                                     (int) n_vocab_, T, r_[r].cs);
        }
        for (int r = 0; r < tp_; ++r) {
            const OnDevice on_dev(r_[r].dev);
            cudaStreamSynchronize(r_[r].cs);
        }
        const OnDevice on_dev(r_[0].dev);
        cudaMemcpyAsync(r_[0].head_logits_full, comm_->h_logits,
                        (size_t) T * (size_t) n_vocab_ * sizeof(float), cudaMemcpyHostToDevice, r_[0].cs);
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        sample_tokens(r_[0].head_logits_full, T, (int) n_vocab_, hist_d_, hist_len_, sp, h_out_, r_[0].cs);
        if (cudaStreamSynchronize(r_[0].cs) != cudaSuccess) {
            err = "tp_verify: head sampling failed";
            return false;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    } else {
        for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    }
    ++windows;
    return true;
}

bool TpVerifier::commit_begin(int n_keep, std::string& err) {
    if (n_keep < 1 || n_keep > last_t_) { err = "tp_verify: invalid n_keep"; return false; }
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // Async dispatch for ranks 1..tp-1 so their cudaGraphLaunch overlaps completely with mtp.draft on GPU 0!
    dispatch_ranks_async(2, 0, 0);
    if (launch_.errs[0] != cudaSuccess) {
        dispatch_ranks_join();
        err = "tp_verify: commit launch rank 0: " + std::string(cudaGetErrorString(launch_.errs[0]));
        return false;
    }
    commit_pending_ = true;
    for (int t = 0; t < n_keep; ++t) {
        r_[0].ss->ple_prev[0] = r_[0].ss->ple_prev[1];
        r_[0].ss->ple_prev[1] = last_tokens_[t];
    }
    return true;
}

bool TpVerifier::commit_wait(std::string& err) {
    if (!commit_pending_) return true;
    dispatch_ranks_join();
    for (int r = 1; r < tp_; ++r) {
        if (launch_.errs[r] != cudaSuccess) {
            commit_pending_ = false;
            err = "tp_verify: commit launch rank " + std::to_string(r) + ": " + cudaGetErrorString(launch_.errs[r]);
            return false;
        }
    }
    for (int r = 0; r < tp_; ++r) {
        const OnDevice on_dev(r_[r].dev);
        const cudaError_t se = cudaStreamSynchronize(r_[r].cs);
        if (se != cudaSuccess) {
            commit_pending_ = false;
            err = "tp_verify: commit sync rank " + std::to_string(r) + ": " + cudaGetErrorString(se);
            return false;
        }
    }
    commit_pending_ = false;
    return true;
}

bool TpVerifier::commit(int n_keep, std::string& err) {
    const Clock::time_point t0 = Clock::now();
    if (!commit_begin(n_keep, err)) return false;
    if (!commit_wait(err)) return false;
    ms_commit += ms_since(t0);
    return true;
}

bool TpVerifier::read_head_logits(int rows, float* host, std::string& err) {
    if (rows < 0 || rows > max_t_) { err = "tp_verify: invalid rows"; return false; }
    if (rows == 0) return true;
    for (int r = 0; r < tp_; ++r) {
        const OnDevice on_dev(r_[r].dev);
        strata::kernels::tp_lmhead_scatter_logits(r_[r].head_logits_local, comm_->h_logits,
                                                  r_[r].head->n_out(), r_[r].head->row_offset(),
                                                  (int) n_vocab_, rows, r_[r].cs);
    }
    for (int r = 0; r < tp_; ++r) {
        const OnDevice on_dev(r_[r].dev);
        if (cudaStreamSynchronize(r_[r].cs) != cudaSuccess) {
            err = "tp_verify: read_head_logits sync failed";
            return false;
        }
    }
    std::memcpy(host, comm_->h_logits, (size_t) rows * (size_t) n_vocab_ * sizeof(float));
    return true;
}

std::string TpVerifier::profile_report() {
    if (!prof_on_ || prof_windows_ == 0) return std::string();
    static const char* names[kProfPer] = {
        "-", "hc-read0", "q8+qkv/z", "conv", "ab", "", "rec", "q8+k-idx",
        "k/v+rope", "kv+idx-app", "q+q-idx", "scores+topk", "", "attention", "gate", "attn-out",
        "AR1-attn", "hc1+router", "shared+quant", "", "VRAM-ep", "", "", "waitCPU", "combine+miss",
        "AR2-moe", "head", "", "", "", "", ""
    };
    std::string out;
    char b[96];
    double total = 0;
    for (int k = 0; k < 2; ++k) {
        out += k == 0 ? " GDN:" : " | QSA:";
        for (int i = 0; i < kProfPer; ++i) {
            if (prof_sum_[k][i] <= 0) continue;
            total += prof_sum_[k][i];
            std::snprintf(b, sizeof b, " %s %.2f", names[i], prof_sum_[k][i] / 1e6 / (double) prof_windows_);
            out += b;
        }
    }
    std::snprintf(b, sizeof b, " | total %.2f ms/win (%lld wins)", total / 1e6 / (double) prof_windows_, (long long) prof_windows_);
    out += b;
    for (auto& row : prof_sum_) for (double& d : row) d = 0;
    prof_windows_ = 0;
    return out;
}

}  // namespace strata::core
