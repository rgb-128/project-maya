// src/core/mtp.cpp - see include/strata/core/mtp.hpp.
#include "strata/core/mtp.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/on_device.hpp"

#include "strata/core/native_head.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/platform/cpu_relax.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <vector>

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
constexpr int GGML_Q8_0 = 8;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
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

// 64-bit seek/tell on a `FILE*`: `fseek`/`ftell` take a 32-bit `long` on Windows and would wrap past 2 GiB.
#if defined(_WIN32)
#define STRATA_FILE_SEEK64(f, o, w) _fseeki64((f), (long long) (o), (w))
#define STRATA_FILE_TELL64(f) _ftelli64(f)
#else
#define STRATA_FILE_SEEK64(f, o, w) fseeko((f), (off_t) (o), (w))
#define STRATA_FILE_TELL64(f) ftello(f)
#endif

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    // Loader fix (0.1.15+loaderfix.2): `ifstream::read` reaches the disk as 4095-byte reads on MSVC - the same
    // split `load_experts_ranges` had - so the drafter's 111 MiB dense blob paid 4 KiB per operation on a cold
    // start.  `fread` passes a request bigger than the stream buffer straight to `ReadFile()`.
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    // A `FILE*` has no destructor and the short-read path below returns early: the guard closes on every path.
    struct Closer {
        FILE* f;
        ~Closer() { if (f != nullptr) std::fclose(f); }
    } closer{f};
    if (STRATA_FILE_SEEK64(f, 0, SEEK_END) != 0) return false;
    const long long n = (long long) STRATA_FILE_TELL64(f);
    if (n < 0 || STRATA_FILE_SEEK64(f, 0, SEEK_SET) != 0) return false;
    out.resize((size_t) n);
    return n == 0 || std::fread(out.data(), 1, (size_t) n, f) == (size_t) n;
}

}  // namespace

MtpDrafter::~MtpDrafter() {
    if (cs_) cudaStreamSynchronize(cs_);
    for (auto& e : prefill_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : prefill_dev_exec_) if (e) cudaGraphExecDestroy(e);
    if (pf_dev_) cudaFree(pf_dev_);
    for (auto& e : round_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : step_exec_) if (e) cudaGraphExecDestroy(e);
    if (cs_) cudaStreamDestroy(cs_);
    if (dense_) cudaFree(dense_);
    if (hc_q8_soa_) cudaFree(hc_q8_soa_);
    if (experts_ && !experts_arena_.ok()) cudaFree(experts_);
    experts_arena_.close();
    if (state_arena_) cudaFree(state_arena_);
    if (arena_) cudaFree(arena_);
    if (head_logits_) cudaFree(head_logits_);
    if (dhead_) cudaFree(dhead_);
    if (dvocab_) cudaFree(dvocab_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_row_, h_out_, h_prob_, h_done_};
    for (void* h : hosts) if (h) cudaFreeHost(h);
}

const float* MtpDrafter::f32(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "f32") return (const float*) (dense_ + t.off);
    return nullptr;
}
const uint16_t* MtpDrafter::bf16(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "bf16") return (const uint16_t*) (dense_ + t.off);
    return nullptr;
}
const void* MtpDrafter::q8(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "q8_0") return dense_ + t.off;
    return nullptr;
}

bool MtpDrafter::lend_experts(uint64_t bytes, uint64_t& given, std::string& err) {
    given = 0;
    if (!experts_arena_.ok()) { err = "the draft layer's experts are one allocation here"; return false; }
    const OnDevice on_device(device_);
    if (cs_) cudaStreamSynchronize(cs_);
    uint64_t from = 0;
    return experts_arena_.lend_tail(bytes, 0, from, given, err);
}

bool MtpDrafter::reclaim_experts(std::string& err) {
    if (!experts_arena_.lent()) return true;
    const OnDevice on_device(device_);
    const uint64_t from = experts_arena_.lent_from();
    if (!experts_arena_.reclaim_tail(err)) return false;
    // the lent part's bytes are gone: read again from the file they came from (the same bytes as at load)
    FILE* f = std::fopen((rt_dir_ + "/experts.bin").c_str(), "rb");
    bool ok = f != nullptr;
    if (ok && from < experts_bytes_) {
#if defined(_WIN32)
        ok = _fseeki64(f, (long long) from, SEEK_SET) == 0;
#else
        ok = fseeko(f, (off_t) from, SEEK_SET) == 0;
#endif
        std::vector<uint8_t> chunk(64u << 20);
        for (uint64_t off = from; ok && off < experts_bytes_;) {
            const uint64_t n = std::min<uint64_t>(chunk.size(), experts_bytes_ - off);
            ok = std::fread(chunk.data(), 1, (size_t) n, f) == (size_t) n &&
                 cudaMemcpy(experts_ + off, chunk.data(), n, cudaMemcpyHostToDevice) == cudaSuccess;
            off += n;
        }
    }
    if (f != nullptr) std::fclose(f);
    if (!ok) {
        err = "the draft layer's experts could not be read again from " + rt_dir_ + "/experts.bin";
        return false;
    }
    if (experts_soa_ && from < experts_bytes_) {
        const int64_t blob = strata::kernels::cpu::BLOB;
        const int64_t e0 = (int64_t) (from / (uint64_t) blob);
        const int64_t n_e = (int64_t) (experts_bytes_ / (uint64_t) blob) - e0;
        if (n_e > 0) strata::kernels::s2_repack_experts_soa(experts_ + (size_t) e0 * (size_t) blob, n_e, blob, cs_);
    }
    return true;
}

bool MtpDrafter::load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
                      int64_t window) {
    cudaGetDevice(&device_);   // a layer split's last stage on another GPU: the drafter lives there
    g_ = &g;
    ss_ = &ss;
    max_t_ = max_t;
    rt_dir_ = rt_dir;
    if (max_t < 1 || max_t > strata::kernels::kVerifyMaxT) { err = "mtp: max_t out of range"; return false; }
    // Loader fix (0.1.15+loaderfix.2): the two reads below are the whole “drafter files” cost; reporting
    // them apart from the rest of the stage is what makes the next regression visible.
    const auto t_files = std::chrono::steady_clock::now();
    // ---- the index and the dense weights
    {
        std::ifstream idx(rt_dir + "/dense.txt");
        if (!idx) { err = "mtp: cannot open " + rt_dir + "/dense.txt (run tools/mtp_rt.py)"; return false; }
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            Tensor t;
            is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
            if (!is) { err = "mtp: malformed dense.txt line: " + line; return false; }
            tensors_.push_back(t);
        }
        std::vector<uint8_t> blob;
        if (!read_file(rt_dir + "/dense.bin", blob)) { err = "mtp: cannot read dense.bin"; return false; }
        if (strata::kernels::p100_soa_enabled()) {
            bool all_ok = true;
            for (const auto& t : tensors_) {
                if (t.kind != "q8_0") continue;
                if ((t.rows % 32) != 0 || (t.cols % 32) != 0 || t.off + t.bytes > blob.size()) {
                    all_ok = false;
                    break;
                }
            }
            if (all_ok) {
                std::vector<uint8_t> tmp;
                for (const auto& t : tensors_) {
                    if (t.kind != "q8_0") continue;
                    tmp.resize((size_t) t.bytes);
                    const int rt = strata::kernels::native_repack_to_soa_host(
                        GGML_Q8_0, blob.data() + t.off, tmp.data(), (int) t.cols, (int) t.rows, false);
                    if (rt != strata::kernels::kGgmlQ8_0_SOA) {
                        all_ok = false;
                        break;
                    }
                    std::memcpy(blob.data() + t.off, tmp.data(), (size_t) t.bytes);
                }
                if (all_ok) q8_type_ = strata::kernels::kGgmlQ8_0_SOA;
            }
        }
        const cudaError_t alloc = cudaMalloc((void**) &dense_, blob.size());
        if (alloc != cudaSuccess) {
            size_t free_bytes = 0, total_bytes = 0;
            const cudaError_t info = cudaMemGetInfo(&free_bytes, &total_bytes);
            err = "mtp: dense weights allocation failed (" + std::string(cudaGetErrorString(alloc)) +
                  "), requested " + std::to_string(blob.size() >> 20) + " MiB, CUDA0 free " +
                  (info == cudaSuccess ? std::to_string(free_bytes >> 20) + " MiB" : "unknown");
            return false;
        }
        cudaMemcpy(dense_, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        vram_ += blob.size();

        // GP100 optimization: quantize the 6 BF16 hyper-connection matrices (39.3 MiB) to Q8_0_SOA (20.9 MiB)
        // so every draft step uses half the HBM2 bandwidth and fast HFMA2 gr_down/gr_up kernels (zero effect on
        // verified output accuracy; dense_ retains original BF16 for single-GPU Prefill::draft_kv).
        if (strata::kernels::p100_soa_enabled()) {
            auto f32_to_f16_bits = [](float v) -> uint16_t {
                uint32_t x;
                std::memcpy(&x, &v, 4);
                const uint32_t sign = (x >> 16) & 0x8000u;
                int32_t exp = (int32_t) ((x >> 23) & 0xffu) - 127 + 15;
                uint32_t mant = x & 0x7fffffu;
                if (exp <= 0) return (uint16_t) sign;
                if (exp >= 31) return (uint16_t) (sign | 0x7bffu);
                const uint32_t round_bit = (mant >> 12) & 1u;
                uint32_t h = sign | ((uint32_t) exp << 10) | (mant >> 13);
                if (round_bit && ((mant & 0xfffu) != 0 || (h & 1u) != 0)) ++h;
                return (uint16_t) h;
            };
            const char* hc_names[6] = {
                "attn_hyper_connection.input_mix_weight_down.weight",
                "attn_hyper_connection.input_mix_weight_up.weight",
                "mlp_hyper_connection.input_mix_weight_down.weight",
                "mlp_hyper_connection.input_mix_weight_up.weight",
                "hyper_connection_mixer.input_mix_weight_down.weight",
                "hyper_connection_mixer.input_mix_weight_up.weight"
            };
            const Tensor* hct[6] = {};
            bool hc_ok = true;
            size_t hc_total = 0;
            size_t hc_off[6] = {};
            for (int k = 0; k < 6; ++k) {
                for (const auto& t : tensors_) if (t.name == hc_names[k] && t.kind == "bf16") { hct[k] = &t; break; }
                if (!hct[k] || (hct[k]->rows % 32) != 0 || (hct[k]->cols % 32) != 0 ||
                    hct[k]->off + hct[k]->bytes > blob.size()) {
                    hc_ok = false; break;
                }
                hc_off[k] = hc_total;
                const size_t q8b = strata::kernels::native_mmvq_weight_bytes(GGML_Q8_0, (int) hct[k]->cols, (int) hct[k]->rows);
                hc_total += (q8b + 255) & ~(size_t) 255;
            }
            if (hc_ok && cudaMalloc((void**) &hc_q8_soa_, hc_total) == cudaSuccess) {
                std::vector<uint8_t> h_pack(hc_total, 0);
                std::vector<uint8_t> q8_tmp;
                for (int k = 0; k < 6 && hc_ok; ++k) {
                    const int64_t rows = hct[k]->rows, cols = hct[k]->cols;
                    const int64_t nblk = rows * (cols / 32);
                    q8_tmp.resize((size_t) nblk * 34);
                    const uint16_t* src = reinterpret_cast<const uint16_t*>(blob.data() + hct[k]->off);
                    for (int64_t b = 0; b < nblk; ++b) {
                        const uint16_t* bp = src + b * 32;
                        float vals[32];
                        float amax = 0.0f;
                        for (int j = 0; j < 32; ++j) {
                            const uint32_t u = (uint32_t) bp[j] << 16;
                            std::memcpy(&vals[j], &u, 4);
                            const float a = std::fabs(vals[j]);
                            if (a > amax) amax = a;
                        }
                        const float d = amax / 127.0f;
                        const float id = d > 0.0f ? (1.0f / d) : 0.0f;
                        uint8_t* dst_b = q8_tmp.data() + (size_t) b * 34;
                        const uint16_t dh = f32_to_f16_bits(d);
                        std::memcpy(dst_b, &dh, 2);
                        int8_t* qs = reinterpret_cast<int8_t*>(dst_b + 2);
                        for (int j = 0; j < 32; ++j) {
                            const int q = (int) std::lround(vals[j] * id);
                            qs[j] = (int8_t) std::clamp(q, -127, 127);
                        }
                    }
                    const int rt = strata::kernels::native_repack_to_soa_host(
                        GGML_Q8_0, q8_tmp.data(), h_pack.data() + hc_off[k], (int) cols, (int) rows, false);
                    const int want_rt = strata::kernels::kGgmlQ8_0_SOA;
                    if (rt != want_rt) hc_ok = false;
                }
                if (hc_ok && cudaMemcpy(hc_q8_soa_, h_pack.data(), hc_total, cudaMemcpyHostToDevice) == cudaSuccess) {
                    for (int k = 0; k < 6; ++k) hc_q8_ptr_[k] = hc_q8_soa_ + hc_off[k];
                    vram_ += hc_total;
                } else {
                    cudaFree(hc_q8_soa_);
                    hc_q8_soa_ = nullptr;
                }
            }
        }
    }
    // ---- the 512 routed experts, one blob each
    {
        const uint64_t bytes = (uint64_t) g.n_expert * strata::kernels::cpu::BLOB;
        // Loader fix (0.1.15+loaderfix.2): each 64 MiB read below reached the disk as ~16k 4095-byte reads under
        // MSVC's `basic_filebuf::xsgetn`, which is what made 675 MiB of drafter experts take minutes.
        FILE* f = std::fopen((rt_dir + "/experts.bin").c_str(), "rb");
        if (f == nullptr) { err = "mtp: cannot open experts.bin"; return false; }
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        // lendable (an on-demand vision encoder): mapped in pieces, else (no virtual memory API) one cudaMalloc
        std::string lerr;
        if (want_lend_ && experts_arena_.open(bytes, lerr)) experts_ = experts_arena_.data();
        else if (want_lend_)
            std::fprintf(stderr, "maya mtp: the draft layer's experts cannot be mapped in pieces on this GPU (%s)\n",
                         lerr.c_str());
        if (experts_ == nullptr && cudaMalloc((void**) &experts_, bytes) != cudaSuccess) {
            experts_ = nullptr;
            err = "mtp: the 512 experts do not fit in VRAM";
            return false;
        }
        experts_bytes_ = bytes;
        std::vector<uint8_t> chunk(64u << 20);
        for (uint64_t off = 0; off < bytes;) {
            const uint64_t n = std::min<uint64_t>(chunk.size(), bytes - off);
            if (std::fread(chunk.data(), 1, (size_t) n, f) != (size_t) n) { err = "mtp: experts.bin is truncated"; return false; }
            cudaMemcpy(experts_ + off, chunk.data(), n, cudaMemcpyHostToDevice);
            off += n;
        }
        if (strata::kernels::p100_soa_enabled()) {
            strata::kernels::s2_repack_experts_soa(experts_, g.n_expert, strata::kernels::cpu::BLOB, nullptr);
            experts_soa_ = true;
        }
        vram_ += bytes;
    }
    const char* required[] = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
                              "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"};
    for (const char* n : required) if (!q8(n)) { err = std::string("mtp: ") + n + " is missing (q8_0)"; return false; }

    // ---- the layer's own K/V (dense attention: no indexer state is read)
    const strata::kernels::QsaShapes s = shapes_of(g);
    const int64_t max_cells = ss.qsa_states[0].max_cells;
    // KV streaming: the drafter only reads its last `window` cells, so with streaming on its K/V is a ring of the
    // window (plus the cells a round writes ahead of its queries) over a host copy, refilled on a resume. The host copy
    // is pinned after the expert arena has pinned what it could: if it does not fit, the K/V stays whole in VRAM.
    int64_t ring = (window > 0 && window < max_cells) ? window + 4 * (int64_t) max_t + 64 : 0;
    // K8V4 never applies to the drafter: its own attention paths (below, and verify.cpp) handle whole formats
    // only, whatever ring shape it takes (0, a window, or the -1 fully-resident fallback).
    const bool kv_hybrid_was = qsa_kv_hybrid();
    const bool kv_int8_was = qsa_kv_int8();
    qsa_set_kv_hybrid(false);
    if (kv_hybrid_was) qsa_set_kv_int8(true);   // the drafter under --kv k8v4: plain INT8
    uint64_t sb = qsa_state_bytes(g, max_cells, false, ring);
    if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
    if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[0], ring) == 0) {
        if (st_.kv_mode == 0) { err = "mtp: state init failed"; return false; }
        std::fprintf(stderr, "maya mtp: no pinned RAM left for the draft layer's K/V copy; keeping it in VRAM\n");
        cudaGetLastError();
        cudaFree(state_arena_);
        st_ = QsaState{};
        ring = -1;   // fully resident
        sb = qsa_state_bytes(g, max_cells, false, ring);
        if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
        if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[0], ring) == 0) { err = "mtp: state init failed"; return false; }
    }
    qsa_set_kv_int8(kv_int8_was);
    qsa_set_kv_hybrid(kv_hybrid_was);
    qsa_state_zero(st_, g, nullptr);
    cudaDeviceSynchronize();
    vram_ += sb;

    // ---- buffers
    window_ = (window > 0 && window < max_cells) ? window : 0;
    cap_ = (((window_ > 0 ? window_ : max_cells) + 63) / 64) * 64;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);
    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t R2 = 2 * T;   // step/pos rows: T catch-up rows + up to T-2 chain steps
    bool ok = mapped(T * 4 + 64, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(R2 * 4 * 4 + 64, (void**) &h_step_, (void**) &m_step_) &&
              mapped(R2 * NH * 4 + 64, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(64, (void**) &h_row_, (void**) &m_row_) &&
              mapped(T * 4 + 64, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 64, (void**) &h_prob_, (void**) &m_prob_) &&
              mapped(64, (void**) &h_done_, (void**) &m_done_);
    if (!ok) { err = "mtp: mapped staging failed"; return false; }
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(R2 * 4); pos_ = b.take<int32_t>(R2 * NH); row_ = b.take<int32_t>(4);
        ident_ = b.take<int32_t>(T * (uint64_t) cap_);
        Rin_ = b.take<float>(T * HC * N); R_ = b.take<float>(T * HC * N);
        emb_ = b.take<float>(T * N); en_ = b.take<float>(T * N); e2_ = b.take<float>(T * N);
        hn_ = b.take<float>(T * HC * N); h2_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); bo_ = b.take<float>(T * N);
        xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes((int) (NH * HD), 8));
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>((uint64_t) attn_scratch_floats_);   // the full layer runs one row at a time
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); y_ = b.take<float>(T * N);
        sample_ = b.take<float>(T * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        grp_ptr_ = b.take<unsigned long long>(T * K); grp_start_ = b.take<int32_t>(T * K + 1);
        grp_counts_ = b.take<int32_t>(4);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        hit_scratch_ = b.take<uint8_t>(strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff));
        sh_scratch_ = (float*) b.take<uint8_t>(strata::kernels::shared_expert_scratch_bytes(g.n_ff));
        x_bf16_ = b.take<uint16_t>(N);
        out_ids_ = b.take<int32_t>(T + 4);
        probs_ = b.take<float>(T + 4);
        dummy_inj_ = b.take<float>(HC);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) { err = "mtp: buffers do not fit"; return false; }
    cudaMemset(arena_, 0, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    vram_ += count.used;
    {
        std::vector<int32_t> id((size_t) (T * (uint64_t) cap_));
        for (uint64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < cap_; ++i) id[(size_t) (t * (uint64_t) cap_ + (uint64_t) i)] = (int32_t) i;
        cudaMemcpy(ident_, id.data(), id.size() * 4, cudaMemcpyHostToDevice);
    }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) { err = "mtp: stream"; return false; }
    const double files_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_files).count();
    std::fprintf(stderr, "maya mtp: draft layer loaded, %.0f MiB of VRAM (experts %.0f, dense %.0f), files read in %.2f s (%.0f MiB/s)\n",
                 (double) vram_ / 1048576.0, (double) g.n_expert * strata::kernels::cpu::BLOB / 1048576.0,
                 (double) tensors_.back().off / 1048576.0, files_s,
                 files_s > 0 ? ((double) g.n_expert * strata::kernels::cpu::BLOB + (double) tensors_.back().off) /
                                   1048576.0 / files_s : 0.0);
    return true;
}

bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err,
                      const std::string& head_gguf) {
    const OnDevice on_device(device_);
    wt_ = &wt;
    head_ = head;
    window_R_ = window_R;
    const WeightRef* wo = wt.find("output.weight");
    if (!wo) { err = "mtp: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;
    if (head == nullptr || !head->loaded()) { err = "mtp: the draft layer needs the native head (--native)"; return false; }
    if (head_logits_ == nullptr &&
        cudaMalloc((void**) &head_logits_, (size_t) max_t_ * (size_t) n_vocab_ * sizeof(float)) != cudaSuccess) {
        err = "mtp: the draft logits do not fit";
        return false;
    }
    // the draft head's token subset, when tools/draft_vocab.py wrote one (row-major, Q6_K_SOA, or Q8_0_SOA)
    if (dhead_ == nullptr) {
        std::vector<uint8_t> raw;
        if (read_file(rt_dir_ + "/draft_vocab.bin", raw) && raw.size() >= 4 && raw.size() % 4 == 0) {
            std::vector<int32_t> ids(raw.size() / 4);
            std::memcpy(ids.data(), raw.data(), raw.size());
            if (const char* lim_s = std::getenv("MAYA_MTP_DVOCAB_MAX")) {
                const int64_t lim = std::atoll(lim_s);
                if (lim >= 1024 && lim < (int64_t) ids.size()) {
                    // Preserve all special/control tokens (>= 248000) at the tail + keep the most frequent BPE prefix.
                    std::vector<int32_t> special;
                    for (int32_t id : ids) if (id >= 248000) special.push_back(id);
                    const int64_t keep_bpe = std::max<int64_t>(0, lim - (int64_t) special.size());
                    std::vector<int32_t> trimmed;
                    trimmed.reserve((size_t) lim);
                    for (int32_t id : ids) {
                        if (id < 248000 && (int64_t) trimmed.size() < keep_bpe) trimmed.push_back(id);
                    }
                    for (int32_t id : special) trimmed.push_back(id);
                    ids.swap(trimmed);
                }
            }
            n_dvocab_ = (int64_t) ids.size();
            const size_t ids_bytes = ids.size() * sizeof(int32_t);
            const bool is_soa_head = (head->type() == strata::kernels::kGgmlQ6_K_SOA ||
                                      head->type() == strata::kernels::kGgmlQ8_0_SOA);
            const size_t dhead_bytes = is_soa_head
                ? strata::kernels::native_mmvq_weight_bytes(head->type(), (int) g_->n_embd, (int) n_dvocab_)
                : (size_t) (n_dvocab_ * (int64_t) head->row_bytes());
            if (cudaMalloc((void**) &dvocab_, ids_bytes) != cudaSuccess ||
                cudaMalloc((void**) &dhead_, dhead_bytes) != cudaSuccess) {
                err = "mtp: the draft head does not fit";
                return false;
            }
            cudaMemcpy(dvocab_, ids.data(), ids_bytes, cudaMemcpyHostToDevice);
            if (head->n_out() < n_vocab_) {
                if (head_gguf.empty()) {
                    err = "mtp: TP-sharded head requires head_gguf path to build draft_vocab head";
                    return false;
                }
                strata::GgufFile gguf(head_gguf);
                const strata::TensorInfo* ot = nullptr;
                for (const auto& t : gguf.tensors()) if (t.name == "output.weight") { ot = &t; break; }
                if (!ot) { err = "mtp: output.weight not found in " + head_gguf; return false; }
                const size_t row_b = strata::kernels::native_mmvq_weight_bytes((int) ot->type, (int) g_->n_embd, 1);
                std::vector<uint8_t> gathered((size_t) n_dvocab_ * row_b);
                const uint8_t* src = static_cast<const uint8_t*>(gguf.tensor_data(*ot));
                for (int64_t i = 0; i < n_dvocab_; ++i) {
                    std::memcpy(gathered.data() + (size_t) i * row_b, src + (size_t) ids[(size_t) i] * row_b, row_b);
                }
                std::vector<uint8_t> soa_buf;
                const void* up = gathered.data();
                if (is_soa_head) {
                    soa_buf.resize(dhead_bytes);
                    strata::kernels::native_repack_to_soa_host((int) ot->type, gathered.data(), soa_buf.data(),
                                                               (int) g_->n_embd, (int) n_dvocab_, false);
                    up = soa_buf.data();
                }
                cudaMemcpy(dhead_, up, dhead_bytes, cudaMemcpyHostToDevice);
            } else if (head->type() == strata::kernels::kGgmlQ6_K_SOA) {
                strata::kernels::native_q6_k_soa_gather_rows(
                    head->weights(), (int) g_->n_embd, (int) n_vocab_, dvocab_, (int) n_dvocab_, dhead_, nullptr);
            } else if (head->type() == strata::kernels::kGgmlQ8_0_SOA) {
                strata::kernels::native_q8_0_soa_gather_rows(
                    head->weights(), (int) g_->n_embd, (int) n_vocab_, dvocab_, (int) n_dvocab_, dhead_, nullptr);
            } else {
                strata::kernels::gather_rows(
                    (const uint8_t*) head->weights(), (int64_t) head->row_bytes(), dvocab_, n_dvocab_, dhead_, nullptr);
            }
            cudaDeviceSynchronize();
            vram_ += (uint64_t) dhead_bytes + ids_bytes;
            std::fprintf(stderr, "maya mtp: draft head over %lld tokens (%.1f MiB%s)\n", (long long) n_dvocab_,
                         (double) dhead_bytes / 1048576.0,
                         head->type() == strata::kernels::kGgmlQ6_K_SOA ? ", Q6_K_SOA"
                         : head->type() == strata::kernels::kGgmlQ8_0_SOA ? ", Q8_0_SOA" : "");
        }
    }
    return true;
}

// The layer for T rows.  full = false stops after the K/V append (the prompt only needs the cache).
bool MtpDrafter::record_forward(int T, int step_row0, cudaStream_t cs, std::string& err, bool skip_prefix) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    // step_row0 >= 0: the full layer on step rows [step_row0, +T); step_row0 < 0: K/V only on rows [-1 - step_row0, +T)
    const bool full = step_row0 >= 0;
    const int row0 = full ? step_row0 : -1 - step_row0;
    if (full && T != 1) { err = "mtp: the full layer runs one row at a time (its attention scratch is sized for one)"; return false; }
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const QsaShapes s = shapes_of(g);
    const int32_t* step = step_ + row0 * 4;
    const int32_t* pos = pos_ + row0 * NH;
    try {
        auto norm_rope = [&](float* data, const float* gamma, int rows, int cols, const int32_t* p) {
            native_qsa_rms_norm_weighted(data, gamma, data, cols, rows, EPS, cs);
            if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, (float) qsa_freq_base(), p, cs);
            else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st_.cos_tab, st_.sin_tab, p, cs);
        };
        if (!skip_prefix) {
        // ---- the two input branches
        const WeightRef* we = wt_->find("token_embd.weight");
        if (!we) { err = "mtp: token_embd.weight is missing"; return false; }
        if (const NativeEmbed* ne = native_embed()) {   // plan v0.3 P6: the GGUF-form table
            ne->gather_dev(tok_, T, emb_, cs);
        } else {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            embedding_gather_dev(codes, scales, offsets, tok_, T, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                 (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), emb_, cs);
        }
        native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, cs);
        native_quantize_q8_1(en_, xq_, (int) N, T, cs);
        native_mmvq(q8_type_, q8("fc_embedding.weight"), xq_, e2_, (int) N, (int) N, T, cs);
        native_qsa_rms_norm_weighted(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) (HC * N), T, EPS, cs);
        for (int c0 = 0; c0 < T * HC; c0 += 8) {
            const int nc = (int) std::min<int64_t>(8, T * HC - c0);
            native_quantize_q8_1(hn_ + (size_t) c0 * N, xq_, (int) N, nc, cs);
            native_mmvq(q8_type_, q8("fc_hidden.weight"), xq_, h2_ + (size_t) c0 * N, (int) N, (int) N, nc, cs);
        }
        add_streams_broadcast(h2_, e2_, R_, N, (int) HC, T, cs);
        // ---- the attention hyper-connection (fuse Q8_1 quantization into hc_up when SoA!)
        bool q8_attn_fused = (hc_q8_ptr_[1] != nullptr);
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                fa[t].w_norm = f32("attn_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_down_q8 = hc_q8_ptr_[0]; fa[t].w_down_q8_soa = (hc_q8_ptr_[0] != nullptr);
                fa[t].w_up_q8 = hc_q8_ptr_[1]; fa[t].w_up_q8_soa = (hc_q8_ptr_[1] != nullptr);
                fa[t].w_inject = bf16("attn_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj_ + t * HC; fa[t].mixed = mixed_ + t * N;
                fa[t].mixed_q8 = (t == 0 && q8_attn_fused) ? xq_ : nullptr;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        // ---- attention: K/V into the layer's own cache, then (full) dense attention over every cell
        if (!q8_attn_fused) native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
        if (q8_type_ == kGgmlQ8_0_SOA) {
            native_q8_0_soa_mmvq_dual(q8("self_attn.k_proj.weight"), q8("self_attn.v_proj.weight"), xq_,
                                      kcur_, vcur_, (int) N, (int) (NKV * HD), (int) (NKV * HD), T, cs);
        } else {
            native_mmvq(q8_type_, q8("self_attn.k_proj.weight"), xq_, kcur_, (int) N, (int) (NKV * HD), T, cs);
            native_mmvq(q8_type_, q8("self_attn.v_proj.weight"), xq_, vcur_, (int) N, (int) (NKV * HD), T, cs);
        }
        if (T > 1 && !st_.kv_q4 && st_.kv_int8) {
            for (int t = 0; t < T; ++t) {
                norm_rope(kcur_ + t * NKV * HD, f32("self_attn.k_norm.weight"), (int) NKV, (int) HD, pos + t * NH);
            }
            kv_append_q8_multi(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step,
                               kcur_, vcur_, T, s, cs, &st_.host);
        } else {
            for (int t = 0; t < T; ++t) {
                norm_rope(kcur_ + t * NKV * HD, f32("self_attn.k_norm.weight"), (int) NKV, (int) HD, pos + t * NH);
                if (st_.kv_q4) {   // Q4_0 KV (kv_q4.hpp): rotated K and V
                    fwht256_inplace_cuda(kcur_ + t * NKV * HD, NKV, cs);
                    fwht256_inplace_cuda(vcur_ + t * NKV * HD, NKV, cs);
                    kv_append_q4_step(st_.k_q4, st_.v_q4, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                      vcur_ + t * NKV * HD, s, cs, &st_.host);
                } else if (st_.kv_int8)
                    kv_append_q8_step(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step + t * 4,
                                      kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st_.host);
                else
                    kv_append_step(st_.k_pool, st_.v_pool, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                   vcur_ + t * NKV * HD, s, cs, &st_.host);
            }
        }
        if (!full) return true;
        } else {
        native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
        }
        native_mmvq(q8_type_, q8("self_attn.q_proj.weight"), xq_, qfull_, (int) N, (int) (NH * 2 * HD), T, cs);
        for (int t = 0; t < T; ++t) {
            float* qc = qcur_ + t * NH * HD;
            native_qsa_rms_norm_weighted_strided(qfull_ + t * NH * 2 * HD, (int) (2 * HD),
                                                 f32("self_attn.q_norm.weight"), qc, (int) HD, (int) NH, EPS, cs);
            if (native_rope_enabled())
                native_rope_apply(qc, qc, (int) NH, (int) HD, (int) s.n_rot, (float) qsa_freq_base(), pos + t * NH, cs);
            else
                rope_neox_apply(qc, qc, (int) NH, (int) HD, (int) s.n_rot, st_.cos_tab, st_.sin_tab, pos + t * NH, cs);
            if (st_.kv_q4) fwht256_inplace_cuda(qc, NH, cs);
        }
        const QsaAttnPools pools = qsa_attn_pools(st_);
        if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
        qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs);
        if (st_.kv_q4) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
        for (int t = 0; t < T; ++t)
            native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD, (int) NH, (int) HD, cs);
        native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
        native_mmvq(q8_type_, q8("self_attn.o_proj.weight"), xq_, bo_, (int) (NH * HD), (int) N, T, cs);
        // ---- the MLP hyper-connection (the attention write + Q8_1 quant folded in when SoA!)
        const bool q8_mlp_fused = (hc_q8_ptr_[3] != nullptr && T == 1);
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = bo_ + t * N; fa[t].inj_prev = inj_ + t * HC;
                fa[t].w_norm = f32("mlp_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("mlp_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("mlp_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_down_q8 = hc_q8_ptr_[2]; fa[t].w_down_q8_soa = (hc_q8_ptr_[2] != nullptr);
                fa[t].w_up_q8 = hc_q8_ptr_[3]; fa[t].w_up_q8_soa = (hc_q8_ptr_[3] != nullptr);
                fa[t].w_inject = bf16("mlp_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj2_ + t * HC; fa[t].mixed = mixed_ + t * N;
                fa[t].mixed_q8 = (t == 0 && q8_mlp_fused) ? xq_ : nullptr;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        // ---- MoE: router, the 512 resident experts, the shared expert, the combine, the write
        for (int t = 0; t < T; ++t) {
            bf16_gemv_fp32_mmvf(mixed_ + t * N, bf16("mlp.gate.weight"), logits_ + t * g.n_expert, (int) N, (int) g.n_expert, cs);
            if (native_router_enabled()) native_router_top10(logits_ + t * g.n_expert, ids_ + t * K, w_ + t * K, cs);
            else router_top10(logits_ + t * g.n_expert, 1, (int) g.n_expert, (int) K, ids_ + t * K, w_ + t * K, cs);
        }
        moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) strata::kernels::cpu::BLOB, grp_ptr_,
                           grp_start_, grp_counts_, hit_dst_, hit_slot_, cs);
        quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
        moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K, hit_xq_,
                       hit_xs_, hit_scratch_, parts_, cs, experts_soa_);
        NativeSharedWeights nsw;
        nsw.gate_type = q8_type_; nsw.gate_data = q8("mlp.shared_expert.gate_proj.weight");
        nsw.up_type = q8_type_; nsw.up_data = q8("mlp.shared_expert.up_proj.weight");
        nsw.down_type = q8_type_; nsw.down_data = q8("mlp.shared_expert.down_proj.weight");
        nsw.q8_1 = xq_;
        const SForm none{};
        uint8_t* sh_p = (uint8_t*) sh_scratch_;
        const uint64_t sh_a = ((uint64_t) g.n_ff * 4 + 15) & ~15ull;
        const uint64_t sh_q0 = ((uint64_t) (g.n_ff / 32) * 34 + 15) & ~15ull;
        const uint64_t sh_qk = ((uint64_t) (g.n_ff / 256) * 292 + 15) & ~15ull;
        float* sh_gate = (float*) sh_p;
        float* sh_up = (float*) (sh_p + sh_a);
        float* sh_g = (float*) (sh_p + sh_a * 2 + sh_q0 + sh_qk);
        for (int t = 0; t < T; ++t) {
            if (!shared_expert_native_bf16())
                f32_to_bf16_bulk(mixed_ + t * N, x_bf16_, N, cs);
            if (q8_type_ == kGgmlQ8_0_SOA) {
                shared_expert_multi(1, mixed_ + t * N, x_bf16_, nsw, bf16("mlp.shared_expert_gate.weight"),
                                    sh_gate, sh_up, sh_g, shared_ + t * N, N, g.n_ff, cs,
                                    q8_mlp_fused ? xq_ : nullptr);
            } else {
                shared_expert(nullptr, nullptr, x_bf16_, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr, none,
                              nullptr, nullptr, nullptr, bf16("mlp.shared_expert_gate.weight"), sh_scratch_, shared_ + t * N,
                              N, g.n_ff, 32, cs, mixed_ + t * N, &nsw);
            }
            if (native_moe_combine_enabled())
                native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            else
                moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
        }
        // ---- the final mixer (folding the FFN gr_write + Q8_1 quant into fused_gr_read_multi) and the main model's head
        bool q8_head_fused = (hc_q8_ptr_[5] != nullptr);
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = y_ + t * N; fa[t].inj_prev = inj2_ + t * HC;
                fa[t].w_norm = f32("hyper_connection_mixer.hc_norm.weight");
                fa[t].w_down = bf16("hyper_connection_mixer.input_mix_weight_down.weight");
                fa[t].w_up = bf16("hyper_connection_mixer.input_mix_weight_up.weight");
                fa[t].w_down_q8 = hc_q8_ptr_[4]; fa[t].w_down_q8_soa = (hc_q8_ptr_[4] != nullptr);
                fa[t].w_up_q8 = hc_q8_ptr_[5]; fa[t].w_up_q8_soa = (hc_q8_ptr_[5] != nullptr);
                fa[t].w_inject = nullptr;
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = nullptr; fa[t].mixed = sample_ + t * N;
                fa[t].mixed_q8 = (t == 0 && q8_head_fused) ? xq_ : nullptr;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        if (!q8_head_fused) native_quantize_q8_1(sample_, xq_, (int) N, T, cs);
        const bool sub = dhead_ != nullptr;
        const int64_t nv = sub ? n_dvocab_ : n_vocab_;
        native_mmvq(head_->type(), sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, T, cs);
        mtp_greedy_prob_map(head_logits_, T, (int) nv, sub ? dvocab_ : nullptr, out_ids_, probs_, cs);
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

namespace {
bool finish_capture(cudaStream_t cs, bool ok, cudaGraphExec_t& exec, const char* what, std::string& err) {
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("mtp: ") + what + " capture: " + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    // an explicit upload: the first launch's implicit one blocked behind a device-side spin (verify.cpp)
    cudaGraphUpload(exec, cs);
    cudaStreamSynchronize(cs);
    return true;
}
}  // namespace

bool MtpDrafter::capture_prefill(int T, std::string& err) {
    if (prefill_exec_[T]) return true;
    using namespace strata::kernels;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) T * g_->n_head, cs_);
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T)
    return finish_capture(cs_, ok, prefill_exec_[T], "prefill", err);
}

bool MtpDrafter::capture_prefill_dev(int T, std::string& err) {
    if (prefill_dev_exec_[T]) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T); tok_/step_/pos_ filled before launch
    return finish_capture(cs_, ok, prefill_dev_exec_[T], "prefill (device inputs)", err);
}

bool MtpDrafter::capture_round(int T, std::string& err) {
    if (round_exec_[T]) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    bool ok = true;
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) 2 * T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) 2 * T * g_->n_head, cs_);
    copy_i32_from_mapped(row_, m_row_, 2, cs_);
    copy_from_mapped(Rin_, window_R_, (int64_t) T * HCN, cs_);
    // the catch-up: K/V for the window's T cells, then the full layer for row a only (its cell's K/V is written
    // again, identically), staged by the host in step row 2*max_t - 1; the draft chain is one graph per step
    // (`capture_step`) so the host can stop it when a draft is unlikely
    // Reuse R_, mixed_, inj_ and already-appended K/V from row `a` of the catch-up pass (skip redundant 50% prefix!).
    const int ra = 2 * max_t_ - 1;
    ok = record_forward(T, -1, cs_, err);
    if (ok && T > 1) {
        copy_indexed(R_, R_, HCN, row_, HCN, cs_);
        copy_indexed(mixed_, mixed_, g_->n_embd, row_, g_->n_embd, cs_);
        copy_indexed(inj_, inj_, g_->hc, row_, g_->hc, cs_);
    }
    if (ok) {
        copy_i32_from_mapped(step_ + ra * 4, m_step_ + ra * 4, 4, cs_);
        copy_i32_from_mapped(pos_ + ra * g_->n_head, m_pos_ + ra * g_->n_head, g_->n_head, cs_);
        ok = record_forward(1, ra, cs_, err, /*skip_prefix=*/true);
    }
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, 0, cs_, probs_, m_prob_, m_done_, 1u);
    return finish_capture(cs_, ok, round_exec_[T], "round", err);
}

// Chain step j (1..max_t-2): one row at the cell staged in step row `max_t + j - 1`, from the previous step's
// residual and token (left in Rin_[0] / tok_[0] by mtp_select); draft j and its probability to the mapped outputs.
bool MtpDrafter::capture_step(int j, std::string& err) {
    if (step_exec_[j]) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    const int row = max_t_ + j - 1;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    copy_i32_from_mapped(step_ + row * 4, m_step_ + row * 4, 4, cs_);
    copy_i32_from_mapped(pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head, cs_);
    bool ok = record_forward(1, row, cs_, err);
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, j, cs_, probs_, m_prob_, m_done_, (uint32_t) (j + 1));
    return finish_capture(cs_, ok, step_exec_[j], "step", err);
}

void MtpDrafter::kv_restore(int64_t upto) {
    const OnDevice on_device(device_);
    if (st_.kv_mode != 2 || upto <= 0) return;
    // the ring's blocks below `upto`, from the host copy: a checkpoint resume may have left later cells in them
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t b1 = (upto + s.page_size - 1) / s.page_size, b0 = std::max<int64_t>(0, b1 - st_.n_slots);
    strata::kernels::kv_ring_restore(qsa_attn_pools(st_), st_.host, qsa_kv_format(st_), b0, b1, st_.n_slots, s, cs_);
    cudaStreamSynchronize(cs_);
}

bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    const int64_t HCN = g_->hc * g_->n_embd;
    // cells the window can never reach again need no K/V
    const int64_t first_needed = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
    // E-4: every group's token / step / position records uploaded at once; each group is then device copies and a
    // graph on the one stream, with a single sync at the end (a group of <= max_t rows used to be staged in mapped
    // memory and synced before the next: ~5,500 host round trips on a 32K prompt).  The same work in the same
    // order.  MAYA_MTP_PREFILL_SYNC=1 keeps the old loop.
    static const bool per_group_sync = std::getenv("MAYA_MTP_PREFILL_SYNC") != nullptr;
    const int64_t NHp = g_->n_head, per_row = 1 + 4 + NHp;
    if (!per_group_sync && n > 0) {
        if (pf_cap_ < n * per_row) {
            if (pf_dev_) cudaFree(pf_dev_);
            pf_dev_ = nullptr;
            pf_cap_ = 0;
            if (cudaMalloc((void**) &pf_dev_, (size_t) (n * per_row) * sizeof(int32_t)) != cudaSuccess) {
                err = "mtp prefill: the input records do not fit";
                return false;
            }
            pf_cap_ = n * per_row;
        }
        std::vector<int32_t> rec((size_t) (n * per_row));
        int32_t* tk = rec.data();
        int32_t* stp = tk + n;
        int32_t* ps = stp + 4 * n;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t cell = cell0 + i;
            tk[i] = next_tokens[i];
            stp[i * 4 + 0] = (int32_t) cell;
            stp[i * 4 + 1] = (int32_t) (cell + 1);
            stp[i * 4 + 2] = (int32_t) ((cell + 1) / 4);
            stp[i * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < NHp; ++h) ps[i * NHp + h] = (int32_t) cell;
        }
        if (cudaMemcpyAsync(pf_dev_, rec.data(), rec.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "mtp prefill: the input records upload failed";
            return false;
        }
        const int32_t* d_tk = pf_dev_;
        const int32_t* d_stp = d_tk + n;
        const int32_t* d_ps = d_stp + 4 * n;
        for (int64_t c = 0; c < n; c += max_t_) {
            const int T = (int) std::min<int64_t>(max_t_, n - c);
            if (cell0 + c + T <= first_needed) continue;
            if (!capture_prefill_dev(T, err)) return false;
            if (cudaMemcpyAsync(tok_, d_tk + c, (size_t) T * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(step_, d_stp + c * 4, (size_t) T * 16, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(pos_, d_ps + c * NHp, (size_t) (T * NHp) * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float), cudaMemcpyDeviceToDevice,
                                cs_) != cudaSuccess ||
                cudaGraphLaunch(prefill_dev_exec_[T], cs_) != cudaSuccess) {
                err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        ms_prefill += ms_since(t0);
        return true;
    }
    for (int64_t c = 0; c < n; c += max_t_) {
        const int T = (int) std::min<int64_t>(max_t_, n - c);
        if (cell0 + c + T <= first_needed) continue;
        if (!capture_prefill(T, err)) return false;
        for (int t = 0; t < T; ++t) {
            const int64_t cell = cell0 + c + t;
            h_tok_[t] = next_tokens[c + t];
            h_step_[t * 4 + 0] = (int32_t) cell;
            h_step_[t * 4 + 1] = (int32_t) (cell + 1);
            h_step_[t * 4 + 2] = (int32_t) ((cell + 1) / 4);
            h_step_[t * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < g_->n_head; ++h) h_pos_[t * g_->n_head + h] = (int32_t) cell;
        }
        if (cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float), cudaMemcpyDeviceToDevice,
                            cs_) != cudaSuccess ||
            cudaGraphLaunch(prefill_exec_[T], cs_) != cudaSuccess ||
            cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    ms_prefill += ms_since(t0);
    return true;
}

bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                       float* probs, float min_p, int* n_drafts) {
    const OnDevice on_device(device_);
    if (T < 1 || T > max_t_ || a < 0 || a >= T) { err = "mtp: draft arguments out of range"; return false; }
    // Only cells p..p+a (nc = a + 1 accepted tokens) are needed in MTP KV; tokens a+1..T-1 were rejected
    // and immediately overwritten by draft steps j=1..3. Bit-identical catch-up with 2-5x less wasted work!
    const int nc = a + 1;
    if (!capture_round(nc, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < nc; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, p + a);
    h_row_[0] = a;
    h_row_[1] = 0;
    *(volatile uint32_t*) h_done_ = 0u;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto wait_done = [&](uint32_t want, const char* what) -> bool {
        volatile uint32_t* const df = h_done_;
        uint32_t spins = 0;
        while (*df < want) {
            if ((++spins & 16383u) == 0) {
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && q != cudaSuccess && *df < want) {
                    err = std::string(what) + cudaGetErrorString(q);
                    return false;
                }
            }
            strata_cpu_pause();
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        return true;
    };
    if (cudaGraphLaunch(round_exec_[nc], cs_) != cudaSuccess) {
        err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    if (!wait_done(1u, "mtp draft: ")) return false;
    drafts[0] = ((volatile int32_t*) h_out_)[0];
    float pj = ((volatile float*) h_prob_)[0];
    if (probs) probs[0] = pj;
    int n = 1;
    // the chain continues while the last draft is likely enough to be verified
    for (int j = 1; j < std::min(max_t_ - 1, max_drafts_) && pj >= min_p; ++j) {
        if (!capture_step(j, err)) return false;
        put(max_t_ + j - 1, p + a + j);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (cudaGraphLaunch(step_exec_[j], cs_) != cudaSuccess) {
            err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (!wait_done((uint32_t) (j + 1), "mtp draft step: ")) return false;
        drafts[j] = ((volatile int32_t*) h_out_)[j];
        pj = ((volatile float*) h_prob_)[j];
        if (probs) probs[j] = pj;
        ++n;
    }
    for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
    if (n_drafts) *n_drafts = n;
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool MtpDrafter::draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                             float* probs, float min_p, int* n_drafts) {
    const OnDevice on_device(device_);
    // row 0 is the real pair; rows 1.. repeat it and only write cells the next round overwrites
    const int64_t HCN = g_->hc * g_->n_embd;
    for (int t = 0; t < T; ++t)
        if (cudaMemcpy((void*) (window_R_ + (size_t) t * HCN), R_row, (size_t) HCN * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            err = "mtp: staging the first residual failed";
            return false;
        }
    std::vector<int32_t> toks((size_t) T, token);
    return draft(T, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts);
}

}  // namespace strata::core
