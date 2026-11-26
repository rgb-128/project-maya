// src/kernels/cpu/pool.cpp - P2.S3: the CPU expert pool.  Read pool.hpp first; it explains the protocol.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include "strata/platform/cpu_relax.hpp"

#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::kernels::cpu {

namespace {
constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return ((uint64_t) epoch << 32) | ((uint64_t) n << 16) | (uint64_t) i;
}
}  // namespace

std::vector<int> physical_cores(bool skip_first) {
    std::vector<int> cores;
#if defined(_WIN32)
    // Ask the OS rather than assuming a layout.  `hardware_concurrency()` returns LOGICAL processors, and on
    // every SMT machine half of them are siblings - pinning one worker to each of the first N would put two
    // workers on each physical core and halve the bandwidth the expert kernel is bound by.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) cores.push_back((int) i);
    } else {
        std::vector<char> buf(len);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                             (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
            const char* p = buf.data();
            const char* end = p + len;
            while (p < end) {
                const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
                if (e->Relationship == RelationProcessorCore) {
                    const GROUP_AFFINITY& g = e->Processor.GroupMask[0];
                    for (int bit = 0; bit < 64; ++bit)
                        if (g.Mask & (1ull << bit)) { cores.push_back((int) (g.Group * 64 + bit)); break; }
                }
                p += e->Size;
            }
        }
    }
#else
    // The logical CPUs this process may run on, ONE PER PHYSICAL CORE (issue #40): SMT siblings share a core's
    // load/store bandwidth, so a worker on each would put two workers on one core, as the Windows branch above
    // explains.  sysfs names each CPU's (package, core); the first allowed CPU of each pair is kept, so a taskset
    // that leaves out the first sibling still gets its core.  Without sysfs every allowed CPU counts, as before.
    auto topo = [](int cpu, const char* what) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    std::vector<int> allowed;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &set)) allowed.push_back(i);
    } else {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) allowed.push_back((int) i);
    }
    std::vector<std::pair<long, long>> seen;
    for (int cpu : allowed) {
        const long pkg = topo(cpu, "physical_package_id"), core = topo(cpu, "core_id");
        if (pkg >= 0 && core >= 0) {
            const std::pair<long, long> key{pkg, core};
            if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;   // an SMT sibling
            seen.push_back(key);
        }
        cores.push_back(cpu);
    }
#endif
    if (skip_first && !cores.empty()) cores.erase(cores.begin());
    return cores;
}

namespace {

void pin_this_thread(int core) {
    if (core < 0) return;
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (core & 63));
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

}  // namespace

long long pin_current_thread(int core) {
    if (core < 0) return -1;
#if defined(_WIN32)
    // `SetThreadAffinityMask` RETURNS the previous mask, or 0 on failure - so 0 doubles as the error, which is
    // why the caller must not treat it as a restorable value.
    const DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (core & 63));
    return prev == 0 ? -1 : (long long) prev;
#else
    cpu_set_t prev;
    CPU_ZERO(&prev);
    if (pthread_getaffinity_np(pthread_self(), sizeof prev, &prev) != 0) return -1;
    unsigned long mask = 0;
    for (int i = 0; i < CPU_SETSIZE && i < 64; ++i)
        if (CPU_ISSET(i, &prev)) mask |= 1ul << i;
    pin_this_thread(core);
    return (long long) mask;
#endif
}

void restore_thread_affinity(long long previous) {
    if (previous <= 0) return;
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) previous);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < 64; ++i)
        if ((previous >> i) & 1) CPU_SET(i, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

namespace {
std::atomic<const ExpertPool*> g_diag_pool{nullptr};
void diag_active_pool(std::FILE* f) {
    if (const ExpertPool* p = g_diag_pool.load()) p->diag(f);
}
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

void ExpertPool::diag(std::FILE* f) const {
    const uint64_t h = head_.load();
    std::fprintf(f, "  expert pool: epoch %u, batch epoch %u: %u of %u jobs claimed, %u done; %u of %d workers parked, "
                    "%u sleeping; mode %d\n", epoch_.load(), (uint32_t) (h >> 32), (uint32_t) h & 0xffffu,
                 (uint32_t) (h >> 16) & 0xffffu, done_.load(), parked_.load(), n_, sleepers_.load(), mode_);
    std::fprintf(f, "  expert pool threads:");
    for (int i = 0; i < n_; ++i) {
        const int32_t s = wstate_[(size_t) i].load();
        if (s == kParked) std::fprintf(f, " w%d=parked", i);
        else if (s == kSleeping) std::fprintf(f, " w%d=sleeping", i);
        else if (s == kBetween) std::fprintf(f, " w%d=draining", i);
        else std::fprintf(f, " w%d=job%d", i, s);
    }
    const int32_t hs = hstate_.load();
    const char* hn = hs == kIdle ? "idle" : hs == kWaitParked ? "waiting for the workers to park"
                   : hs == kWaitDone ? "waiting for the jobs to finish" : "running a job";
    std::fprintf(f, "; host %s", hn);
    if (hs >= 0) std::fprintf(f, " %d", hs);
    std::fprintf(f, " for %lld ms\n", (long long) (now_ms() - hstate_ms_.load()));
}

ExpertPool::ExpertPool(int n_workers, bool pin, bool host_works) : host_works_(host_works) {
    if (const char* e = std::getenv("MAYA_POOL_SPIN_US"))   // a test knob; see kSpinBeforeSleep
        spin_before_sleep_ = std::chrono::microseconds((std::max)(0, std::atoi(e)));
    const std::vector<int> cores = physical_cores(true);
    n_ = n_workers > 0 ? n_workers : (int) cores.size();
    if (n_ < 1) n_ = 1;
    scratch_.resize((size_t) n_);
    wstate_.reset(new std::atomic<int32_t>[(size_t) n_]);
    wdone_.reset(new WorkerDone[(size_t) n_]);
    w_node_.assign((size_t) n_, 0);
    fast_rank_all_.assign((size_t) n_, 0);
    fast_rank_node_.assign((size_t) n_, 0);
    for (int i = 0; i < n_; ++i) wstate_[(size_t) i].store(kParked);
    hstate_ms_.store(now_ms());
    g_diag_pool.store(this);
    strata::core::diag_pool_fn().store(&diag_active_pool);
    split_.resize((size_t) kMaxSplit);
    split_multi_.resize((size_t) kMaxSplitMulti);
    auto cpu_pkg = [](int core) -> int {
#if !defined(_WIN32)
        if (core < 0) return 0;
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", core);
        long v = 0;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = 0;
            std::fclose(f);
        }
        return (v == 1) ? 1 : 0;
#else
        (void) core;
        return 0;
#endif
    };
    for (int i = 0; i < n_; ++i) {
        const int core = pin ? (i < (int) cores.size() ? cores[(size_t) i] : -1) : -1;
        const int nd = cpu_pkg(core);
        w_node_[(size_t) i] = nd;
        node_workers_[nd].push_back(i);
        fast_rank_all_[(size_t) i] = (host_works_ ? 1 : 0) + i;
    }
    n_numa_nodes_ = (!node_workers_[0].empty() && !node_workers_[1].empty()) ? 2 : 1;
    // In NUMA-local mode: Node 0 uses host (rank 0, core 0) + Node 0 workers (ranks 1..W0);
    // Node 1 uses Node 1 workers (ranks 0..W1-1, zero cross-socket host reads!).
    for (size_t k = 0; k < node_workers_[0].size(); ++k)
        fast_rank_node_[(size_t) node_workers_[0][k]] = (host_works_ ? 1 : 0) + (int) k;
    for (size_t k = 0; k < node_workers_[1].size(); ++k)
        fast_rank_node_[(size_t) node_workers_[1][k]] = (int) k;

    threads_.reserve((size_t) n_);
    for (int i = 0; i < n_; ++i) {
        const int core = pin ? (i < (int) cores.size() ? cores[(size_t) i] : -1) : -1;
        threads_.emplace_back([this, i, core] {
            pin_this_thread(core);
            worker(i);
        });
    }
}

ExpertPool::~ExpertPool() {
    const ExpertPool* self = this;
    g_diag_pool.compare_exchange_strong(self, nullptr);
    stop_.store(true, std::memory_order_release);
    // Bump the epoch so a PARKED worker notices the stop flag rather than sleeping through it.
    publish();
    for (auto& t : threads_) t.join();
}

void ExpertPool::publish() {
    // Both sides are seq_cst, and that is the whole lost-wakeup argument: a worker going to sleep does
    // `sleepers_++` and then reads `epoch_`, the host does `epoch_++` and then reads `sleepers_`.  In one total
    // order at least one of them sees the other's write - the worker sees the new epoch and does not sleep, or
    // the host sees the sleeper and notifies under the mutex the worker holds until it is inside `wait`.
    // On x86 the fetch_add is a locked xadd either way, so this costs the token path nothing.
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) {
        std::lock_guard<std::mutex> lk(sleep_mu_);
        sleep_cv_.notify_all();
    }
}

void ExpertPool::worker(int id) {
    const int my_node = w_node_[(size_t) id];
    uint32_t seen = 0;
    uint32_t seen_fast = 0;
    parked_.fetch_add(1, std::memory_order_acq_rel);
    for (;;) {
        const auto parked_at = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        while (fast_epoch_[my_node].val.load(std::memory_order_acquire) == seen_fast &&
               epoch_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
            strata_cpu_pause();
            if ((++spins & 1023u) != 0) continue;
            if (std::chrono::steady_clock::now() - parked_at < spin_before_sleep_) continue;
            std::unique_lock<std::mutex> lk(sleep_mu_);
            wstate_[(size_t) id].store(kSleeping, std::memory_order_relaxed);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            sleep_cv_.wait(lk, [&] {
                return fast_epoch_[my_node].val.load(std::memory_order_seq_cst) != seen_fast ||
                       epoch_.load(std::memory_order_seq_cst) != seen ||
                       stop_.load(std::memory_order_relaxed);
            });
            sleepers_.fetch_sub(1, std::memory_order_relaxed);
            wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        const uint32_t fe = fast_epoch_[my_node].val.load(std::memory_order_acquire);
        if (fe != seen_fast) {
            seen_fast = fe;
            const int rank = fast_use_numa_ ? fast_rank_node_[(size_t) id] : fast_rank_all_[(size_t) id];
            run_fast_slice(rank, fast_n_part_, my_node);
            wdone_[(size_t) id].val.store(fe, std::memory_order_release);
            continue;
        }
        // acquire: the batch this epoch published (`head`, and the description before it) is visible from here
        seen = epoch_.load(std::memory_order_acquire);
        parked_.fetch_sub(1, std::memory_order_acq_rel);   // leaving the park

        wstate_[(size_t) id].store(kBetween, std::memory_order_relaxed);
        drain(id, scratch_[(size_t) id], seen);
        wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        parked_.fetch_add(1, std::memory_order_acq_rel);   // back at the park
    }
}

void ExpertPool::run_fast_slice(int rank, int n_part, int worker_node) {
    if (fast_mode_ == 8) {
        if (rank < actq_n_) {
            const int t = actq_toks_[rank];
            native_quant_act(*nfmt_, actq_x_ + (size_t) t * (size_t) actq_H_, actq_dst_ + (size_t) t * actq_stride_);
        }
        return;
    }
    if (fast_mode_ == 7) {
        // Parallel intermediate activation quantization across (e, t) pairs!
        int idx = 0;
        const bool q2_simd = (nfmt_->d_type == 42) && q2_own_kernels() && (cpu_avx2_ok() || cpu_avx512_ok());
        for (int e = 0; e < fast_nb_; ++e) {
            for (int t = 0; t < mjobs_[e].nt; ++t) {
                if (idx % n_part == rank) {
                    if (q2_simd)
                        act_quant_any(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
                    else
                        native_quant_h(*nfmt_, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);
                }
                ++idx;
            }
        }
        return;
    }
    const bool q2_simd = q2_own_kernels() && (cpu_avx2_ok() || cpu_avx512_ok());
    if (fast_mode_ == 5 && fast_nb_ >= 2 && !(nfmt_->gu_type == 42 && q2_simd) && !(nfmt_->d_type == 42 && q2_simd)) {
        int total_q = 0;
        for (int e = 0; e < fast_nb_; ++e) total_q += mjobs_[e].nt;
        int qk = 0;
        size_t qts = 0;
        native_quant_h_info(*nfmt_, qk, qts);
        if (total_q >= 3 && qk > 0 && qk <= 64 && (FF % qk) == 0) {
            // Wave 6: When nb >= 2 and total_q >= 3 (otherwise mode=7 isn't run and nb=1 needs full 24-core split),
            // partition FF in qk-aligned blocks and fuse native_quant_h while sb.ff is hot in L1!
            const int per_blk = FF / qk;
            const int64_t total_blk = (int64_t) fast_nb_ * per_blk;
            const int64_t b0 = total_blk * (int64_t) rank / n_part;
            const int64_t b1 = total_blk * (int64_t) (rank + 1) / n_part;
            for (int64_t bk = b0; bk < b1;) {
                const int e = (int) (bk / per_blk), rb0 = (int) (bk % per_blk);
                const int rb1 = (int) std::min<int64_t>(per_blk, rb0 + (b1 - bk));
                const int r0 = rb0 * qk, r1 = rb1 * qk;
                SplitBufMulti& sb = split_multi_[(size_t) e];
                const uint8_t* const blob = (worker_node == 1 && mjobs_[e].blob_n1 != nullptr) ? mjobs_[e].blob_n1 : mjobs_[e].blob;
                float* ff[MAXT];
                for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                native_gu_rows(*nfmt_, blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
                for (int t = 0; t < mjobs_[e].nt; ++t) {
                    native_quant_h_range(*nfmt_, sb.ff[t] + r0,
                                         reinterpret_cast<uint8_t*>(sb.hq[t]) + (size_t) rb0 * qts, r1 - r0);
                }
                bk += rb1 - rb0;
            }
            return;
        }
    }
    const int per = fast_mode_ == 5 ? FF : H;
    const int64_t g0 = mrows_ * (int64_t) rank / n_part;
    const int64_t g1 = mrows_ * (int64_t) (rank + 1) / n_part;
    for (int64_t r = g0; r < g1;) {
        const int e = (int) (r / per), r0 = (int) (r % per);
        const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
        SplitBufMulti& sb = split_multi_[(size_t) e];
        const uint8_t* const blob = (worker_node == 1 && mjobs_[e].blob_n1 != nullptr) ? mjobs_[e].blob_n1 : mjobs_[e].blob;
        if (fast_mode_ == 5 && nfmt_->gu_type == 42 && q2_simd) {
            thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF];
            float* gp[MAXT];
            float* up[MAXT];
            for (int t = 0; t < mjobs_[e].nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
            const int nbk = (int) (nfmt_->n_embd / 64);
            q2_rows_any(blob, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, gp, r0, r1);
            q2_rows_any(blob + nfmt_->up_off, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, up, r0, r1);
            for (int t = 0; t < mjobs_[e].nt; ++t)
                for (int rr = r0; rr < r1; ++rr)
                    sb.ff[t][rr] = (gbuf[t][rr] / (1.f + std::exp(-gbuf[t][rr]))) * ubuf[t][rr];
        } else if (fast_mode_ == 5) {
            float* ff[MAXT];
            for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
            native_gu_rows(*nfmt_, blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
        } else if (nfmt_->d_type == 42 && q2_simd) {
            const ActQ* a2[MAXT];
            for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
            q2_rows_any(blob + nfmt_->down_off, nfmt_->d_row, (int) (nfmt_->n_ff / 64), a2,
                        mjobs_[e].nt, mjobs_[e].out, r0, r1);
        } else {
            const void* hq[MAXT];
            for (int t = 0; t < mjobs_[e].nt; ++t) hq[t] = sb.hq[t];
            native_down_rows(*nfmt_, blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);
        }
        r += r1 - r0;
    }
}

void ExpertPool::run_fast_phase(int mode, int target_node) {
    fast_mode_ = mode;
    if (target_node >= 0 && n_numa_nodes_ == 2) {
        fast_use_numa_ = true;
        const bool host_in = (target_node == 0) && host_works_;
        const auto& ws = node_workers_[target_node];
        fast_n_part_ = (int) ws.size() + (host_in ? 1 : 0);
        const uint32_t e = fast_epoch_[target_node].val.fetch_add(1, std::memory_order_seq_cst) + 1;
        if (sleepers_.load(std::memory_order_seq_cst) != 0) {
            std::lock_guard<std::mutex> lk(sleep_mu_);
            sleep_cv_.notify_all();
        }
        if (host_in) run_fast_slice(0, fast_n_part_);
        for (int id : ws) {
            while (wdone_[(size_t) id].val.load(std::memory_order_acquire) != e) strata_cpu_pause();
        }
    } else {
        fast_use_numa_ = false;
        fast_n_part_ = n_ + (host_works_ ? 1 : 0);
        const uint32_t e0 = !node_workers_[0].empty()
            ? fast_epoch_[0].val.fetch_add(1, std::memory_order_seq_cst) + 1 : 0;
        const uint32_t e1 = !node_workers_[1].empty()
            ? fast_epoch_[1].val.fetch_add(1, std::memory_order_seq_cst) + 1 : 0;
        if (sleepers_.load(std::memory_order_seq_cst) != 0) {
            std::lock_guard<std::mutex> lk(sleep_mu_);
            sleep_cv_.notify_all();
        }
        if (host_works_) run_fast_slice(0, fast_n_part_);
        for (int id : node_workers_[0]) {
            while (wdone_[(size_t) id].val.load(std::memory_order_acquire) != e0) strata_cpu_pause();
        }
        for (int id : node_workers_[1]) {
            while (wdone_[(size_t) id].val.load(std::memory_order_acquire) != e1) strata_cpu_pause();
        }
    }
}

int ExpertPool::claim(uint32_t epoch) {
    uint64_t h = head_.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;               // not the batch this thread woke for
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        if (i >= n) return -1;                                       // exhausted
        if (head_.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return (int) i;
    }
}

uint32_t ExpertPool::begin_batch(int n) {
    if (n < 0 || n > 0xffff) {
        std::fprintf(stderr, "maya: expert pool batch of %d jobs is out of range\n", n);
        std::abort();
    }
    // Every job of the previous batch has completed (`wait_done`), and a claim of it can no longer succeed, so
    // nothing adds to `done` until this batch's first claim - which the release below orders after the reset.
    done_.store(0, std::memory_order_relaxed);
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1;   // only the host bumps the epoch
    head_.store(pack_head(e, (uint32_t) n, 0), std::memory_order_release);
    publish();
    return e;
}

void ExpertPool::wait_parked(const char* what) {
    hstate_.store(kWaitParked, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0;
    std::chrono::steady_clock::time_point t0{};
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) {
        strata_cpu_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u) t0 = now;
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "maya: the CPU expert pool stalled %s (%u of %d workers parked) - stopping the engine "
                                 "so the server can start it again (issue #29)\n",
                         what, parked_.load(), n_);
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::wait_done(int n) {
    hstate_.store(kWaitDone, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0, seen = 0;
    std::chrono::steady_clock::time_point t0{};
    for (;;) {
        const uint32_t d = done_.load(std::memory_order_acquire);
        if (d >= (uint32_t) n) return;                                // `>=`: never a wait that an overshoot outlives
        strata_cpu_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u || d != seen) { t0 = now; seen = d; }     // progress restarts the clock
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "maya: the CPU expert pool stalled: %u of %d jobs done, %u of %d workers parked - "
                                 "stopping the engine so the server can start it again (issue #29)\n",
                         d, n, parked_.load(), n_);
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::drain(int id, ExpertScratch& scratch, uint32_t epoch) {
    (void) id;
    for (;;) {
        const int ci = claim(epoch);
        if (ci < 0) break;
        const uint32_t i = (uint32_t) ci;
        if (id >= 0) wstate_[(size_t) id].store(ci, std::memory_order_relaxed);
        else { hstate_.store(ci, std::memory_order_relaxed); hstate_ms_.store(now_ms(), std::memory_order_relaxed); }
        if (mode_ == 0) {
            const ExpertJob& j = jobs_[i];
            s2_expert_vnni_q(j.blob, *j.act, j.out, scratch);
        } else if (mode_ == 1) {
            const int e = (int) i / parts_a_, part = (int) i % parts_a_;
            const int r0 = FF * part / parts_a_, r1 = FF * (part + 1) / parts_a_;
            s2_expert_gu_rows(jobs_[e].blob, *jobs_[e].act, split_[(size_t) e].ff, r0, r1);
        } else if (mode_ == 2) {
            const int e = (int) i / parts_b_, part = (int) i % parts_b_;
            const int r0 = H * part / parts_b_, r1 = H * (part + 1) / parts_b_;
            s2_expert_down_rows(jobs_[e].blob, split_[(size_t) e].a2, jobs_[e].out, r0, r1);
        } else if (mode_ >= 5) {
            // plan v0.3 P6: native layers, 5 = gate/up rows, 6 = down rows
            const int per = mode_ == 5 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                // Own Q2_0 row kernels need AVX2/AVX-512; pre-AVX2 x86 falls through to ggml-cpu.
                const bool q2_simd = q2_own_kernels() && (cpu_avx2_ok() || cpu_avx512_ok());
                if (mode_ == 5 && nfmt_->gu_type == 42 && q2_simd) {
                    // a native Q2_0 pack: gate and up rows on the Q2_0 kernels, then SwiGLU
                    thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF];
                    float* gp[MAXT];
                    float* up[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
                    const int nbk = (int) (nfmt_->n_embd / 64);
                    q2_rows_any(mjobs_[e].blob, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, gp, r0, r1);
                    q2_rows_any(mjobs_[e].blob + nfmt_->up_off, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, up, r0, r1);
                    for (int t = 0; t < mjobs_[e].nt; ++t)
                        for (int r = r0; r < r1; ++r)
                            sb.ff[t][r] = (gbuf[t][r] / (1.f + std::exp(-gbuf[t][r]))) * ubuf[t][r];
                } else if (mode_ == 5) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    native_gu_rows(*nfmt_, mjobs_[e].blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
                } else if (nfmt_->d_type == 42 && q2_simd) {
                    // Q2_0 down (most IQ layers): the AVX-512 kernel, ggml-cpu has only a scalar one on x86
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    q2_rows_any(mjobs_[e].blob + nfmt_->down_off, nfmt_->d_row, (int) (nfmt_->n_ff / 64), a2,
                                mjobs_[e].nt, mjobs_[e].out, r0, r1);
                } else {
                    const void* hq[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) hq[t] = sb.hq[t];
                    native_down_rows(*nfmt_, mjobs_[e].blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        } else {
            // plan v0.3 P6: an equal range of the phase's rows across ALL its experts (a range may span two)
            const int per = mode_ == 3 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 3) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    s2_expert_gu_rows_multi(mjobs_[e].blob, mjobs_[e].act, mjobs_[e].nt, ff, r0, r1);
                } else {
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    s2_expert_down_rows_multi(mjobs_[e].blob, a2, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        }
        done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertPool::run_phase(int mode, int n_tasks) {
    wait_parked("before a phase");
    mode_ = mode;
    njobs_ = n_tasks;
    const uint32_t e = begin_batch(n_tasks);
    if (host_works_) drain(-1, host_scratch_, e);
    wait_done(n_tasks);
    wait_parked("after a phase");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
}

void ExpertPool::run_split(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplit || n_ == 1 || expert_oracle_q8_0_enabled()) { run(jobs, n); return; }
    const auto t0 = std::chrono::steady_clock::now();
    jobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    // about three tasks per thread in each phase, so the tail is short
    parts_a_ = (std::max)(1, (3 * threads + n - 1) / n);
    parts_b_ = parts_a_;
    run_phase(1, n * parts_a_);
    for (int e = 0; e < n; ++e) act_quant_q8_1(split_[(size_t) e].ff, FF, split_[(size_t) e].a2);
    run_phase(2, n * parts_b_);
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi(ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplitMulti || expert_oracle_q8_0_enabled()) {
        // one token at a time through the single-token path (the oracle contract has no multi kernel)
        std::vector<ExpertJob> single;
        for (int e = 0; e < n; ++e)
            for (int t = 0; t < jobs[e].nt; ++t) {
                ExpertJob j;
                j.blob = jobs[e].blob;
                j.act = jobs[e].act[t];
                j.out = jobs[e].out[t];
                single.push_back(j);
            }
        for (size_t i = 0; i < single.size(); i += kMaxSplit)
            run_split(single.data() + i, (int) (std::min)((size_t) kMaxSplit, single.size() - i));
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    mjobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    mtasks_ = 3 * threads;
    mrows_ = (int64_t) n * FF;
    run_phase(3, mtasks_);
    const auto t1 = std::chrono::steady_clock::now();
    for (int e = 0; e < n; ++e)
        for (int t = 0; t < jobs[e].nt; ++t)
            act_quant_q8_1(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
    const auto t2 = std::chrono::steady_clock::now();
    mrows_ = (int64_t) n * H;
    run_phase(4, mtasks_);
    const auto t3 = std::chrono::steady_clock::now();
    ms_multi_gu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ms_multi_q += std::chrono::duration<double, std::milli>(t2 - t1).count();
    ms_multi_down += std::chrono::duration<double, std::milli>(t3 - t2).count();
    multi_bytes += (int64_t) n * (int64_t) BLOB;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::quant_acts_native(const NativeFmt& f, const float* x_f, int64_t H_dim, uint32_t tok_mask,
                                   int n_tok, uint8_t* dst_base, size_t stride_bytes) {
    int nq = 0;
    for (int t = 0; t < n_tok && t < MAXT; ++t) {
        if (tok_mask & (1u << (unsigned) t)) actq_toks_[nq++] = t;
    }
    if (nq <= 0) return;
    static const bool use_fast = [] {
        const char* v = std::getenv("MAYA_POOL_FAST");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (nq == 1 || !use_fast) {
        for (int i = 0; i < nq; ++i) {
            const int t = actq_toks_[i];
            native_quant_act(f, x_f + (size_t) t * (size_t) H_dim, dst_base + (size_t) t * stride_bytes);
        }
        return;
    }
    nfmt_ = &f;
    actq_x_ = x_f;
    actq_H_ = H_dim;
    actq_dst_ = dst_base;
    actq_stride_ = stride_bytes;
    actq_n_ = nq;
    // Wake BOTH sockets together so Socket 1 pre-warms out of C6 sleep in parallel with Socket 0 quant!
    run_fast_phase(8, -1);
}

void ExpertPool::run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n, int layer_hint) {
    if (n <= 0) return;
    static const bool use_fast = [] {
        const char* v = std::getenv("MAYA_POOL_FAST");
        return v == nullptr || std::atoi(v) != 0;
    }();
    static const bool numa_local = [] {
        const char* v = std::getenv("MAYA_POOL_NUMA_LOCAL");
        return v != nullptr && std::atoi(v) != 0;
    }();
    static const int numa_split_l = [] {
        const char* v = std::getenv("MAYA_NUMA_SPLIT_LAYER");
        return v ? std::atoi(v) : 24;
    }();
    const int target_node = (numa_local && n_numa_nodes_ == 2 && layer_hint >= 0)
                            ? (layer_hint >= numa_split_l ? 1 : 0) : -1;
    const auto t0 = std::chrono::steady_clock::now();
    // more distinct experts than buffers: run them in batches
    for (int b0 = 0; b0 < n; b0 += kMaxSplitMulti) {
        const int nb = (std::min)(kMaxSplitMulti, n - b0);
        mjobs_ = jobs + b0;
        nfmt_ = &f;
        fast_nb_ = nb;
        if (use_fast) {
            const bool q2_simd = q2_own_kernels() && (cpu_avx2_ok() || cpu_avx512_ok());
            int qk = 0;
            size_t qts = 0;
            native_quant_h_info(f, qk, qts);
            int total_q = 0;
            for (int e = 0; e < nb; ++e) total_q += mjobs_[e].nt;
            const bool fused_q = (nb >= 2 && total_q >= 3) &&
                                 !(f.gu_type == 42 && q2_simd) && !(f.d_type == 42 && q2_simd) &&
                                 qk > 0 && qk <= 64 && (FF % qk) == 0;
            mrows_ = (int64_t) nb * FF;
            const auto a = std::chrono::steady_clock::now();
            run_fast_phase(5, target_node);
            const auto b = std::chrono::steady_clock::now();
            if (!fused_q) {
                if (total_q >= 3) {
                    run_fast_phase(7, target_node);
                } else {
                    const bool q2_d = (f.d_type == 42) && q2_simd;
                    for (int e = 0; e < nb; ++e)
                        for (int t = 0; t < mjobs_[e].nt; ++t)
                            if (q2_d) act_quant_any(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
                            else native_quant_h(f, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);
                }
            }
            const auto c = std::chrono::steady_clock::now();
            mrows_ = (int64_t) nb * H;
            run_fast_phase(6, target_node);
            const auto d = std::chrono::steady_clock::now();
            ms_multi_gu += std::chrono::duration<double, std::milli>(b - a).count();
            ms_multi_q += std::chrono::duration<double, std::milli>(c - b).count();
            ms_multi_down += std::chrono::duration<double, std::milli>(d - c).count();
        } else {
            const int threads = n_ + (host_works_ ? 1 : 0);
            static const int task_mult = [] {
                const char* v = std::getenv("MAYA_POOL_TASKS_MULT");
                return v ? std::max(1, std::min(4, std::atoi(v))) : 3;
            }();
            mrows_ = (int64_t) nb * FF;
            mtasks_ = std::max(1, std::min<int>((int) ((mrows_ + 15) / 16), task_mult * threads));
            const auto a = std::chrono::steady_clock::now();
            run_phase(5, mtasks_);
            const auto b = std::chrono::steady_clock::now();
            for (int e = 0; e < nb; ++e)
                for (int t = 0; t < mjobs_[e].nt; ++t)
                    if (f.d_type == 42 && q2_own_kernels() && (cpu_avx2_ok() || cpu_avx512_ok()))
                        act_quant_any(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
                    else native_quant_h(f, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);
            const auto c = std::chrono::steady_clock::now();
            mrows_ = (int64_t) nb * H;
            mtasks_ = std::max(1, std::min<int>((int) ((mrows_ + 15) / 16), task_mult * threads));
            run_phase(6, mtasks_);
            const auto d = std::chrono::steady_clock::now();
            ms_multi_gu += std::chrono::duration<double, std::milli>(b - a).count();
            ms_multi_q += std::chrono::duration<double, std::milli>(c - b).count();
            ms_multi_down += std::chrono::duration<double, std::milli>(d - c).count();
        }
    }
    multi_bytes += (int64_t) n * (int64_t) f.bytes;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n_ == 1) {   // no workers: run inline, so a single-core machine still produces a token
        for (int i = 0; i < n; ++i) s2_expert_vnni_q(jobs[i].blob, *jobs[i].act, jobs[i].out, scratch_[0]);
        return;
    }
    // Wait for every worker to be parked BEFORE touching the batch, so the publish below is the only thing
    // that can move a worker into the drain loop.
    //
    // THE THREE PHASES ARE TIMED SEPARATELY.  They were one number, which cannot distinguish a pool that is
    // slow at the WORK from one that is slow at the SYNCHRONISATION - and those need opposite fixes.
    const auto t_a = std::chrono::steady_clock::now();
    wait_parked("before a batch");
    const auto t_b = std::chrono::steady_clock::now();
    jobs_ = jobs;
    njobs_ = n;
    mode_ = 0;
    const uint32_t e = begin_batch(n);   // done, then head (release), then the epoch: the batch is described first

    // ---- **THE HOST DRAINS TOO (R2.2), INSTEAD OF SPINNING ON `done_`.**
    //
    // The loop below used to be `while (done_ != n) strata_cpu_pause();`.  The host is pinned to core 0 - the core
    // `physical_cores(true)` deliberately keeps the five workers off - so for the whole drain that core was
    // idle while five cores did six cores' worth of work.  Measured before the change: 33.7 GB/s against
    // 5/6 x 44.14 = 36.8 for five workers and 44.14 for six.
    //
    // The host claims through the SAME `head_` counter, so this is not a second scheduler and nothing about
    // the ordering changes: `head_` is a single `fetch_add`, every job is the same size, and a thread that
    // arrives late simply claims nothing.  `done_` is still the completion signal and the host still waits for
    // it - what changed is only that the host arrives at that wait having done a share of the work.
    //
    // The host's `done_.fetch_add` is a release for the same reason a worker's is: `j.out` is read by the
    // device after `run()` returns, so the write must be published, not merely performed.
    if (host_works_) {
        for (;;) {
            const int ci = claim(e);
            if (ci < 0) break;
            hstate_.store(ci, std::memory_order_relaxed);
            hstate_ms_.store(now_ms(), std::memory_order_relaxed);
            const ExpertJob& j = jobs_[ci];
            s2_expert_vnni_q(j.blob, *j.act, j.out, host_scratch_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    wait_done(n);
    // And park again, so the next `run` starts from a known state.  See the header for why `done` alone is
    // not enough.
    const auto t_c = std::chrono::steady_clock::now();
    wait_parked("after a batch");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    const auto t_d = std::chrono::steady_clock::now();

    ms_wait_park_ += std::chrono::duration<double, std::milli>(t_b - t_a).count();
    ms_drain_ += std::chrono::duration<double, std::milli>(t_c - t_b).count();
    ms_repark_ += std::chrono::duration<double, std::milli>(t_d - t_c).count();
}

}  // namespace strata::kernels::cpu
