#pragma once

// Layer-split residual hops: optional GPU→GPU peer copy (MAYA_GLM_P2P=1) instead of
// device→host→device through the pinned hop buffer. Same bytes either way.
// Peer access is enabled lazily per (dst,src) pair; unsupported pairs keep the host path.
//
// Integrity: a driver can report peer access that the platform does not carry correctly
// (Tesla cards behind a Xeon PHB: copies arrive with most bytes wrong). Prefill already
// probes this for MAYA_GLM_DUAL_PREFILL; decode hops must too, or the next GPU's first
// router sees a non-finite residual ("glm router: layer N decode invalid expert ID...").

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(STRATA_USE_HIP)
#include "strata/hip_compat/cuda_runtime.h"
#else
#include <cuda_runtime.h>
#endif

namespace strata::core::glm_p2p {

// MAYA_GLM_P2P: 0 off, 1 cudaMemcpyPeer, 2 cudaMemcpy DeviceToDevice (after EnablePeerAccess).
// Both are integrity-probed; PHB-broken Tesla pairs fail the probe and stay on the host hop.
inline int mode() {
    static const int m = [] {
        const char* v = std::getenv("MAYA_GLM_P2P");
        if (v == nullptr) return 0;
        const int n = std::atoi(v);
        return n < 0 ? 0 : (n > 2 ? 2 : n);
    }();
    return m;
}

inline bool enabled() { return mode() != 0; }

// 4 MB pattern each way; returns false if enable fails or any byte differs.
inline bool peer_copies_intact(int dst_dev, int src_dev) {
    const size_t nb = (size_t) 4 << 20;
    std::vector<uint8_t> pat(nb), got(nb);
    for (size_t i = 0; i < nb; ++i) pat[i] = (uint8_t) ((i * 2654435761u) >> 13);
    void *dsrc = nullptr, *ddst = nullptr;
    int cur = 0;
    cudaGetDevice(&cur);
    bool ok = true;
    cudaSetDevice(src_dev);
    ok = ok && cudaMalloc(&dsrc, nb) == cudaSuccess;
    cudaSetDevice(dst_dev);
    ok = ok && cudaMalloc(&ddst, nb) == cudaSuccess;
    const int how = mode();  // 1 = MemcpyPeer, 2 = D2D
    for (int dir = 0; dir < 2 && ok; ++dir) {
        // dir 0: src -> dst (the hop direction); dir 1: reverse (same link quality).
        void* s = dir == 0 ? dsrc : ddst;
        void* d = dir == 0 ? ddst : dsrc;
        const int sd = dir == 0 ? src_dev : dst_dev;
        const int td = dir == 0 ? dst_dev : src_dev;
        cudaSetDevice(sd);
        ok = cudaMemcpy(s, pat.data(), nb, cudaMemcpyHostToDevice) == cudaSuccess;
        if (!ok) break;
        if (how == 2) {
            // DeviceToDevice after EnablePeerAccess — some drivers take a different path than MemcpyPeer.
            cudaSetDevice(td);
            ok = cudaMemcpy(d, s, nb, cudaMemcpyDeviceToDevice) == cudaSuccess;
        } else {
            ok = cudaMemcpyPeer(d, td, s, sd, nb) == cudaSuccess;
            cudaSetDevice(td);
        }
        ok = ok && cudaMemcpy(got.data(), d, nb, cudaMemcpyDeviceToHost) == cudaSuccess &&
             std::memcmp(got.data(), pat.data(), nb) == 0;
    }
    (void) cudaGetLastError();
    if (dsrc) {
        cudaSetDevice(src_dev);
        cudaFree(dsrc);
    }
    if (ddst) {
        cudaSetDevice(dst_dev);
        cudaFree(ddst);
    }
    cudaSetDevice(cur);
    return ok;
}

inline bool enable_peer(int dst_dev, int src_dev) {
    if (!enabled()) return false;
    if (dst_dev < 0 || dst_dev >= 16 || src_dev < 0 || src_dev >= 16 || dst_dev == src_dev) return false;
    static std::atomic<int> cache[16][16]{};  // 0 unknown, 1 ok, -1 no
    int cached = cache[dst_dev][src_dev].load(std::memory_order_acquire);
    if (cached != 0) return cached > 0;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    cached = cache[dst_dev][src_dev].load(std::memory_order_relaxed);
    if (cached != 0) return cached > 0;
    int can = 0;
    bool ok = false;
    if (cudaDeviceCanAccessPeer(&can, dst_dev, src_dev) == cudaSuccess && can != 0) {
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(dst_dev);
        const cudaError_t pe = cudaDeviceEnablePeerAccess(src_dev, 0);
        if (pe != cudaSuccess) (void) cudaGetLastError();
        ok = (pe == cudaSuccess || pe == cudaErrorPeerAccessAlreadyEnabled);
        // Also enable the reverse direction so the integrity probe can check both ways
        // (and so later hops the other way reuse this pair's enable).
        if (ok) {
            cudaSetDevice(src_dev);
            const cudaError_t pe2 = cudaDeviceEnablePeerAccess(dst_dev, 0);
            if (pe2 != cudaSuccess) (void) cudaGetLastError();
            ok = (pe2 == cudaSuccess || pe2 == cudaErrorPeerAccessAlreadyEnabled);
        }
        cudaSetDevice(cur);
        if (ok && !peer_copies_intact(dst_dev, src_dev)) {
            std::fprintf(stderr,
                         "maya: P2P residual hop CUDA%d→CUDA%d reports peer access but a peer copy came out wrong "
                         "on this machine (MAYA_GLM_P2P=%d) - keeping the host hop\n",
                         src_dev, dst_dev, mode());
            ok = false;
        }
    } else {
        (void) cudaGetLastError();
    }
    cache[dst_dev][src_dev].store(ok ? 1 : -1, std::memory_order_release);
    // Mirror the result for the reverse pair so we do not re-probe / re-enable.
    cache[src_dev][dst_dev].store(ok ? 1 : -1, std::memory_order_release);
    if (ok) {
        std::fprintf(stderr, "maya: P2P residual hop CUDA%d→CUDA%d enabled (MAYA_GLM_P2P=%d)\n", src_dev,
                     dst_dev, mode());
    }
    return ok;
}

// Record src ready, then copy residual into dst. Host path uses hop_h; peer path reads src_state directly.
inline void hop_residual(int src_dev, cudaStream_t src_cs, const void* src_state, void* hop_h,
                         cudaEvent_t ev_hop, int dst_dev, cudaStream_t dst_cs, void* dst_state, size_t bytes) {
    if (enable_peer(dst_dev, src_dev)) {
        cudaSetDevice(src_dev);
        cudaEventRecord(ev_hop, src_cs);
        cudaSetDevice(dst_dev);
        cudaStreamWaitEvent(dst_cs, ev_hop, 0);
        if (mode() == 2) {
            cudaMemcpyAsync(dst_state, src_state, bytes, cudaMemcpyDeviceToDevice, dst_cs);
        } else {
            cudaMemcpyPeerAsync(dst_state, dst_dev, src_state, src_dev, bytes, dst_cs);
        }
        return;
    }
    cudaSetDevice(src_dev);
    cudaMemcpyAsync(hop_h, src_state, bytes, cudaMemcpyDeviceToHost, src_cs);
    cudaEventRecord(ev_hop, src_cs);
    cudaSetDevice(dst_dev);
    cudaStreamWaitEvent(dst_cs, ev_hop, 0);
    cudaMemcpyAsync(dst_state, hop_h, bytes, cudaMemcpyHostToDevice, dst_cs);
}

// Split form used by speculative decode: signal (and optional D2H), then separate wait+copy on dst.
inline void hop_out(int src_dev, cudaStream_t src_cs, const void* src_state, void* hop_h, cudaEvent_t ev_hop,
                    size_t bytes, int dst_dev_for_peer) {
    cudaSetDevice(src_dev);
    if (enable_peer(dst_dev_for_peer, src_dev)) {
        cudaEventRecord(ev_hop, src_cs);
        return;
    }
    cudaMemcpyAsync(hop_h, src_state, bytes, cudaMemcpyDeviceToHost, src_cs);
    cudaEventRecord(ev_hop, src_cs);
}

inline void hop_in(int dst_dev, cudaStream_t dst_cs, void* dst_state, int src_dev, const void* src_state,
                   const void* hop_h, cudaEvent_t ev_hop, size_t bytes) {
    cudaSetDevice(dst_dev);
    cudaStreamWaitEvent(dst_cs, ev_hop, 0);
    if (enable_peer(dst_dev, src_dev)) {
        if (mode() == 2) {
            cudaMemcpyAsync(dst_state, src_state, bytes, cudaMemcpyDeviceToDevice, dst_cs);
        } else {
            cudaMemcpyPeerAsync(dst_state, dst_dev, src_state, src_dev, bytes, dst_cs);
        }
        return;
    }
    cudaMemcpyAsync(dst_state, hop_h, bytes, cudaMemcpyHostToDevice, dst_cs);
}

}  // namespace strata::core::glm_p2p
