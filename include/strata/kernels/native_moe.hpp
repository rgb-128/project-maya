#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before constructing/capturing sessions. Existing captures keep their
// selected kernels; changing this flag does not rewrite an existing graph.
void native_moe_combine_set_enabled(bool enabled);
bool native_moe_combine_enabled();

// Pinned CUDA weighted-reduction contract for one token, k in [1, 15].
// The backend fuses k=2..15; k=1 is its ordinary multiply/contiguous/add path.
// parts: k contiguous F32 rows of n_embd; weights: k F32 values. The first
// product rounds to F32, following products accumulate with FMA in expert order,
// and optional shared is added once afterward. Shared is not router weighted.
// Requires a nonnull ordered stream and disjoint output. No allocation or sync.
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream);
/// n_tok rows (parts [n,k,N], weights [n,k], shared/output [n,N]) in one launch, each as the single call.
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream);
/// Resident fast-path: reads directly from `hit_out` when `*skip == ring` (bypassing copy_or_zero + moe_hit_add),
/// otherwise reads from `parts` (exact miss path).
void native_moe_combine_multi_or_hit(const float* hit_out, const float* parts, const float* weights,
                                     const float* shared, float* output, const uint32_t* skip, uint32_t ring,
                                     int64_t n_embd, int64_t k, int n_tok, void* stream);
}
