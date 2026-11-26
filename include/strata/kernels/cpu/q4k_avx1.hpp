// include/strata/kernels/cpu/q4k_avx1.hpp - Ivy Bridge / Sandy Bridge (AVX1 + F16C + SSSE3) multi-token Q4_K x Q8_K
// expert row kernels. On CPUs without AVX2/AVX-512, native_expert.cpp previously fell back to ggml-cpu's 1-token
// vec_dot (re-reading and re-unpacking each 144-byte block_q4_K for every token). These kernels unpack each block_q4_K
// once and dot against 4-token register tiles (fitting in the 16 XMM registers with zero spills).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Runtime CPUID check: OSXSAVE + YMM state + SSSE3 + SSE4.1 + AVX + F16C.
bool cpu_avx1_f16c_ok() noexcept;

/// Fast multi-token Q4_K x Q8_K rows (exact int16 pre-add halving pmaddwd, ~1e-6 fp32 sum-order diff vs ggml).
/// Signature matches native_expert.cpp's RowsFn.
void q4k_avx1_rows_fast(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                        float* const* out, int r0, int r1);

/// Bit-for-bit identical multi-token Q4_K x Q8_K rows matching ggml-cpu's __AVX__ vec_dot_q4_K_q8_K.
void q4k_avx1_rows_exact(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                         float* const* out, int r0, int r1);

/// Fast fused Gate+Up+SwiGLU Q4_K x Q8_K (16-row L1d tile, precomputes Q8_K bsums once for both gate & up).
void q4k_avx1_gu_swiglu(const uint8_t* w_gate, const uint8_t* w_up, size_t row_bytes, int n,
                        const void* const* act, int nt, float* const* ff, int r0, int r1);

/// Ivy Bridge AVX1 + F16C + SSSE3 4-token tiled Q5_K x Q8_K, Q5_1 x Q8_1 & Q8_0 x Q8_0 multi-token rows.
void q5k_avx1_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                   float* const* out, int r0, int r1);
void q51_avx1_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                   float* const* out, int r0, int r1);
void q80_avx1_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                   float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
