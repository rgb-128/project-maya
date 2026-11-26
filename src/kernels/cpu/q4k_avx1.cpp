// src/kernels/cpu/q4k_avx1.cpp - Ivy Bridge (AVX1 + F16C + SSSE3) multi-token Q4_K x Q8_K expert row kernels.
// Compiled with -mavx -mf16c -ffp-contract=off; entered only after cpu_avx1_f16c_ok() (called from native_expert.cpp,
// which has no AVX compile flags).
#include "strata/kernels/cpu/q4k_avx1.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace strata::kernels::cpu {
namespace {

struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(BlockQ4K) == 144, "block_q4_K size");

struct BlockQ8K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(BlockQ8K) == 292, "block_q8_K size");

inline float h2f_f16c(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}

inline float hsum_f4(__m128 v) {
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_movehdup_ps(v));
    return _mm_cvtss_f32(v);
}

inline float hsum_f8(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    return hsum_f4(r);
}

// Fast multi-token Q4_K x Q8_K for NT in {1..4}:
// Unpacks each 144-byte block_q4_K ONCE and pre-adds p16l_0 + p16l_1 (exact int16: max |val| <= 7680 << 32767),
// halving pmaddwd (16 -> 8 per block) and keeping all NT=4 accumulators inside Ivy Bridge's 16 XMM registers.
template<int NT>
void q4k_fast_rows_t(const uint8_t* w, size_t row_bytes, int nb, const BlockQ8K* const* y, float* const* out,
                     int r0, int r1) {
    static constexpr uint32_t kmask1 = 0x3f3f3f3fu, kmask2 = 0x0f0f0f0fu, kmask3 = 0x03030303u;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i m2 = _mm_set1_epi8(0x02);
    // Precompute q8s = hadd_epi16(bsums[0..7], bsums[8..15]) once per (token, block) instead of per weight row.
    __m128i q8s_pre[NT][32];
    const bool use_pre = nb <= 32;
    if (use_pre) {
        for (int t = 0; t < NT; ++t)
            for (int i = 0; i < nb; ++i)
                q8s_pre[t][i] = _mm_hadd_epi16(_mm_loadu_si128((const __m128i*) &y[t][i].bsums[0]),
                                               _mm_loadu_si128((const __m128i*) &y[t][i].bsums[8]));
    }
    const char* const end_ptr = (const char*) (w + (size_t) r1 * row_bytes);
    for (int r = r0; r < r1; ++r) {
        const BlockQ4K* __restrict x = (const BlockQ4K*) (w + (size_t) r * row_bytes);
        __m128 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const char* pf = (const char*) (&x[i + 2]);
            if (pf < end_ptr) _mm_prefetch(pf, _MM_HINT_T0);
            const float xd = h2f_f16c(x[i].d);
            const float xdmin = -h2f_f16c(x[i].dmin);
            uint32_t utmp[4];
            std::memcpy(utmp, x[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
            const __m128i utmps = _mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]);
            const __m128i scales = _mm_cvtepu8_epi16(utmps);
            const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));
            for (int t = 0; t < NT; ++t) {
                const __m128i q8s = use_pre ? q8s_pre[t][i]
                                            : _mm_hadd_epi16(_mm_loadu_si128((const __m128i*) &y[t][i].bsums[0]),
                                                             _mm_loadu_si128((const __m128i*) &y[t][i].bsums[8]));
                const __m128i prod = _mm_madd_epi16(mins, q8s);
                acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_set1_ps(y[t][i].d * xdmin), _mm_cvtepi32_ps(prod)));
            }
            const uint8_t* q4 = x[i].qs;
            __m128i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm_setzero_si128();
            __m128i shuf = _mm_set1_epi16(0x0100);
            for (int j = 0; j < 4; ++j) {
                const __m128i sc_l = _mm_shuffle_epi8(scales, shuf);
                shuf = _mm_add_epi16(shuf, m2);
                const __m128i sc_h = _mm_shuffle_epi8(scales, shuf);
                shuf = _mm_add_epi16(shuf, m2);
                const __m128i b0 = _mm_loadu_si128((const __m128i*) (q4 + 0));
                const __m128i b1 = _mm_loadu_si128((const __m128i*) (q4 + 16));
                q4 += 32;
                const __m128i q4l0 = _mm_and_si128(b0, m4);
                const __m128i q4l1 = _mm_and_si128(b1, m4);
                const __m128i q4h0 = _mm_and_si128(_mm_srli_epi16(b0, 4), m4);
                const __m128i q4h1 = _mm_and_si128(_mm_srli_epi16(b1, 4), m4);
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = y[t][i].qs + j * 64;
                    const __m128i pl = _mm_add_epi16(
                        _mm_maddubs_epi16(q4l0, _mm_loadu_si128((const __m128i*) (q8 + 0))),
                        _mm_maddubs_epi16(q4l1, _mm_loadu_si128((const __m128i*) (q8 + 16))));
                    const __m128i ph = _mm_add_epi16(
                        _mm_maddubs_epi16(q4h0, _mm_loadu_si128((const __m128i*) (q8 + 32))),
                        _mm_maddubs_epi16(q4h1, _mm_loadu_si128((const __m128i*) (q8 + 48))));
                    sumi[t] = _mm_add_epi32(sumi[t], _mm_add_epi32(_mm_madd_epi16(sc_l, pl), _mm_madd_epi16(sc_h, ph)));
                }
            }
            for (int t = 0; t < NT; ++t)
                acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_set1_ps(y[t][i].d * xd), _mm_cvtepi32_ps(sumi[t])));
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum_f4(acc[t]);
    }
}

// Bit-exact multi-token Q4_K x Q8_K matching ggml-cpu's __AVX__ vec_dot_q4_K_q8_K bit for bit.
template<int NT>
void q4k_exact_rows_t(const uint8_t* w, size_t row_bytes, int nb, const BlockQ8K* const* y, float* const* out,
                      int r0, int r1) {
    static constexpr uint32_t kmask1 = 0x3f3f3f3fu, kmask2 = 0x0f0f0f0fu, kmask3 = 0x03030303u;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i m2 = _mm_set1_epi8(0x02);
    for (int r = r0; r < r1; ++r) {
        const BlockQ4K* __restrict x = (const BlockQ4K*) (w + (size_t) r * row_bytes);
        __m256 acc[NT];
        __m128 acc_m[NT];
        for (int t = 0; t < NT; ++t) {
            acc[t] = _mm256_setzero_ps();
            acc_m[t] = _mm_setzero_ps();
        }
        for (int i = 0; i < nb; ++i) {
            if (i + 2 < nb) _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            const float xd = h2f_f16c(x[i].d);
            const float xdmin = h2f_f16c(x[i].dmin);
            uint32_t utmp[4];
            std::memcpy(utmp, x[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
            const __m128i utmps = _mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]);
            const __m128i scales = _mm_cvtepu8_epi16(utmps);
            const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));
            for (int t = 0; t < NT; ++t) {
                const float dmin = -y[t][i].d * xdmin;
                const __m128i q8s = _mm_hadd_epi16(_mm_loadu_si128((const __m128i*) &y[t][i].bsums[0]),
                                                   _mm_loadu_si128((const __m128i*) &y[t][i].bsums[8]));
                const __m128i prod = _mm_madd_epi16(mins, q8s);
                acc_m[t] = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m[t]);
            }
            const uint8_t* q4 = x[i].qs;
            __m128i sum0[NT], sum1[NT];
            for (int t = 0; t < NT; ++t) {
                sum0[t] = _mm_setzero_si128();
                sum1[t] = _mm_setzero_si128();
            }
            __m128i shuf = _mm_set1_epi16(0x0100);
            for (int j = 0; j < 4; ++j) {
                const __m128i sc_l = _mm_shuffle_epi8(scales, shuf);
                shuf = _mm_add_epi16(shuf, m2);
                const __m128i sc_h = _mm_shuffle_epi8(scales, shuf);
                shuf = _mm_add_epi16(shuf, m2);
                const __m128i b0 = _mm_loadu_si128((const __m128i*) (q4 + 0));
                const __m128i b1 = _mm_loadu_si128((const __m128i*) (q4 + 16));
                q4 += 32;
                const __m128i q4l0 = _mm_and_si128(b0, m4);
                const __m128i q4l1 = _mm_and_si128(b1, m4);
                const __m128i q4h0 = _mm_and_si128(_mm_srli_epi16(b0, 4), m4);
                const __m128i q4h1 = _mm_and_si128(_mm_srli_epi16(b1, 4), m4);
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = y[t][i].qs + j * 64;
                    sum0[t] = _mm_add_epi32(sum0[t], _mm_madd_epi16(sc_l, _mm_maddubs_epi16(q4l0, _mm_loadu_si128((const __m128i*) (q8 + 0)))));
                    sum1[t] = _mm_add_epi32(sum1[t], _mm_madd_epi16(sc_l, _mm_maddubs_epi16(q4l1, _mm_loadu_si128((const __m128i*) (q8 + 16)))));
                    sum0[t] = _mm_add_epi32(sum0[t], _mm_madd_epi16(sc_h, _mm_maddubs_epi16(q4h0, _mm_loadu_si128((const __m128i*) (q8 + 32)))));
                    sum1[t] = _mm_add_epi32(sum1[t], _mm_madd_epi16(sc_h, _mm_maddubs_epi16(q4h1, _mm_loadu_si128((const __m128i*) (q8 + 48)))));
                }
            }
            for (int t = 0; t < NT; ++t) {
                const __m256 vd = _mm256_set1_ps(y[t][i].d * xd);
                const __m256i sumi = _mm256_insertf128_si256(_mm256_castsi128_si256(sum0[t]), sum1[t], 1);
                acc[t] = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum_f8(acc[t]) + hsum_f4(acc_m[t]);
    }
}

template<int NT>
void q4k_gu_swiglu_t(const uint8_t* wg, const uint8_t* wu, size_t row_bytes, int nb,
                     const BlockQ8K* const* y, float* const* ff, int r0, int r1) {
    static constexpr uint32_t kmask1 = 0x3f3f3f3fu, kmask2 = 0x0f0f0f0fu, kmask3 = 0x03030303u;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i m2 = _mm_set1_epi8(0x02);
    __m128i q8s_pre[NT][32];
    const bool use_pre = nb <= 32;
    if (use_pre) {
        for (int t = 0; t < NT; ++t)
            for (int i = 0; i < nb; ++i)
                q8s_pre[t][i] = _mm_hadd_epi16(_mm_loadu_si128((const __m128i*) &y[t][i].bsums[0]),
                                               _mm_loadu_si128((const __m128i*) &y[t][i].bsums[8]));
    }
    auto dot_row = [&](const BlockQ4K* __restrict x, const char* end_ptr, float out_t[NT]) {
        __m128 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const char* pf = (const char*) (&x[i + 2]);
            if (pf < end_ptr) _mm_prefetch(pf, _MM_HINT_T0);
            const float xd = h2f_f16c(x[i].d);
            const float xdmin = -h2f_f16c(x[i].dmin);
            uint32_t utmp[4];
            std::memcpy(utmp, x[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
            const __m128i utmps = _mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]);
            const __m128i scales = _mm_cvtepu8_epi16(utmps);
            const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));
            for (int t = 0; t < NT; ++t) {
                const __m128i q8s = use_pre ? q8s_pre[t][i]
                                            : _mm_hadd_epi16(_mm_loadu_si128((const __m128i*) &y[t][i].bsums[0]),
                                                             _mm_loadu_si128((const __m128i*) &y[t][i].bsums[8]));
                const __m128i prod = _mm_madd_epi16(mins, q8s);
                acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_set1_ps(y[t][i].d * xdmin), _mm_cvtepi32_ps(prod)));
            }
            const uint8_t* q4 = x[i].qs;
            __m128i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm_setzero_si128();
            __m128i shuf = _mm_set1_epi16(0x0100);
            for (int j = 0; j < 4; ++j) {
                const __m128i sc_l = _mm_shuffle_epi8(scales, shuf);
                shuf = _mm_add_epi16(shuf, m2);
                const __m128i sc_h = _mm_shuffle_epi8(scales, shuf);
                shuf = _mm_add_epi16(shuf, m2);
                const __m128i b0 = _mm_loadu_si128((const __m128i*) (q4 + 0));
                const __m128i b1 = _mm_loadu_si128((const __m128i*) (q4 + 16));
                q4 += 32;
                const __m128i q4l0 = _mm_and_si128(b0, m4);
                const __m128i q4l1 = _mm_and_si128(b1, m4);
                const __m128i q4h0 = _mm_and_si128(_mm_srli_epi16(b0, 4), m4);
                const __m128i q4h1 = _mm_and_si128(_mm_srli_epi16(b1, 4), m4);
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = y[t][i].qs + j * 64;
                    const __m128i pl = _mm_add_epi16(
                        _mm_maddubs_epi16(q4l0, _mm_loadu_si128((const __m128i*) (q8 + 0))),
                        _mm_maddubs_epi16(q4l1, _mm_loadu_si128((const __m128i*) (q8 + 16))));
                    const __m128i ph = _mm_add_epi16(
                        _mm_maddubs_epi16(q4h0, _mm_loadu_si128((const __m128i*) (q8 + 32))),
                        _mm_maddubs_epi16(q4h1, _mm_loadu_si128((const __m128i*) (q8 + 48))));
                    sumi[t] = _mm_add_epi32(sumi[t], _mm_add_epi32(_mm_madd_epi16(sc_l, pl), _mm_madd_epi16(sc_h, ph)));
                }
            }
            for (int t = 0; t < NT; ++t)
                acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_set1_ps(y[t][i].d * xd), _mm_cvtepi32_ps(sumi[t])));
        }
        for (int t = 0; t < NT; ++t) out_t[t] = hsum_f4(acc[t]);
    };
    const char* const end_g = (const char*) (wg + (size_t) r1 * row_bytes);
    const char* const end_u = (const char*) (wu + (size_t) r1 * row_bytes);
    for (int r = r0; r < r1; ++r) {
        float gv[NT], uv[NT];
        dot_row((const BlockQ4K*) (wg + (size_t) r * row_bytes), end_g, gv);
        dot_row((const BlockQ4K*) (wu + (size_t) r * row_bytes), end_u, uv);
        for (int t = 0; t < NT; ++t)
            ff[t][r] = (gv[t] / (1.0f + std::exp(-gv[t]))) * uv[t];
    }
}

}  // namespace

void q4k_avx1_gu_swiglu(const uint8_t* w_gate, const uint8_t* w_up, size_t row_bytes, int n,
                        const void* const* act, int nt, float* const* ff, int r0, int r1) {
    const int nb = n / 256;
    for (int rb = r0; rb < r1; rb += 16) {
        const int re = rb + 16 < r1 ? rb + 16 : r1;
        int t = 0;
        for (; t + 4 <= nt; t += 4) {
            const BlockQ8K* y[4] = {(const BlockQ8K*) act[t + 0], (const BlockQ8K*) act[t + 1],
                                    (const BlockQ8K*) act[t + 2], (const BlockQ8K*) act[t + 3]};
            q4k_gu_swiglu_t<4>(w_gate, w_up, row_bytes, nb, y, ff + t, rb, re);
        }
        const int rem = nt - t;
        if (rem == 3) {
            const BlockQ8K* y[3] = {(const BlockQ8K*) act[t + 0], (const BlockQ8K*) act[t + 1], (const BlockQ8K*) act[t + 2]};
            q4k_gu_swiglu_t<3>(w_gate, w_up, row_bytes, nb, y, ff + t, rb, re);
        } else if (rem == 2) {
            const BlockQ8K* y[2] = {(const BlockQ8K*) act[t + 0], (const BlockQ8K*) act[t + 1]};
            q4k_gu_swiglu_t<2>(w_gate, w_up, row_bytes, nb, y, ff + t, rb, re);
        } else if (rem == 1) {
            const BlockQ8K* y[1] = {(const BlockQ8K*) act[t + 0]};
            q4k_gu_swiglu_t<1>(w_gate, w_up, row_bytes, nb, y, ff + t, rb, re);
        }
    }
}

void q4k_avx1_rows_fast(int, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                        float* const* out, int r0, int r1) {
    const int nb = n / 256;
    // Tile rows in strips of 16 so the weight row stays hot in L1d when nt > 4 (16 * 2.3 KB = 36 KB -> L2).
    for (int rb = r0; rb < r1; rb += 16) {
        const int re = rb + 16 < r1 ? rb + 16 : r1;
        int t = 0;
        for (; t + 4 <= nt; t += 4) {
            const BlockQ8K* y[4] = {(const BlockQ8K*) act[t + 0], (const BlockQ8K*) act[t + 1],
                                    (const BlockQ8K*) act[t + 2], (const BlockQ8K*) act[t + 3]};
            q4k_fast_rows_t<4>(w, row_bytes, nb, y, out + t, rb, re);
        }
        const int rem = nt - t;
        if (rem == 3) {
            const BlockQ8K* y[3] = {(const BlockQ8K*) act[t + 0], (const BlockQ8K*) act[t + 1], (const BlockQ8K*) act[t + 2]};
            q4k_fast_rows_t<3>(w, row_bytes, nb, y, out + t, rb, re);
        } else if (rem == 2) {
            const BlockQ8K* y[2] = {(const BlockQ8K*) act[t + 0], (const BlockQ8K*) act[t + 1]};
            q4k_fast_rows_t<2>(w, row_bytes, nb, y, out + t, rb, re);
        } else if (rem == 1) {
            const BlockQ8K* y[1] = {(const BlockQ8K*) act[t + 0]};
            q4k_fast_rows_t<1>(w, row_bytes, nb, y, out + t, rb, re);
        }
    }
}

void q4k_avx1_rows_exact(int, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                         float* const* out, int r0, int r1) {
    const int nb = n / 256;
    for (int rb = r0; rb < r1; rb += 16) {
        const int re = rb + 16 < r1 ? rb + 16 : r1;
        int t = 0;
        for (; t + 4 <= nt; t += 4) {
            const BlockQ8K* y[4] = {(const BlockQ8K*) act[t + 0], (const BlockQ8K*) act[t + 1],
                                    (const BlockQ8K*) act[t + 2], (const BlockQ8K*) act[t + 3]};
            q4k_exact_rows_t<4>(w, row_bytes, nb, y, out + t, rb, re);
        }
        const int rem = nt - t;
        if (rem == 3) {
            const BlockQ8K* y[3] = {(const BlockQ8K*) act[t + 0], (const BlockQ8K*) act[t + 1], (const BlockQ8K*) act[t + 2]};
            q4k_exact_rows_t<3>(w, row_bytes, nb, y, out + t, rb, re);
        } else if (rem == 2) {
            const BlockQ8K* y[2] = {(const BlockQ8K*) act[t + 0], (const BlockQ8K*) act[t + 1]};
            q4k_exact_rows_t<2>(w, row_bytes, nb, y, out + t, rb, re);
        } else if (rem == 1) {
            const BlockQ8K* y[1] = {(const BlockQ8K*) act[t + 0]};
            q4k_exact_rows_t<1>(w, row_bytes, nb, y, out + t, rb, re);
        }
    }
}

namespace {

struct BlockQ5K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
static_assert(sizeof(BlockQ5K) == 176, "block_q5_K size");

template<int NT>
void q5k_avx1_rows_t(const uint8_t* w, size_t row_bytes, int nb, const BlockQ8K* const* y, float* const* out,
                     int r0, int r1) {
    static constexpr uint32_t kmask1 = 0x3f3f3f3fu, kmask2 = 0x0f0f0f0fu, kmask3 = 0x03030303u;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i m2 = _mm_set1_epi8(0x02);
    const __m128i bit4 = _mm_set1_epi8(0x10);
    __m128i q8s_pre[NT][32];
    const bool use_pre = nb <= 32;
    if (use_pre) {
        for (int t = 0; t < NT; ++t)
            for (int i = 0; i < nb; ++i)
                q8s_pre[t][i] = _mm_hadd_epi16(_mm_loadu_si128((const __m128i*) &y[t][i].bsums[0]),
                                               _mm_loadu_si128((const __m128i*) &y[t][i].bsums[8]));
    }
    const char* const end_ptr = (const char*) (w + (size_t) r1 * row_bytes);
    for (int r = r0; r < r1; ++r) {
        const BlockQ5K* __restrict x = (const BlockQ5K*) (w + (size_t) r * row_bytes);
        __m128 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const char* pf = (const char*) (&x[i + 2]);
            if (pf < end_ptr) _mm_prefetch(pf, _MM_HINT_T0);
            const float xd = h2f_f16c(x[i].d);
            const float xdmin = -h2f_f16c(x[i].dmin);
            uint32_t utmp[4];
            std::memcpy(utmp, x[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
            const __m128i utmps = _mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]);
            const __m128i scales = _mm_cvtepu8_epi16(utmps);
            const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));
            for (int t = 0; t < NT; ++t) {
                const __m128i q8s = use_pre ? q8s_pre[t][i]
                                            : _mm_hadd_epi16(_mm_loadu_si128((const __m128i*) &y[t][i].bsums[0]),
                                                             _mm_loadu_si128((const __m128i*) &y[t][i].bsums[8]));
                const __m128i prod = _mm_madd_epi16(mins, q8s);
                acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_set1_ps(y[t][i].d * xdmin), _mm_cvtepi32_ps(prod)));
            }
            const __m128i qh0 = _mm_loadu_si128((const __m128i*) (x[i].qh + 0));
            const __m128i qh1 = _mm_loadu_si128((const __m128i*) (x[i].qh + 16));
            const uint8_t* q4 = x[i].qs;
            __m128i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm_setzero_si128();
            __m128i shuf = _mm_set1_epi16(0x0100);
            for (int j = 0; j < 4; ++j) {
                const __m128i sc_l = _mm_shuffle_epi8(scales, shuf);
                shuf = _mm_add_epi16(shuf, m2);
                const __m128i sc_h = _mm_shuffle_epi8(scales, shuf);
                shuf = _mm_add_epi16(shuf, m2);
                const __m128i u1 = _mm_set1_epi8((char) (1 << (2 * j)));
                const __m128i u2 = _mm_set1_epi8((char) (2 << (2 * j)));
                const __m128i b0 = _mm_loadu_si128((const __m128i*) (q4 + 0));
                const __m128i b1 = _mm_loadu_si128((const __m128i*) (q4 + 16));
                q4 += 32;
                const __m128i h0l = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qh0, u1), u1), bit4);
                const __m128i h1l = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qh1, u1), u1), bit4);
                const __m128i h0h = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qh0, u2), u2), bit4);
                const __m128i h1h = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qh1, u2), u2), bit4);
                const __m128i q5l0 = _mm_or_si128(_mm_and_si128(b0, m4), h0l);
                const __m128i q5l1 = _mm_or_si128(_mm_and_si128(b1, m4), h1l);
                const __m128i q5h0 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(b0, 4), m4), h0h);
                const __m128i q5h1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(b1, 4), m4), h1h);
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = y[t][i].qs + j * 64;
                    const __m128i pl = _mm_add_epi16(
                        _mm_maddubs_epi16(q5l0, _mm_loadu_si128((const __m128i*) (q8 + 0))),
                        _mm_maddubs_epi16(q5l1, _mm_loadu_si128((const __m128i*) (q8 + 16))));
                    const __m128i ph = _mm_add_epi16(
                        _mm_maddubs_epi16(q5h0, _mm_loadu_si128((const __m128i*) (q8 + 32))),
                        _mm_maddubs_epi16(q5h1, _mm_loadu_si128((const __m128i*) (q8 + 48))));
                    sumi[t] = _mm_add_epi32(sumi[t], _mm_add_epi32(_mm_madd_epi16(sc_l, pl), _mm_madd_epi16(sc_h, ph)));
                }
            }
            for (int t = 0; t < NT; ++t)
                acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_set1_ps(y[t][i].d * xd), _mm_cvtepi32_ps(sumi[t])));
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum_f4(acc[t]);
    }
}

struct BlockQ51 { uint16_t d; uint16_t m; uint8_t qh[4]; uint8_t qs[16]; };
struct BlockQ81 { uint16_t d; uint16_t s; int8_t qs[32]; };
struct BlockQ80 { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(BlockQ51) == 24, "");
static_assert(sizeof(BlockQ81) == 36, "");
static_assert(sizeof(BlockQ80) == 34, "");

static inline float fp16_to_f32(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}

static inline int hsum_i32_128(__m128i v) {
    __m128i s = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

template <int NT>
static void q51_avx1_rows_t(const uint8_t* __restrict__ w, size_t row_bytes, int nb,
                            const BlockQ81* const* __restrict__ y, float* const* __restrict__ out, int r0, int r1) {
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m128i mask_bit = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 128, 1, 2, 4, 8, 16, 32, 64, (char) 128);
    const __m128i shuf_lo = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m128i shuf_hi = _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m128i bit4 = _mm_set1_epi8(0x10);
    const __m128i ones16 = _mm_set1_epi16(1);

    alignas(16) float yd_pre[NT][64];
    alignas(16) float ys_pre[NT][64];
    const bool use_pre = nb <= 64;
    if (use_pre) {
        for (int t = 0; t < NT; ++t) {
            for (int b = 0; b < nb; ++b) {
                yd_pre[t][b] = fp16_to_f32(y[t][b].d);
                ys_pre[t][b] = fp16_to_f32(y[t][b].s);
            }
        }
    }

    for (int r = r0; r < r1; ++r) {
        const BlockQ51* __restrict__ x = reinterpret_cast<const BlockQ51*>(w + (size_t) r * row_bytes);
        if (r + 2 < r1) {
            const char* pf = (const char*) (w + (size_t) (r + 2) * row_bytes);
            _mm_prefetch(pf + 0, _MM_HINT_T0);
            _mm_prefetch(pf + 128, _MM_HINT_T0);
            _mm_prefetch(pf + 256, _MM_HINT_T0);
            _mm_prefetch(pf + 384, _MM_HINT_T0);
        }
        __m128 acc4[NT];
        for (int t = 0; t < NT; ++t) acc4[t] = _mm_setzero_ps();

        int b = 0;
        for (; b + 4 <= nb; b += 4) {
            uint32_t dm0, dm1, dm2, dm3;
            std::memcpy(&dm0, &x[b + 0].d, 4);
            std::memcpy(&dm1, &x[b + 1].d, 4);
            std::memcpy(&dm2, &x[b + 2].d, 4);
            std::memcpy(&dm3, &x[b + 3].d, 4);
            const __m256 f8 = _mm256_cvtph_ps(_mm_set_epi32((int) dm3, (int) dm2, (int) dm1, (int) dm0));
            const __m128 f_lo = _mm256_castps256_ps128(f8);
            const __m128 f_hi = _mm256_extractf128_ps(f8, 1);
            const __m128 dw4 = _mm_shuffle_ps(f_lo, f_hi, _MM_SHUFFLE(2, 0, 2, 0));
            const __m128 mw4 = _mm_shuffle_ps(f_lo, f_hi, _MM_SHUFFLE(3, 1, 3, 1));

            __m128i s[4][NT];
            for (int u = 0; u < 4; ++u) {
                uint32_t qh32;
                std::memcpy(&qh32, x[b + u].qh, 4);
                const __m128i vqh = _mm_cvtsi32_si128((int) qh32);
                const __m128i hlo = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_shuffle_epi8(vqh, shuf_lo), mask_bit), mask_bit), bit4);
                const __m128i hhi = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_shuffle_epi8(vqh, shuf_hi), mask_bit), mask_bit), bit4);
                const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[b + u].qs));
                const __m128i wlo = _mm_or_si128(_mm_and_si128(q, mask_lo), hlo);
                const __m128i whi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q, 4), mask_lo), hhi);
                for (int t = 0; t < NT; ++t) {
                    const BlockQ81& yt = y[t][b + u];
                    const __m128i p0 = _mm_maddubs_epi16(wlo, _mm_loadu_si128(reinterpret_cast<const __m128i*>(yt.qs)));
                    const __m128i p1 = _mm_maddubs_epi16(whi, _mm_loadu_si128(reinterpret_cast<const __m128i*>(yt.qs + 16)));
                    s[u][t] = _mm_madd_epi16(_mm_add_epi16(p0, p1), ones16);
                }
            }
            for (int t = 0; t < NT; ++t) {
                const __m128i sum4 = _mm_hadd_epi32(_mm_hadd_epi32(s[0][t], s[1][t]),
                                                    _mm_hadd_epi32(s[2][t], s[3][t]));
                const __m128 yd4 = use_pre ? _mm_loadu_ps(&yd_pre[t][b])
                                           : _mm_set_ps(fp16_to_f32(y[t][b + 3].d), fp16_to_f32(y[t][b + 2].d),
                                                        fp16_to_f32(y[t][b + 1].d), fp16_to_f32(y[t][b + 0].d));
                const __m128 ys4 = use_pre ? _mm_loadu_ps(&ys_pre[t][b])
                                           : _mm_set_ps(fp16_to_f32(y[t][b + 3].s), fp16_to_f32(y[t][b + 2].s),
                                                        fp16_to_f32(y[t][b + 1].s), fp16_to_f32(y[t][b + 0].s));
                acc4[t] = _mm_add_ps(acc4[t],
                    _mm_add_ps(_mm_mul_ps(_mm_mul_ps(dw4, yd4), _mm_cvtepi32_ps(sum4)),
                               _mm_mul_ps(mw4, ys4)));
            }
        }
        float acc_rem[NT];
        for (int t = 0; t < NT; ++t) acc_rem[t] = hsum_f4(acc4[t]);
        for (; b < nb; ++b) {
            const float dw = fp16_to_f32(x[b].d);
            const float mw = fp16_to_f32(x[b].m);
            uint32_t qh32;
            std::memcpy(&qh32, x[b].qh, 4);
            const __m128i vqh = _mm_cvtsi32_si128((int) qh32);
            const __m128i hlo = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_shuffle_epi8(vqh, shuf_lo), mask_bit), mask_bit), bit4);
            const __m128i hhi = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_shuffle_epi8(vqh, shuf_hi), mask_bit), mask_bit), bit4);
            const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[b].qs));
            const __m128i wlo = _mm_or_si128(_mm_and_si128(q, mask_lo), hlo);
            const __m128i whi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q, 4), mask_lo), hhi);
            for (int t = 0; t < NT; ++t) {
                const BlockQ81& yt = y[t][b];
                const __m128i p0 = _mm_maddubs_epi16(wlo, _mm_loadu_si128(reinterpret_cast<const __m128i*>(yt.qs)));
                const __m128i p1 = _mm_maddubs_epi16(whi, _mm_loadu_si128(reinterpret_cast<const __m128i*>(yt.qs + 16)));
                const int sumi = hsum_i32_128(_mm_madd_epi16(_mm_add_epi16(p0, p1), ones16));
                const float ydf = use_pre ? yd_pre[t][b] : fp16_to_f32(yt.d);
                const float ysf = use_pre ? ys_pre[t][b] : fp16_to_f32(yt.s);
                acc_rem[t] += dw * ydf * (float) sumi + mw * ysf;
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = acc_rem[t];
    }
}

template <int NT>
static void q80_avx1_rows_t(const uint8_t* __restrict__ w, size_t row_bytes, int nb,
                            const BlockQ80* const* __restrict__ y, float* const* __restrict__ out, int r0, int r1) {
    const __m128i ones16 = _mm_set1_epi16(1);
    alignas(16) float yd_pre[NT][64];
    const bool use_pre = nb <= 64;
    if (use_pre) {
        for (int t = 0; t < NT; ++t)
            for (int b = 0; b < nb; ++b)
                yd_pre[t][b] = fp16_to_f32(y[t][b].d);
    }
    for (int r = r0; r < r1; ++r) {
        const BlockQ80* __restrict__ x = reinterpret_cast<const BlockQ80*>(w + (size_t) r * row_bytes);
        if (r + 2 < r1) {
            const char* pf = (const char*) (w + (size_t) (r + 2) * row_bytes);
            _mm_prefetch(pf + 0, _MM_HINT_T0);
            _mm_prefetch(pf + 192, _MM_HINT_T0);
            _mm_prefetch(pf + 384, _MM_HINT_T0);
            _mm_prefetch(pf + 576, _MM_HINT_T0);
        }
        __m128 acc4[NT];
        for (int t = 0; t < NT; ++t) acc4[t] = _mm_setzero_ps();

        int b = 0;
        for (; b + 4 <= nb; b += 4) {
            const __m128i h4 = _mm_set_epi16(0, 0, 0, 0, (short) x[b + 3].d, (short) x[b + 2].d,
                                             (short) x[b + 1].d, (short) x[b + 0].d);
            const __m128 dw4 = _mm_cvtph_ps(h4);
            __m128i s[4][NT];
            for (int u = 0; u < 4; ++u) {
                const __m128i w0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[b + u].qs));
                const __m128i w1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[b + u].qs + 16));
                const __m128i aw0 = _mm_abs_epi8(w0);
                const __m128i aw1 = _mm_abs_epi8(w1);
                for (int t = 0; t < NT; ++t) {
                    const BlockQ80& yt = y[t][b + u];
                    const __m128i sx0 = _mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(yt.qs)), w0);
                    const __m128i sx1 = _mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(yt.qs + 16)), w1);
                    const __m128i p0 = _mm_madd_epi16(_mm_maddubs_epi16(aw0, sx0), ones16);
                    const __m128i p1 = _mm_madd_epi16(_mm_maddubs_epi16(aw1, sx1), ones16);
                    s[u][t] = _mm_add_epi32(p0, p1);
                }
            }
            for (int t = 0; t < NT; ++t) {
                const __m128i sum4 = _mm_hadd_epi32(_mm_hadd_epi32(s[0][t], s[1][t]),
                                                    _mm_hadd_epi32(s[2][t], s[3][t]));
                const __m128 yd4 = use_pre ? _mm_loadu_ps(&yd_pre[t][b])
                                           : _mm_set_ps(fp16_to_f32(y[t][b + 3].d), fp16_to_f32(y[t][b + 2].d),
                                                        fp16_to_f32(y[t][b + 1].d), fp16_to_f32(y[t][b + 0].d));
                acc4[t] = _mm_add_ps(acc4[t], _mm_mul_ps(_mm_mul_ps(dw4, yd4), _mm_cvtepi32_ps(sum4)));
            }
        }
        float acc_rem[NT];
        for (int t = 0; t < NT; ++t) acc_rem[t] = hsum_f4(acc4[t]);
        for (; b < nb; ++b) {
            const float dw = fp16_to_f32(x[b].d);
            const __m128i w0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[b].qs));
            const __m128i w1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[b].qs + 16));
            const __m128i aw0 = _mm_abs_epi8(w0);
            const __m128i aw1 = _mm_abs_epi8(w1);
            for (int t = 0; t < NT; ++t) {
                const BlockQ80& yt = y[t][b];
                const __m128i sx0 = _mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(yt.qs)), w0);
                const __m128i sx1 = _mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(yt.qs + 16)), w1);
                const __m128i p0 = _mm_madd_epi16(_mm_maddubs_epi16(aw0, sx0), ones16);
                const __m128i p1 = _mm_madd_epi16(_mm_maddubs_epi16(aw1, sx1), ones16);
                const int sumi = hsum_i32_128(_mm_add_epi32(p0, p1));
                const float ydf = use_pre ? yd_pre[t][b] : fp16_to_f32(yt.d);
                acc_rem[t] += dw * ydf * (float) sumi;
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = acc_rem[t];
    }
}

}  // namespace

void q5k_avx1_rows(int, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                   float* const* out, int r0, int r1) {
    const int nb = n / 256;
    for (int rb = r0; rb < r1; rb += 16) {
        const int re = rb + 16 < r1 ? rb + 16 : r1;
        int t = 0;
        for (; t + 4 <= nt; t += 4) {
            const BlockQ8K* y[4] = {(const BlockQ8K*) act[t + 0], (const BlockQ8K*) act[t + 1],
                                    (const BlockQ8K*) act[t + 2], (const BlockQ8K*) act[t + 3]};
            q5k_avx1_rows_t<4>(w, row_bytes, nb, y, out + t, rb, re);
        }
        const int rem = nt - t;
        if (rem == 3) {
            const BlockQ8K* y[3] = {(const BlockQ8K*) act[t + 0], (const BlockQ8K*) act[t + 1], (const BlockQ8K*) act[t + 2]};
            q5k_avx1_rows_t<3>(w, row_bytes, nb, y, out + t, rb, re);
        } else if (rem == 2) {
            const BlockQ8K* y[2] = {(const BlockQ8K*) act[t + 0], (const BlockQ8K*) act[t + 1]};
            q5k_avx1_rows_t<2>(w, row_bytes, nb, y, out + t, rb, re);
        } else if (rem == 1) {
            const BlockQ8K* y[1] = {(const BlockQ8K*) act[t + 0]};
            q5k_avx1_rows_t<1>(w, row_bytes, nb, y, out + t, rb, re);
        }
    }
}

void q51_avx1_rows(int, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                   float* const* out, int r0, int r1) {
    const int nb = n / 32;
    for (int rb = r0; rb < r1; rb += 16) {
        const int re = rb + 16 < r1 ? rb + 16 : r1;
        int t = 0;
        for (; t + 4 <= nt; t += 4) {
            const BlockQ81* y[4] = {(const BlockQ81*) act[t + 0], (const BlockQ81*) act[t + 1],
                                    (const BlockQ81*) act[t + 2], (const BlockQ81*) act[t + 3]};
            q51_avx1_rows_t<4>(w, row_bytes, nb, y, out + t, rb, re);
        }
        const int rem = nt - t;
        if (rem == 3) {
            const BlockQ81* y[3] = {(const BlockQ81*) act[t + 0], (const BlockQ81*) act[t + 1], (const BlockQ81*) act[t + 2]};
            q51_avx1_rows_t<3>(w, row_bytes, nb, y, out + t, rb, re);
        } else if (rem == 2) {
            const BlockQ81* y[2] = {(const BlockQ81*) act[t + 0], (const BlockQ81*) act[t + 1]};
            q51_avx1_rows_t<2>(w, row_bytes, nb, y, out + t, rb, re);
        } else if (rem == 1) {
            const BlockQ81* y[1] = {(const BlockQ81*) act[t + 0]};
            q51_avx1_rows_t<1>(w, row_bytes, nb, y, out + t, rb, re);
        }
    }
}

void q80_avx1_rows(int, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                   float* const* out, int r0, int r1) {
    const int nb = n / 32;
    for (int rb = r0; rb < r1; rb += 16) {
        const int re = rb + 16 < r1 ? rb + 16 : r1;
        int t = 0;
        for (; t + 4 <= nt; t += 4) {
            const BlockQ80* y[4] = {(const BlockQ80*) act[t + 0], (const BlockQ80*) act[t + 1],
                                    (const BlockQ80*) act[t + 2], (const BlockQ80*) act[t + 3]};
            q80_avx1_rows_t<4>(w, row_bytes, nb, y, out + t, rb, re);
        }
        const int rem = nt - t;
        if (rem == 3) {
            const BlockQ80* y[3] = {(const BlockQ80*) act[t + 0], (const BlockQ80*) act[t + 1], (const BlockQ80*) act[t + 2]};
            q80_avx1_rows_t<3>(w, row_bytes, nb, y, out + t, rb, re);
        } else if (rem == 2) {
            const BlockQ80* y[2] = {(const BlockQ80*) act[t + 0], (const BlockQ80*) act[t + 1]};
            q80_avx1_rows_t<2>(w, row_bytes, nb, y, out + t, rb, re);
        } else if (rem == 1) {
            const BlockQ80* y[1] = {(const BlockQ80*) act[t + 0]};
            q80_avx1_rows_t<1>(w, row_bytes, nb, y, out + t, rb, re);
        }
    }
}

}  // namespace strata::kernels::cpu
