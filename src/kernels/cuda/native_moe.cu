// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_moe.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__global__ void combine(const float* __restrict__ parts, const float* __restrict__ weights,
                        const float* __restrict__ shared, float* __restrict__ output,
                        int64_t n_embd, int k) {
    // blockIdx.y = the token of a multi-token launch (0 for the single one)
    const int64_t tk = blockIdx.y;
    parts += tk * k * n_embd; weights += tk * k; if (shared) shared += tk * n_embd; output += tk * n_embd;
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if ((n_embd & 3) == 0) {
        const int64_t n4 = n_embd >> 2;
        if (col >= n4) return;
        const float4* p4 = reinterpret_cast<const float4*>(parts);
        const float w0 = weights[0];
        const float4 a0 = p4[col];
        float4 sum = make_float4(a0.x * w0, a0.y * w0, a0.z * w0, a0.w * w0);
        for (int expert = 1; expert < k; ++expert) {
            const float we = weights[expert];
            const float4 ae = p4[int64_t(expert) * n4 + col];
            sum.x = fmaf(ae.x, we, sum.x); sum.y = fmaf(ae.y, we, sum.y);
            sum.z = fmaf(ae.z, we, sum.z); sum.w = fmaf(ae.w, we, sum.w);
        }
        if (shared) {
            const float4 sh = reinterpret_cast<const float4*>(shared)[col];
            sum.x += sh.x; sum.y += sh.y; sum.z += sh.z; sum.w += sh.w;
        }
        reinterpret_cast<float4*>(output)[col] = sum;
        return;
    }
    if (col >= n_embd) return;
    float sum = parts[col] * weights[0];
    for (int expert = 1; expert < k; ++expert) {
        sum += parts[int64_t(expert) * n_embd + col] * weights[expert];
    }
    if (shared) sum += shared[col];
    output[col] = sum;
}
bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
__global__ void combine_or_hit(const float* __restrict__ hit_out, const float* __restrict__ parts,
                               const float* __restrict__ weights, const float* __restrict__ shared,
                               float* __restrict__ output, const uint32_t* __restrict__ skip, uint32_t ring,
                               int64_t n_embd, int k) {
    const int64_t tk = blockIdx.y;
    const bool use_hit = (*skip == ring);
    const float* src = (use_hit ? hit_out : parts) + tk * k * n_embd;
    weights += tk * k;
    if (shared) shared += tk * n_embd;
    output += tk * n_embd;
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if ((n_embd & 3) == 0) {
        const int64_t n4 = n_embd >> 2;
        if (col >= n4) return;
        const float4* s4 = reinterpret_cast<const float4*>(src);
        const float w0 = weights[0];
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (w0 != 0.0f) {
            const float4 a0 = s4[col];
            sum = make_float4(a0.x * w0, a0.y * w0, a0.z * w0, a0.w * w0);
        }
        for (int expert = 1; expert < k; ++expert) {
            const float we = weights[expert];
            if (we != 0.0f) {
                const float4 ae = s4[int64_t(expert) * n4 + col];
                sum.x = fmaf(ae.x, we, sum.x); sum.y = fmaf(ae.y, we, sum.y);
                sum.z = fmaf(ae.z, we, sum.z); sum.w = fmaf(ae.w, we, sum.w);
            }
        }
        if (shared) {
            const float4 sh = reinterpret_cast<const float4*>(shared)[col];
            sum.x += sh.x; sum.y += sh.y; sum.z += sh.z; sum.w += sh.w;
        }
        reinterpret_cast<float4*>(output)[col] = sum;
        return;
    }
    if (col >= n_embd) return;
    const float w0 = weights[0];
    float sum = (w0 != 0.0f) ? (src[col] * w0) : 0.0f;
    for (int expert = 1; expert < k; ++expert) {
        const float we = weights[expert];
        if (we != 0.0f) sum += src[int64_t(expert) * n_embd + col] * we;
    }
    if (shared) sum += shared[col];
    output[col] = sum;
}
}
void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes)
            || (shared && !valid_span(shared, row_bytes))
            || overlap(output, row_bytes, parts, part_bytes)
            || overlap(output, row_bytes, weights, weight_bytes)
            || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    const unsigned gx = (n_embd & 3) == 0 ? unsigned(((n_embd >> 2) + 63) / 64) : unsigned((n_embd + 255) / 256);
    const unsigned bx = (n_embd & 3) == 0 ? 64u : 256u;
    combine<<<gx, bx, 0, static_cast<cudaStream_t>(stream)>>>(
        parts, weights, shared, output, n_embd, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    const unsigned gx = (n_embd & 3) == 0 ? unsigned(((n_embd >> 2) + 63) / 64) : unsigned((n_embd + 255) / 256);
    const unsigned bx = (n_embd & 3) == 0 ? 64u : 256u;
    combine<<<dim3(gx, (unsigned) n_tok), bx, 0, static_cast<cudaStream_t>(stream)>>>(
        parts, weights, shared, output, n_embd, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_multi_or_hit(const float* hit_out, const float* parts, const float* weights,
                                     const float* shared, float* output, const uint32_t* skip, uint32_t ring,
                                     int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (or_hit) requires a stream, width, 1..15 experts, tokens");
    const unsigned gx = (n_embd & 3) == 0 ? unsigned(((n_embd >> 2) + 63) / 64) : unsigned((n_embd + 255) / 256);
    const unsigned bx = (n_embd & 3) == 0 ? 64u : 256u;
    combine_or_hit<<<dim3(gx, (unsigned) n_tok), bx, 0, static_cast<cudaStream_t>(stream)>>>(
        hit_out, parts, weights, shared, output, skip, ring, n_embd, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
