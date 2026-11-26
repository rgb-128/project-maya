// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu}.
//
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

#include "strata/kernels/native_qsa.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}
template<int BlockSize>
__global__ void norm(const float* input, const float* __restrict__ gamma, float* output,
                     int n_cols, float epsilon) {
    const int tid = threadIdx.x;
    const std::size_t row_offset = std::size_t(blockIdx.x) * n_cols;
    input += row_offset; output += row_offset;
    float partial = 0.0f;
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
        const float x = input[col];
        partial += x * x;
    }
    __shared__ float sums[32];
    partial = warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    // All reads of input for the reduction precede this barrier. Afterwards,
    // each thread reads/writes only its own elements, allowing exact in-place use.
    __syncthreads();
    partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(partial);
    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize)
        output[col] = scale * input[col] * gamma[col];
}
template<int BlockSize>
__global__ void norm_strided(const float* __restrict__ input, int in_stride, const float* __restrict__ gamma,
                             float* __restrict__ output, int n_cols, float epsilon) {
    const int tid = threadIdx.x;
    input += std::size_t(blockIdx.x) * in_stride;
    output += std::size_t(blockIdx.x) * n_cols;
    float partial = 0.0f;
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
        const float x = input[col];
        partial += x * x;
    }
    __shared__ float sums[32];
    partial = warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(partial);
    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize)
        output[col] = scale * input[col] * gamma[col];
}
__global__ void gate(const float* attn, const float* __restrict__ q_full, float* output,
                     int n_head, int head_dim) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(n_head) * head_dim) return;
    const std::size_t head = i / head_dim, channel = i % head_dim;
    const float raw = q_full[head * 2 * head_dim + head_dim + channel];
    const float sigmoid = 1.0f / (1.0f + expf(-raw));
    output[i] = attn[i] * sigmoid;
}
struct QsaQ81Block {
    half2 ds;
    int8_t qs[32];
};
__global__ void gate_q8_1(const float* __restrict__ attn, const float* __restrict__ q_full,
                          QsaQ81Block* __restrict__ y_q8, int n_head, int head_dim) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(n_head) * head_dim) return;
    const std::size_t head = i / head_dim, channel = i % head_dim;
    const float raw = q_full[head * 2 * head_dim + head_dim + channel];
    const float xi = attn[i] * (1.0f / (1.0f + expf(-raw)));
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
        sum += __shfl_xor_sync(0xffffffffu, sum, offset, 32);
    }
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, roundf(xi / d))));
    const std::size_t ib = i >> 5;
    const int iqs = static_cast<int>(i & 31);
    y_q8[ib].qs[iqs] = q;
    if (iqs == 0) y_q8[ib].ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}
std::size_t elements(int cols, int rows) {
    if (cols <= 0 || rows <= 0 || std::uint64_t(cols) * rows > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native QSA requires positive bounded dimensions");
    return std::size_t(cols) * rows;
}
bool valid(const void* ptr, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    return ptr && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto ap = reinterpret_cast<std::uintptr_t>(a), bp = reinterpret_cast<std::uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void buffers(const float* input, std::size_t in_bytes, const float* weight, std::size_t weight_bytes,
             float* output, void* stream) {
    if (!stream || !valid(input, in_bytes) || !valid(weight, weight_bytes) || !valid(output, in_bytes) ||
        overlap(input, in_bytes, weight, weight_bytes) || overlap(output, in_bytes, weight, weight_bytes) ||
        (input != output && overlap(input, in_bytes, output, in_bytes)))
        throw std::invalid_argument("native QSA requires a stream, aligned spans, and disjoint buffers or exact input/output alias");
}
void check_launch() {
    const auto result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("native QSA launch: ") + cudaGetErrorString(result));
}
} // namespace

void native_qsa_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_cols, int n_rows, float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);
    if (n_cols < 1024)
        norm<256><<<unsigned(n_rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(input, gamma, output, n_cols, epsilon);
    else
        norm<1024><<<unsigned(n_rows), 1024, 0, static_cast<cudaStream_t>(stream)>>>(input, gamma, output, n_cols, epsilon);
    check_launch();
}
void native_qsa_rms_norm_weighted_strided(const float* input, int in_stride, const float* gamma, float* output,
                                          int n_cols, int n_rows, float epsilon, void* stream) {
    if (in_stride < n_cols)
        throw std::invalid_argument("native QSA strided requires in_stride >= n_cols");
    const auto out_count = elements(n_cols, n_rows);
    const auto in_count = elements(in_stride, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    if (!stream || !valid(input, in_count * 4) || !valid(gamma, std::size_t(n_cols) * 4) || !valid(output, out_count * 4) ||
        overlap(input, in_count * 4, output, out_count * 4))
        throw std::invalid_argument("native QSA strided requires valid disjoint spans");
    if (n_cols < 1024)
        norm_strided<256><<<unsigned(n_rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(
            input, in_stride, gamma, output, n_cols, epsilon);
    else
        norm_strided<1024><<<unsigned(n_rows), 1024, 0, static_cast<cudaStream_t>(stream)>>>(
            input, in_stride, gamma, output, n_cols, epsilon);
    check_launch();
}
void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream) {
    const auto count = elements(head_dim, n_head);
    buffers(attn, count * 4, q_full, count * 8, output, stream);
    gate<<<unsigned((count + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(attn, q_full, output, n_head, head_dim);
    check_launch();
}
void native_qsa_gate_apply_q8_1(const float* attn, const float* q_full, void* y_q8,
                                int n_head, int head_dim, void* stream) {
    const auto count = elements(head_dim, n_head);
    if (!stream || !y_q8 || (head_dim & 31) != 0 || !valid(attn, count * 4) || !valid(q_full, count * 8))
        throw std::invalid_argument("native QSA gate_apply_q8_1 requires valid stream, 32-aligned head_dim, and aligned spans");
    gate_q8_1<<<unsigned((count + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        attn, q_full, static_cast<QsaQ81Block*>(y_q8), n_head, head_dim);
    check_launch();
}
} // namespace strata::kernels
