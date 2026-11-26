# Pascal GPUs (experimental)

Tesla P100 (sm_60) and, by the reports in #189 and #212, Tesla P40 / GTX 10 series (sm_61). The setup offers them as
experimental (#212). This page covers the P100 kernels below; the kernel numbers in it were measured on v1.1.12, the
end-to-end ones at the end on v1.1.12, v1.1.44 and v1.1.58.

**What was checked.** The tree builds for sm_60 and sm_61 (CUDA 12.9). On a Tesla P100-PCIE-16GB the engine's test suite
runs: the kernel parity tests, the GLM layer and model tests against the oracle (`glm_model_test`), the multi-column GEMV
and the prompt MMQ test, and `pascal_moe_gemm_test` (the prompt products below against ggml's dequantizer). The
end-to-end rates are at the end. sm_61 was only compiled here; its run evidence is #189 and #212.

## Build

CUDA 12.x is required (CUDA 13 dropped Pascal). With CUDA 13 also installed, point CMake at the 12.x toolkit:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=60 -DSTRATA_NATIVE_EXPERTS=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-12.9
cmake --build build --target maya -j8
```

For sm_61 use `-DCMAKE_CUDA_ARCHITECTURES=61` (the build then adds `-Xptxas -dlcm=ca`, #214).

## What differs on Pascal

- **No `__dp4a` on sm_60** (it starts at sm_61): `include/strata/kernels/dp4a.hpp` gives it as four `vmad` (#261).
- **No `__nanosleep` below sm_70:** the host-flag waits spin instead.
- **No tensor cores:** the two wmma attention kernels compile to nothing below sm_70 and `mla_attn` takes the F32 kernel
  (the log says `prompt attention on the F32 cores`); the tensor-core prompt attention (`qsa_prompt_attn`) needs sm_80 and refuses.
- **Prompt GEMMs:** for FP16 x FP16 -> F32 on sm_60, cuBLAS's default algorithm is a legacy kernel. `ALGO6` is used from 1024
  tokens up (`include/strata/cuda_compat/pascal_gemm.hpp`): through `Gemm::f16` on a P100, 11.4 -> 9.4 ms for 2048 x 4096 x 4096,
  34.3 -> 28.6 ms for 2048 x 12288 x 4096 and 23.3 -> 19.1 ms for 8192 x 4096 x 2048 (about +20%), the same output bits. Below
  512 tokens ALGO6 is slower, so smaller products keep the default. bf16 and the batched calls keep it too (they refuse an
  explicit algo).
- **48 KB of shared memory a block:** the EXL3 several-activation kernels need 58 KB, so a verify window's tokens run one at
  a time on Pascal (the log says so). `exl3_test` therefore fails its window cases on sm_60.
- **sm_61 (P40, GTX 10):** global loads are cached in L2 only, which starves the small loads of the GEMVs; #214 adds
  `-Xptxas -dlcm=ca` for sm_6x (1.7-3x decode on P40s). The P100 (sm_60) caches them in L1 already: `glm_fast.cu` built
  for sm_60 with and without the flag gives the same SASS.

## Measured on a Tesla P100 (`glm_mv_bench`, 8192 x 4096, one column)

| type | size | time | bandwidth |
| --- | ---: | ---: | ---: |
| Q8_0 | 34.0 MB | 120 us | 283 GB/s |
| Q4_K | 18.0 MB | 86 us | 209 GB/s |
| Q6_K | 26.2 MB | 162 us | 162 GB/s |
| bf16 | 64.0 MB | 128 us | 500 GB/s |

The P100 has about 730 GB/s (a plain stream of the same bytes reads 550-590 GB/s here). The bf16 GEMV, which has no integer
dot, gets 500; the quantised ones sit at a third to a half of that.

**It is not the emulated `dp4a`.** A build with `__dp4a` replaced by a free add (wrong answers, timing only) runs the same
GEMV in the same time (Q4_K 88.2 against 84.5 us, Q6_K 154.8 against 156.4, Q8_0 121.9 against 119.7). What limits them is
the arithmetic around the dot and, on sm_60, the integer multiply itself: there is no 32-bit IMAD, the compiler emits XMADs
(16 x 16 -> 32 bits, about 3 for one 32-bit multiply). Maya's 16-byte-load pattern with the arithmetic removed reaches 490-560 GB/s, so
the loads are not the limit either.

`mv_q4k_pascal` (Q4_K, 1-4 columns, `MAYA_GLM_PASCAL_MV=0` turns it off) takes two weights and two activations through one
multiply (`(a0 << 16) + a2` against `w0 | w2 << 16`: the high half is `a0*w0 + a2*w2`; 8 of them stay under 2^15 in both halves,
so the result is the exact integer dot), stages the activations in shared memory in a bank-conflict-free layout, and unpacks the
eight sub-block scales without a branch. 8192 x 4096, one column: 84 -> 55 us (210 -> 330 GB/s); two columns 92.5 -> 80.5 us;
four columns are slower than the tiled kernels (190 against 165 us), and each column is bit for bit the one-column result.
The IQ2_XXS / IQ2_S / IQ3_XXS dots flip the signs of four weights with `(g ^ m) + (m & 1)` and `(x * 0x00204081) & 0x01010101`
instead of `__vcmpne4` + `__vsub4` (about 16 emulated ops each): the same bits, the routed gate/up kernel 113 -> 139 GB/s
(`moe_gate_up`, 6 experts), `moe_down` 106 -> 119 GB/s (IQ2_S).

#261 adds the Q6_K one-column GEMV (`MAYA_GLM_MV_P100`) and a sign table in shared memory for the routed experts' IQ3_XXS /
IQ3_S / IQ2_S kernels (in place of the dot above there; IQ2_XXS and the other kernels keep it). See docs/SETTINGS.md.
On the P100 the table is the faster of the two there (decode profile, Maya-S24: `moe_down` 25.4 against 28.0, `moe_gate_up`
47.3 against 48.0). `STRATA_GLM_GU_STAGE=0` makes `moe_gate_up` faster still (45.5); decode moves less than its noise.

## End to end on four Tesla P100 16 GB (PCIe 3 behind a PEX 8749 switch), Maya-S24, i5-14500, 62 GB DDR4

Layer split 13/11/11/10, 32K context, MTP drafts (82-93% accepted), `--bench`-style 256-token answers:

| | tokens/s |
| --- | ---: |
| decode (three questions) | 21.7 / 22.1 / 23.5 |
| prompt, 2K tokens, `--prefill auto` (one 8192 chunk, the cards run in turn) | 105 |
| prompt, 2K / 8K tokens, `--prefill 512` | 133 / 220 |
| prompt, 2K / 8K tokens, `--prefill 1024` | 131 / 257 |
| prompt, 4K tokens: `--prefill 1024` / 1536 / 2048 | 175 / 173 / 160 |
| prompt, 4K tokens, `--prefill 1024` with `STRATA_GLM_PREFILL_CPU=0` / `STRATA_GLM_PRESTAGE=96` / `STRATA_GLM_PREFILL_SUB=512` | 147 / 169 / 175 |

Decode is bound by the host: the RAM tier's experts (about a fifth of the routes, 2-3 a layer) go through the CPU lane (0.32 ms an
expert, the host's 26 GB/s DDR read bandwidth) or over PCIe (1.1 ms), and the GPU waits for them (`moe_cpu_wait` is the largest
single category after the idle time of the layer split). So GEMV speed buys little: the Q4_K kernel above, 1.5x faster in isolation
(`kda_proj_mv` -34% in a profiled run), gave 22.31 -> 23.52 tokens/s (+5.4%) over three interleaved A/B rounds (all nine paired
questions faster; one unpaired A/B pair differed by less than the +-5% run-to-run noise, so compare interleaved runs). CPU-lane
threads (8 or 11 equal, 14 slower), `STRATA_GLM_PREFETCH_N` and `RAM_RESIDENT` (stops with "454 of 544 slots": the tier is 33 GB of
the 37.3 needed) did not help. More free RAM and more memory bandwidth are what would. Prompts are bound the same way: a 1024-token chunk stages or computes about 100 non-resident experts a layer, the CPU
lane's multi-token dots run at roughly 0.26 TFLOP/s, and the three idle cards wait for the working one.

**On v1.1.44** (the same four cards, 32K context, 256-token greedy answers to three questions, then a 4096-token prompt; three
interleaved rounds against the v1.1.12 build): decode 24.8 -> 28.2 tokens/s (all nine paired answers faster, drafts accepted
the same), prompt 203 -> 209. With `MAYA_GLM_PASCAL_MV=0` decode is 6% slower (26.2 -> 24.6, `kda_proj_mv` +50%); with
`MAYA_GLM_PREFILL_PASCAL=0` (MMQ) the prompt is 8% slower (189 -> 174). This prompt (docs text) sends 600-900 experts a
1024-token chunk to each card's CPU lane, so it reads slower than the 4K figure above.

**On v1.1.58** (`main` at a358252 against the same commit with this page's kernels; the same four cards behind one PCIe 3.0 x4
uplink, 26 GB RAM tier, 131072 context, `--parallel 4`, `--prefill auto`, no other settings; three interleaved rounds, temperature 0.7,
thinking off): one conversation's decode 23.5 -> 26.5 tokens/s (nine 400-token answers each, +13%), with 5K tokens of context
20.7 -> 22.2; a 2.9K-token prompt's first token after 31.5 -> 20.7 s (93 -> 143 tokens/s), a 6.2K-token one 57.3 -> 43.1 s
(109 -> 146). Four conversations at once decode the same (37.0 against 37.4 tokens/s in total, one run a round).

## GP100 / Pascal Silicon Breakthroughs (Qwen3.8-Flash-Next, GLM-5.3, DeepSeek-V4.1)

The Pascal port extends beyond the original GLM-5.3 kernels to full **GP100 (`sm_60`) and Pascal (`sm_61`) acceleration** across `maya` (Qwen3.8-Flash-Next + GLM-5.3) and `maya-ds4` (DeepSeek-V4.1), with **zero accuracy/intelligence loss** (100% exact top-K routed experts, Vision ON, 64K context):

### Core Silicon-Level Techniques

1. **K-Major Structure-of-Arrays (SoA) Expert Repacking (`src/kernels/cuda/iq_kernels.cu`, `src/core/weights.cpp`, `src/core/expert_cache.cpp`)**
   - Standard GGUF super-blocks (`IQ2_S` 82 B, `IQ2_XXS` 66 B, `Q2_0`, `IQ4_NL`, `Q5_0`) are 2-byte aligned and force uncoalesced/strided byte-level loads across warp lanes.
   - Repacking resident VRAM experts on load into K-Major SoA (`qs`, `signs`, `qh`, `scales`, `d` in separate contiguous 16-byte-aligned streams) turns scattered reads into 100% coalesced `LDG.128` (`uint4`) bursts.
   - Generalized via K-tiling for **any** `n_embd % 256 == 0` and `n_ff % 64 == 0` (`% 32 == 0` for `Q5_0`/`IQ4_NL`), removing Qwen3.8 dimension hardcoding so any MoE model shape using these quant types automatically accelerates.

2. **32-Bit Bank-Split SoA Shared-Memory Codebooks (`iq_kernels.cu`, `glm_fast.cu`, `ds4_iq.cu`)**
   - GP100 has a small 24 KB L1 TEX cache per SM and 32-bit shared-memory banks (64-bit `LDS.64` hits 2-way bank conflicts).
   - Splitting the 64-bit IQ2 grid tables (`iq2xxs_grid` 2 KB, `iq2xs_grid` 4 KB, `iq2s_grid` 8 KB) into separate `uint32_t sg_lo[]` and `uint32_t sg_hi[]` arrays in shared memory replaces L1 TEX cache thrashing with conflict-free 1-cycle `LDS.32` lookups.
   - Ported across **all three engines**: Qwen3.8 (`iq_kernels.cu`), GLM-5.3 (`glm_fast.cu`), and DeepSeek-V4.1 (`ds4_iq.cu`).

3. **Native FP16 `PRMT + HSUB2 + HFMA2` Arithmetic (`iq_kernels.cu`, `native_mmvq.cu`, `shared_expert.cu`, `fused_gr.cu`)**
   - Tesla P100 (`sm_60`) lacks hardware `__dp4a` and Tensor Cores, **but** executes native `HFMA2` (`__hfma2`) at **2× FP32 rate (18.7 TFLOPS peak FP16)**.
   - Converting unsigned grid bytes (`0..255`) straight into exact FP16 pairs via `prmt.b32(..., 0x64646464)` (`1024 + g`) → `__hsub2(..., 1024)` → XOR sign bit → `__hfma2` avoids slow emulated `dp4a` (`vmad`) entirely and unlocks GP100's full dual-FP16 pipelines.

4. **TM=8 + `LDS.128` (`uint4`) Prefill GEMM Tiling (`iq_kernels.cu`)**
   - Upgraded `p100_prefill_iq2s_gu_soa_kernel` and `p100_prefill_iq2xxs_gu_soa_kernel` to `TM = 8` register tiling with `LDS.128` (`uint4`) shared-memory broadcasts and a fast path for `nt <= 4` tokens (`+23.6%` micro-benchmark speedup, `0.000000e+00` max diff).
   - Added native single-pass `NT = 6, 7` dispatch across `native_mmvq.cu`, `shared_expert.cu`, and `native_bf16.cu` to eliminate double weight reads on read/suffix-draft windows.

5. **Parallel Stage-Scoped GDN & QSA Checkpoints (`src/program/generate.cpp`)**
   - Eliminated a ~700 ms/turn stall in multi-turn agentic workloads by scoping `checkpoint_save` and `checkpoint_restore` to each GPU's own `[lb, le)` layer range and executing all 4 GPU PCIe transfers in parallel (`4×` less host RAM and PCIe traffic per turn).

6. **DeepSeek-V4.1 (`maya-ds4`) Full `sm_60` Compilation (`src/ds4/ds4_iq.cu`, `src/ds4/ds4_gguf.cu`)**
   - Added exact `sm_60` `ds4_dp4a` / `ds4_gguf_dp4a` int8 fallbacks, guarded `nvcuda::wmma` Tensor-Core-only kernels cleanly with `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700` (`__trap()`), and ported 32-bit Bank-Split SoA codebooks (`IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ1_M`).
   - Both `maya` and `maya-ds4` now compile and link cleanly on `sm_60` (`ninja maya maya-ds4`).

### Verified End-to-End Benchmarks — 4× Tesla P100-PCIE-16GB (Qwen3.8-Flash-Next UD-IQ2_XS)

System: 4× Tesla P100-PCIE-16GB (`sm_60`, PCIe Gen3, clocks locked 1328/715 MHz, `MAYA_GLM_P2P=0`), Dual Xeon (Ivy Bridge, no AVX2), Vision ON (`mmproj-F16.gguf`, `3,352 MiB` free VRAM on GPU 0), **100% exact top-10 routed experts (zero TOPK truncation)**, 64K context (`--max-context 65536`, `--kv int8`, `--spec 5`, `--spec-min-p 0.70`, `--short-read 128`):

| Workload | Baseline (Nov 16, exact) | Optimized (Increment 13, exact) | Gain |
| --- | ---: | ---: | ---: |
| **Short Fresh Prompt TTFT (~40–60 tok)** | `1.26 s` | **`0.443 – 0.494 s`** | **2.8× faster** |
| **Decode Throughput (256-tok E2E, Vision ON)** | `29.7 tok/s` | **`66.8 – 71.5 tok/s`** *(up to ~80 server)* | **2.3 – 2.4× faster** |
| **15.6K Unique Telemetry Prefill + Needle Recall** | `~187 tok/s` | **`609.8 tok/s` (`25.65 s`) + `70.1 tok/s` decode** | **3.26× faster** *(100% exact needle)* |
| **17.6K Cold Codebase Prefill** | `~187 tok/s` | **`598.9 tok/s` (`29.33 s`) + `69.0 tok/s` decode** | **3.20× faster** |
| **31.2K Context Agentic Follow-Up (+75 tok)** | `1.68 s` *(at 15K)* | **`0.82 s` TTFT (`96.4 tok/s`) + `71.1 tok/s` decode** | **>2× faster at 2× context** |
| **31.5K Context Reasoning Turn (`effort=medium`)** | — | **`0.56 s` TTFT (`87.6 tok/s`) + `57.7 tok/s` decode** | Exact verdict |

### Cross-GPU & Engine Compatibility Matrix

| Target / Engine | `sm_60` (Tesla P100) | `sm_61` (Tesla P40 / GTX 1080 Ti) | `sm_70+` (V100 / RTX 20–50) | Notes |
| --- | :---: | :---: | :---: | --- |
| **`maya` (Qwen3.8-Flash-Next)** | Full (HFMA2 + SoA) | Supported (`-dlcm=ca` + `__dp4a`) | Native / unchanged | Generalized K-tiled SoA kernels |
| **`maya` (GLM-5.3 / Maya-S24)** | Full (32-bit SoA LUT + P100 MV) | Supported (`-dlcm=ca` + `__dp4a`) | Native / unchanged | Exact bit-for-bit |
| **`maya-ds4` (DeepSeek-V4.1)** | Builds & links cleanly | Supported (`__dp4a` + SoA LUT) | Native / unchanged | WMMA guarded `< sm_70` |
| **`maya-mm3` (MiniMax-M3.1)** | Skipped by CMake (`< sm_70`) | Skipped by CMake (`< sm_70`) | Supported | `mm3_exl3.cu` exceeds Pascal 4 KB param limit |

## Tests that do not pass on sm_60

- `exl3_test`: the several-activation window cases (48 KB of shared memory, above).
- `kv_hybrid_parity`: its last case expects the sm_80 tensor-core prompt attention to take the pools; below sm_80 it refuses
  by design. The other four cases of the test pass.
- `ple_parity`, `glm_model_big_*`: need a model file and numpy; unrelated to the GPU.
- `cuda_device_selftest`: `device.cu` still asks for compute capability 7.5; only this diagnostic calls it.
- `maya-mm3` (MiniMax-M3.1) is skipped below `sm_70` in `CMakeLists.txt` because `mm3_exl3.cu` exceeds Pascal's 4 KB CUDA kernel parameter limit; `maya` and `maya-ds4` both build cleanly for `sm_60`.
