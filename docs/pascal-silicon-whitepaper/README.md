# Project Maya Pascal (`sm_60` / `sm_61`) Notes & Benchmarks

Runs large MoE models on 1 to 4 NVIDIA Pascal GPUs on Linux x86_64, single-CPU or multi-socket. Exact top-K routing is kept on all runs.

For 4B-40B models like Qwen3.8-27B, Qwen3.6-35B, Gemma-4, LFM2.5, Nemotron-3.5-Bolt, Flint, and Inkling-2, see [rgb-128/llama.cpp-p100](https://github.com/rgb-128/llama.cpp-p100).

## Install

Requires NVIDIA driver 580.x and CUDA Toolkit 12.x.

```bash
git clone -b p100-pascal-952-port https://github.com/rgb-128/project-maya.git
cd project-maya
./install-pascal.sh --install-deps
./scripts/run-maya-pascal.sh -- --setup
```

`install-pascal.sh` detects `sm_60` or `sm_61`, CUDA 12.x, and CPU core count. `run-maya-pascal.sh` wraps `./maya.sh` with Pascal flags and adjusts NUMA settings for 1-socket or multi-socket systems automatically.

## Hardware support

| GPUs | Recommended mode | CPU / RAM |
| :--- | :--- | :--- |
| 1x Pascal | Single GPU, experts overflow to RAM | Local RAM, AVX1/AVX2/AVX-512 pool |
| 2x Pascal | `--tp 2` or `--layer-split auto` | Local RAM on 1 socket; interleave or mirror on 2+ sockets |
| 3x Pascal | `--layer-split auto` | Same |
| 4x Pascal | `--tp 4` or `--layer-split auto` | Mirror on 2+ sockets with >= 280 GiB RAM, else interleave |

P2P BAR transfers are tested at startup. Boards without P2P fall back to mapped host RAM automatically.

## Benchmarks

Measured on 4x Tesla P100-PCIE-16GB over PCIe Gen3 x16 with dual-socket AVX1/DDR3 CPU offloading. All runs use exact top-K routing and Vision enabled where supported. All speeds are tokens per second (`tps`).

| Model | Size | Mode | Prompt / Max Ctx | Prefill (tps) | Decode (tps) |
| :--- | ---: | :--- | :--- | ---: | ---: |
| Qwen3.8-Flash-Next UD-Q4_K_XL | 105.2 GiB | 4x P100 TP=4, Vision | 57K / 256K | 1276.3 | 76.7 |
| Qwen3.8-Flash-Next UD-Q4_K_XL | 105.2 GiB | 4x P100 TP=4, Vision | 31K / 256K | 1207.8 | 75.9-99.6 |
| Qwen3.8-Flash-Next UD-Q4_K_XL | 105.2 GiB | 4x P100 TP=4, greedy | Short / 256K | 1207.8 | 90.4 |
| Qwen3.8-Flash-Next UD-Q4_K_XL | 105.2 GiB | 4x P100 layer-split | 32K / 64K | 1140.7 | 105.1 |
| Qwen3.8-Flash-Next UD-IQ2_XS | 52 GiB | 4x P100, Vision | 15.6K / 64K | 609.8 | 101.3 |
| GLM-5.3-Flash Q4_K | 169 GB | 4x P100, exact top-8 | 6K / 16K | 155.0 | 17.3-20.9 |

Baselines before these Pascal optimizations on the same 4x P100 setup: `Qwen3.8-Flash-Next` UD-Q4_K_XL ran at 345-632 tps prefill and 34.7 tps warm decode; `Qwen3.8-Flash-Next` UD-IQ2_XS ran at 187 tps prefill and 29.7 tps decode; `GLM-5.3-Flash` Q4_K ran at 1.7 tps prefill and 2.0 tps decode in `llama.cpp` or 5.6 tps decode in stock Maya.

## Measured hardware ceilings

Measured on the same 4x P100 PCIe Gen3 x16 setup with `iommu=pt`, `ACS=0`, `MRRS=4096B`, and `ECC=ON`:

| Link | Measured |
| :--- | ---: |
| HBM2 read per P100, ECC ON | 568-570 GB/s |
| Dual-socket DDR3-1600 | 70.4 GB/s |
| Cross-socket QPI | 22.4 GB/s |
| PCIe Gen3 x16 DMA per card | 12.9 GB/s D2H, 11.9 GB/s H2D |
| 4-GPU concurrent H2D | 46.4 GB/s |
| Intra-socket BAR P2P | 10.3 GB/s DMA, 10.2 GB/s SM read |
| Mapped host RAM, 20 KB | 4.1 us write, 6.0 us read |

## Key changes in this branch

- **GP100 FP16x2 and SoA kernels:** `Q4_K`, `Q5_1`, `Q8_0`, `Q6_K`, `IQ4_NL`, `IQ2_XS`, and `BF16` router kernels written for `HFMA2`, offline bit-transposed `Q5_1`/`Q8_0` SoA layouts, `LDS.U.128` SMEM loads, and `STACK:0`.
- **Collectives:** Single-TX mapped-RAM DDIO AllReduce for small decode buffers and BAR-P2P + NUMA exchange for prefill.
- **Attention and GDN:** Chunkwise WY FP32 GDN prefill (`1.8x` faster), 64x64 tiled QSA scoring (`5.9x` faster at 57K), zero-spill Top-K up to 256K context, and `float4` prefill `moe_combine` (`3.9x` faster).
- **Host CPU:** 4-row tiled AVX1+F16C+SSSE3 kernels (`90-94 GB/s` effective), lock-free worker pool, and optional mirrored NUMA arenas (`MAYA_ARENA_NUMA=mirror`) on multi-socket machines.

## Usage examples

Run interactive setup or a saved config:

```bash
./scripts/run-maya-pascal.sh -- --setup
./scripts/run-maya-pascal.sh -- --config /path/to/maya-config.json --port 8080
./scripts/run-maya-pascal.sh --gpus 2 -- --config /path/to/maya-config.json
```

For `Qwen3.8-Flash-Next` on 2 or 4 GPUs, add `--tp 2` or `--tp 4` and `--kv-resident 32768` inside the config's `args` array alongside `--native`, `--ple-gguf`, and `--expert-profile ./data/expert-profile-qwen.bin`. On 1 GPU omit `--tp`; on 3 GPUs use `--layer-split auto`.
