# P1 — local inference benchmark

Compare one model across three runtimes (Ollama, llama.cpp, vLLM) under
identical conditions, and report TTFT / TPOT / throughput / peak VRAM / cold
start. Original brief: `docs/SPEC.md`.

## The one rule

**Anything that differs between runtimes other than the runtime itself is a
bug in the experiment.** Same weights, same prompts, same sampling, same
context, same measuring code. Differences that cannot be removed go in the
README's "Limitations" section explicitly — never silently.

## Bench

WSL2 (Ubuntu 22.04), GTX 1650 SUPER 4 GB (Turing, **compute capability 7.5**),
Intel i5-2500K (4 cores, **AVX only — no AVX2/FMA/F16C/BMI2**), 5.5 GB RAM.

Consequences that shape almost every decision in this repo:

* **No bfloat16 and no FlashAttention-2 on sm75**, and vLLM's quantized kernels
  require capability >= 8.0. So: Qwen3-0.6B in **FP16, unquantized, everywhere**.
  Side effect worth having — quantization stops being a variable at all.
* **KV cache costs 112 KiB/token** for this model (2 x 28 layers x 8 kv_heads x
  128 head_dim x 2 bytes). With ~3.5 GB usable VRAM the concurrency sweep and
  the long-prompt scenario cannot share one server configuration, hence the two
  `server_profiles` in `configs/experiment.yaml`.
* **Stock llama.cpp binaries SIGILL here** (AVX2). Always build via
  `scripts/build_llamacpp.sh`.
* **Ubuntu's nvcc 11.5 cannot compile llama.cpp** with GCC 11; the build uses
  the CUDA 12.3 toolkit under `/usr/local/cuda`.
* **WSL2 does not report per-process GPU memory.** Peak VRAM is device-wide
  `used` minus a baseline captured before each server launch. The baseline is
  the Windows desktop's share and it drifts — it was observed moving from 544 to
  860 MiB within a single session — so it is never reused across servers.

## Layout

```
configs/experiment.yaml   single source of truth; nothing is hardcoded in bench/
prompts/build_prompts.py  generates prompts.jsonl with exact token counts
bench/client.py           the one measuring client, shared by all runtimes
bench/metrics.py          pure functions (TTFT/TPOT/percentiles) -- unit tested
bench/monitor.py          NVML sampler for VRAM/RSS
bench/run.py              orchestrator for the runtime x concurrency matrix
scripts/                  build llama.cpp, install ollama, prepare the model
analysis/plots.py         figures for the README
```

## Conventions

* `configs/experiment.yaml` is the only place model paths, URLs and load
  parameters live. `bench/` reads them; it never hardcodes them. This repo's
  client/metrics/monitor are meant to be lifted into P2 and P5 unchanged.
* Measurement code goes in `bench/`; anything runtime-specific goes in the
  config or in `runtimes/*.md`, never in the client.
* A measurement that cannot be trusted must fail loudly. `metrics.py` raises
  rather than returning a plausible-looking number; the client records a
  tokenizer/usage disagreement instead of reconciling it.
* Run `.venv/bin/python -m pytest` after touching `bench/`.

## Commands

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt

bash scripts/build_llamacpp.sh          # source build, ISA pinned for this CPU
bash scripts/install_ollama.sh          # local install, no root, no systemd
bash scripts/prepare_model.sh           # safetensors -> GGUF F16 -> ollama import
.venv/bin/python prompts/build_prompts.py

.venv/bin/python -m pytest
.venv/bin/python -m bench.run --dry-run
.venv/bin/python -m bench.run --runtime llamacpp
.venv/bin/python -m analysis.plots
```

Only one runtime may hold the GPU at a time; `bench/run.py` enforces this by
refusing to start when the port is already listening.
