# vLLM

The hard case on this bench, and the reason the model is what it is.

## Why this hardware constrains vLLM more than the others

The GPU is a GTX 1650 SUPER: Turing, **compute capability 7.5**, 4 GB. vLLM
supports capability >= 7.0, so it runs — but a lot of its fast paths are gated
at >= 8.0:

| Feature | On sm75 | Consequence here |
|---|---|---|
| `bfloat16` | no hardware support | `--dtype float16` is mandatory; Qwen3 ships bf16 |
| FlashAttention-2 | requires sm80 | falls back to another backend; `VLLM_ATTENTION_BACKEND=XFORMERS` |
| AWQ / GPTQ / compressed-tensors kernels | require sm80 | fail with `Min capability: 80. Current capability: 75` |
| Marlin kernels | require sm80 | unavailable |

The third row is the load-bearing one. **On this card vLLM can only serve
unquantized weights**, which puts a hard ceiling on model size: at FP16, 4 GB of
VRAM minus the desktop's share minus a usable KV cache leaves room for well
under 2B parameters.

That constraint is what selected Qwen3-0.6B for the whole project — and it
turned out to be a gift. Since vLLM has to run FP16 anyway, llama.cpp and Ollama
can be given a GGUF **converted from those same safetensors at F16**, and
quantization stops being a variable of the experiment instead of being a caveat
in the README.

## Install

Separate venv: vLLM pins its own CUDA torch build (~10 GB) and would clobber the
harness venv, which is deliberately light enough to install anywhere.

```bash
python3 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install -U pip
.venv-vllm/bin/python -m pip install -r requirements-vllm.txt
```

## Running

```bash
.venv-vllm/bin/vllm serve models/Qwen3-0.6B \
  --dtype float16 --enforce-eager \
  --max-model-len 512 --max-num-seqs 16 \
  --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching --disable-log-requests --port 8000
```

* `--dtype float16` — no bfloat16 on sm75.
* `--enforce-eager` — CUDA graph capture buys 5–15% but costs several hundred MiB
  of VRAM and a lot of host RAM, and this box has ~3.5 GB of the former and
  5.5 GB of the latter. Disabled to get a working configuration first; measured
  separately afterwards, because "what eager mode costs you" is itself a result.
* `--gpu-memory-utilization 0.80` — vLLM sizes its pool as a fraction of the
  **whole** card, and the Windows desktop is already holding 13–20% of it.
* `--max-model-len` is per sequence, unlike llama.cpp's `-c`, which is the total
  across slots. Both come from the same `ctx_per_slot` in the config.

## Peak VRAM is not comparable as-is

vLLM reserves its KV pool up front. Its "peak VRAM" is therefore a
**reservation**, while llama.cpp's and Ollama's are **high-water marks**. The
figure says so on its face rather than only in prose, and the README repeats it
under Limitations. Comparing the two silently would be exactly the kind of
apples-to-oranges number the SPEC warns about.

## Debugging ladder

If the server fails to start, in this order:

1. `VLLM_ATTENTION_BACKEND=XFORMERS` — FlashAttention-2 needs sm80.
2. `VLLM_USE_V1=0` — fall back to the V0 engine if V1 refuses capability 7.5.
3. Lower `--max-model-len` and `--max-num-seqs`.
4. If it still will not run, that is a **result, not a failure**: record the
   exact version, flags and log, and write it up as vLLM's hardware floor. "This
   runtime does not run on this class of hardware" is a legitimate and useful
   on-prem finding.

## Status: runs, and the GPU was never the problem

The expectation going in was that Turing would block it. Compute capability 7.5
never came up. Three unrelated environment issues did, and each killed the
engine at startup rather than degrading anything:

1. **`RuntimeError: UVA is not available`.** vLLM disables pinned memory under
   WSL by default, and its engine then requires UVA, which requires pinned
   memory. Fixed with `VLLM_WSL2_ENABLE_PIN_MEMORY=1` -- vLLM's own opt-in for
   WSL2 kernels past 4.19.121 (ours is 6.18). Pinned allocation was verified
   working directly before enabling it.
2. **`TypeError: 'type' object is not subscriptable`** from flashinfer, which
   annotates with `array.array[int]` -- subscriptable only from Python 3.11.
   Ubuntu 22.04 ships 3.10 and 3.11 needs root. Uninstall flashinfer and set
   `VLLM_USE_FLASHINFER_SAMPLER=0`, which stops the sampler probing for it by
   import.
3. **`--disable-log-requests` was removed in 0.27** and makes `vllm serve` exit
   immediately with "unrecognized arguments".

It then starts in ~60 s on **TRITON_ATTN** with 1.9 GiB of KV cache -- 17,792
tokens, about 34 concurrent 512-token sequences.

### Measured

| concurrency | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| tok/s | 12.5 | 20.7 | 38.3 | 68.6 |
| TTFT p50, ms | 308 | 644 | 696 | 775 |
| TPOT p50, ms | 76.3 | 175.9 | 180.9 | 200.2 |

Scaling from 1 to 16 concurrent requests is **x5.48**, against llama.cpp's
x2.14 -- continuous batching doing exactly what it claims. It still loses in
absolute terms at every point measured, because 76 ms/token at concurrency 1 is
a very deep hole to climb out of, and that number is `--enforce-eager`: CUDA
graph capture is what removes vLLM's per-step Python overhead, and it needs
VRAM this card does not have. On a 0.6B model that overhead is most of the step.

The crossover implied by the two curves sits past concurrency 30, which this
card's KV budget could just about reach. That measurement is the obvious next
step and is listed in the README.
