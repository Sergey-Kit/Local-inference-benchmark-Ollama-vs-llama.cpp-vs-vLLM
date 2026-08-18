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

## Status

<!-- filled in once the runtime has actually been exercised on this bench -->
