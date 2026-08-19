# llama.cpp

Reference: `llama-server`, the OpenAI-compatible HTTP server shipped with llama.cpp.

## Version

| | |
|---|---|
| Commit | `af51726` (`ggml` 0.20.2) |
| Built with | CUDA 12.3.107, GCC 11.4.0, CMake 4.4.2 |
| CUDA arch | `75` only (Turing) |

## Building

```bash
bash scripts/build_llamacpp.sh
```

Two things about this machine make a source build mandatory rather than a
convenience.

**The published Linux binaries assume AVX2.** This CPU is an i5-2500K (Sandy
Bridge, 2011): it has AVX and SSE4.2 but no AVX2, no FMA, no F16C and no
BMI1/BMI2. A stock binary dies with SIGILL. The build therefore pins the ISA
explicitly:

```
-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF
-DGGML_F16C=OFF  -DGGML_BMI2=OFF
```

`GGML_BMI2` is the one that bites: it defaults to ON, BMI2 arrived with Haswell
in 2013, and nothing warns you at configure time. A correct configure prints

```
-- Adding CPU backend variant ggml-cpu: -msse4.2;-mavx GGML_SSE42;GGML_AVX
```

If `-mavx2`, `-mfma` or `-mbmi2` appear in that line, the binary will crash on
this CPU.

**Ubuntu's packaged nvcc is 11.5 and cannot build this.** Its C++ frontend
fails on GCC 11's libstdc++ with `error: parameter packs not expanded with
'...'` in `<functional>`. The build script points `CMAKE_CUDA_COMPILER` at the
CUDA 12.3 toolkit under `/usr/local/cuda` instead of whatever `nvcc` resolves to
on `PATH`.

## Running

Started automatically by `bench/run.py`; the equivalent manual command is:

```bash
vendor/llama.cpp/build/bin/llama-server \
  -m models/Qwen3-0.6B-f16.gguf \
  -ngl 99 --parallel 16 -c 8192 \
  --host 127.0.0.1 --port 8080 -t 4 --no-warmup --cache-reuse 0
```

`-c` is the **total** context shared by all slots and is divided by
`--parallel`. Passing the per-slot value here is the classic mistake: with
`--parallel 16` and `-c 4096`, every slot silently gets 256 tokens, the
long-prompt scenario no longer fits, and the comparison is wrong without
erroring. `configs/experiment.yaml` computes `ctx_total = parallel *
ctx_per_slot` for this reason.

`--cache-reuse 0` disables cross-request prompt-cache reuse. Prompts are
distinct per request anyway, but a cache hit would report a fake TTFT and it is
cheaper to forbid it than to detect it.

`-ngl 99` puts all 28 layers on the GPU. At F16 the weights are ~1.2 GB, so the
model fits entirely in the 1650 SUPER's 4 GB and never touches the PCIe 2.0 bus
during decode.

## Endpoint

`http://127.0.0.1:8080/v1/completions`, with
`stream_options: {"include_usage": true}` so a streamed response carries a usage
block at all.

## Two behaviours worth knowing before you trust a number

**Errors arrive inside the stream.** Once `llama-server` has sent HTTP 200 it
reports some failures as an SSE chunk:

```json
{"error":{"code":500,"message":"The model produced output that does not match the expected Content-only format","type":"server_error"}}
```

A client that only reads `choices[].text` sees a generation that stopped early
and no error at all — a plausible token count with a plausible TTFT. `ignore_eos`
triggers this reliably on Qwen3 (2 of 32 requests): forced past its natural stop,
the model emits control text the output parser rejects. Neither `--no-jinja` nor
`--reasoning-format none` prevents it. This benchmark runs without `ignore_eos`
and treats any stream that ends without a terminator as a failed measurement.

**It leaks host RAM per request.** Measured on this build (`af51726`), with the
model fully offloaded to the GPU:

| prompt | RSS growth per request |
|---|---|
| 64 tokens | ~78 MiB |
| 2560 tokens | ~295 MiB |

Roughly 70 MiB fixed plus ~70 KiB per prompt token, climbing linearly and never
returned. On a 5.5 GB box that is about sixteen long-prompt requests before the
kernel OOM-kills the server mid-measurement:

```
Out of memory: Killed process llama-server, anon-rss:4871812kB
```

Note that this is **host** RAM, not VRAM — VRAM peaked at 2.3 GB of the ~3.5 GB
available. Ruled out as causes: CUDA graph caching
(`GGML_CUDA_DISABLE_GRAPHS=1`), glibc arena fragmentation (`MALLOC_ARENA_MAX`),
and compute-buffer sizing (`-b`/`-ub`) — none change the slope. It also explains
throughput sagging run over run at concurrency 8 (120 → 92 → 72 tok/s): the box
was going to swap. The `longctx` profile therefore takes a fresh server per run.
