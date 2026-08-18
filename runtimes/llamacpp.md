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

`http://127.0.0.1:8080/v1/completions` — supports `stream_options.include_usage`
and `ignore_eos`.
