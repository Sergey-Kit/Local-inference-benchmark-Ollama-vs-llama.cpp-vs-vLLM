# Ollama

## Version

| | |
|---|---|
| Version | `v0.32.14` |
| Installed | `vendor/ollama/`, unpacked from the GitHub release, no root |

## Installing

```bash
bash scripts/install_ollama.sh
```

Deliberately **not** the upstream `install.sh`. That script wants sudo and
installs a systemd unit, and a systemd Ollama starts on boot and holds the GPU
— which is precisely what this bench must not have. `bench/run.py` refuses to
launch a runtime whose port is already listening, so a background Ollama would
block the whole run; worse, if it were sharing the card during a llama.cpp or
vLLM measurement it would corrupt the numbers silently.

Releases are now `.tar.zst` and this box has neither the `zstd` binary nor a
`tar` built against it, so the install script decompresses through the venv's
`zstandard` rather than asking for root.

## The model must be imported, not pulled

```bash
# WRONG for this benchmark:
ollama pull qwen3:0.6b        # gives you Ollama's own Q4_K_M

# What scripts/prepare_model.sh does:
printf 'FROM ./models/Qwen3-0.6B-f16.gguf\n' > Modelfile
vendor/ollama/bin/ollama create qwen3-06b-f16 -f Modelfile
```

This is the single most important step for a fair comparison. `ollama pull`
fetches Ollama's own quantization, so llama.cpp would be running F16 while
Ollama ran Q4_K_M and every number would be measuring quantization rather than
the runtime. Importing the GGUF that `scripts/prepare_model.sh` converted means
Ollama and llama.cpp execute a bit-identical file — verified by comparing
`sha256sum` of the GGUF against the path in `ollama show --modelfile`.

## Running

```bash
OLLAMA_NUM_PARALLEL=16 OLLAMA_CONTEXT_LENGTH=512 OLLAMA_KEEP_ALIVE=24h \
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_FLASH_ATTENTION=0 \
  vendor/ollama/bin/ollama serve
```

* `OLLAMA_KEEP_ALIVE=24h` — the default unloads the model after five minutes of
  idle, which would turn some measurement points into accidental cold starts.
* `OLLAMA_FLASH_ATTENTION=0` — Turing (sm75) has no FlashAttention-2; set
  explicitly so the configuration is stated rather than inferred.
* `OLLAMA_CONTEXT_LENGTH` is **per slot** here, unlike llama.cpp's `-c`.

## Endpoint: the native API, not the OpenAI one

`http://127.0.0.1:11434/api/generate` with `raw: true`.

This is the one place the benchmark deviates from "same endpoint everywhere",
and it is not a preference. Ollama's OpenAI-compatible `/v1/completions`
**re-applies the chat template to a prompt that already carries one**:

| endpoint | prompt_tokens | output begins |
|---|---|---|
| `/v1/completions` | **72** | `<think>\n…` |
| `/api/generate`, `raw: true` | **64** | the answer |

llama.cpp reports 64 for the same string. The extra wrapping also switches
Qwen3's thinking mode back on, so the model spends its budget emitting a
`<think>` block and runs into `max_tokens` -- 128 tokens in 141 of 408
requests, against llama.cpp's natural ~63. Measured that way, Ollama was doing
different work on a different prompt, and nothing in the response said so.
`raw: true` passes the string through untouched.

`bench/run.py` now preflights every runtime and aborts unless it reports the
prompt length that was sent, so this class of mismatch cannot recur silently.

## Do not set OLLAMA_FLASH_ATTENTION

It was set to `0` here on the reasoning that sm75 has no FlashAttention-2 and
the configuration should be explicit rather than inferred. Measured cost:

| | TTFT p50 |
|---|---|
| `OLLAMA_FLASH_ATTENTION=0` | **3074 ms** |
| unset | **570 ms** |

Everything else identical. Ollama's startup log prints `FLASH_ATTENTION:false`
in **both** cases, so the setting responsible for a 5.4x difference does not
appear in the configuration the server reports. Leaving it unset also matches
llama-server, which is likewise left on its default.

## Known deviation

Ollama does not support `ignore_eos`. The benchmark runs without it on every
runtime for that reason among others, so this is no longer an asymmetry --
see the README's Limitations.
