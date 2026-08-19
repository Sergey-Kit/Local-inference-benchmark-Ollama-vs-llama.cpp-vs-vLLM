#!/usr/bin/env bash
# Produce the three artefacts the benchmark needs, all from ONE set of weights.
#
# This is the step that makes the comparison honest. vLLM eats safetensors;
# llama.cpp and Ollama eat GGUF. If each runtime fetched its own copy the
# comparison would silently be across different quantizations -- the exact
# failure mode the SPEC warns about, and the one Ollama walks into by default
# (`ollama pull` hands you its own Q4_K_M).
#
# So: download safetensors once, convert THOSE to GGUF F16, and import THAT
# file into Ollama. All three runtimes then execute bit-identical weights.
#
# F16 and not BF16: Qwen3 ships bfloat16, which Turing (sm75) has no hardware
# support for. FP16 is the one dtype all three runtimes agree on here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${ROOT}/.venv/bin/python"
CFG="${ROOT}/configs/experiment.yaml"

read -r HF_ID LOCAL_DIR GGUF OLLAMA_TAG <<<"$(
  "${PY}" -c "
import yaml
m = yaml.safe_load(open('${CFG}'))['model']
print(m['hf_id'], m['local_dir'], m['gguf'], m['ollama_tag'])
"
)"

echo "==> 1/3 safetensors: ${HF_ID} -> ${LOCAL_DIR}"
"${ROOT}/.venv/bin/hf" download "${HF_ID}" --local-dir "${LOCAL_DIR}"

echo "==> 2/3 GGUF F16: ${LOCAL_DIR} -> ${GGUF}"
[ -d vendor/llama.cpp ] || { echo "clone llama.cpp first (scripts/build_llamacpp.sh)"; exit 1; }
# Its own venv: the converter needs torch, and the measurement venv is kept
# light on purpose so it stays liftable into P2/P5.
[ -d .venv-convert ] || python3 -m venv .venv-convert
.venv-convert/bin/python -m pip install -q --upgrade pip
.venv-convert/bin/python -m pip install -q -r vendor/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
# scripts/verify_weights.py runs from this venv (it needs `gguf`) and reads the
# experiment config, which upstream's requirements do not pull in.
.venv-convert/bin/python -m pip install -q pyyaml
.venv-convert/bin/python vendor/llama.cpp/convert_hf_to_gguf.py "${LOCAL_DIR}" \
    --outtype f16 --outfile "${GGUF}"

echo "==> 3/3 Ollama import: ${GGUF} -> ${OLLAMA_TAG}"
# NOT `ollama pull`: that would fetch Ollama's own Q4_K_M and break the
# comparison. Importing the GGUF we just built keeps the weights identical.

# `ollama create` is a client command -- it needs a server, and it writes into
# whatever store that server was started with. Both have to match what
# bench/run.py will later read, or the import lands in ~/.ollama and the
# benchmark reports "model not found" for a model that was just built.
export OLLAMA_MODELS="${ROOT}/vendor/ollama/models"
mkdir -p "${OLLAMA_MODELS}"

if curl -s --noproxy '*' -o /dev/null http://127.0.0.1:11434/ 2>/dev/null; then
  echo "something is already listening on 11434 -- stop it first, so the import"
  echo "cannot land in a different model store than the one this script sets."
  exit 1
fi

IMPORT_LOG="$(mktemp)"
"${ROOT}/vendor/ollama/bin/ollama" serve > "${IMPORT_LOG}" 2>&1 &
OLLAMA_PID=$!
# --noproxy: this machine exports http_proxy for 127.0.0.1, and curl would
# otherwise ask the proxy about a server on loopback.
trap 'kill "${OLLAMA_PID}" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -s --noproxy '*' -o /dev/null http://127.0.0.1:11434/ 2>/dev/null && break
  sleep 1
done
if ! curl -s --noproxy '*' -o /dev/null http://127.0.0.1:11434/ 2>/dev/null; then
  echo "ollama serve did not come up; last lines:"; tail -20 "${IMPORT_LOG}"; exit 1
fi

printf 'FROM ./%s\n' "${GGUF}" > Modelfile
"${ROOT}/vendor/ollama/bin/ollama" create "${OLLAMA_TAG}" -f Modelfile

echo
echo "GGUF sha256 (must match what llama-server and Ollama both load):"
sha256sum "${GGUF}"
echo
echo "Ollama's view of the model:"
"${ROOT}/vendor/ollama/bin/ollama" show "${OLLAMA_TAG}" || true
