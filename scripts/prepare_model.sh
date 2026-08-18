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
[ -d vendor/llama.cpp ] || { echo "run scripts/build_llamacpp.sh first"; exit 1; }
"${PY}" -m pip install -q -r vendor/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
"${PY}" vendor/llama.cpp/convert_hf_to_gguf.py "${LOCAL_DIR}" \
    --outtype f16 --outfile "${GGUF}"

echo "==> 3/3 Ollama import: ${GGUF} -> ${OLLAMA_TAG}"
# NOT `ollama pull`: that would fetch Ollama's own Q4_K_M and break the
# comparison. Importing the GGUF we just built keeps the weights identical.
printf 'FROM ./%s\n' "${GGUF}" > Modelfile
"${ROOT}/vendor/ollama/bin/ollama" create "${OLLAMA_TAG}" -f Modelfile

echo
echo "GGUF sha256 (must match what llama-server and Ollama both load):"
sha256sum "${GGUF}"
echo
echo "Ollama's view of the model:"
"${ROOT}/vendor/ollama/bin/ollama" show "${OLLAMA_TAG}" || true
