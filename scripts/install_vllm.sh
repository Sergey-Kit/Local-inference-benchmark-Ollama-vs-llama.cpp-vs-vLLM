#!/usr/bin/env bash
# Install vLLM into its own venv, with the two workarounds this box needs.
#
# Separate venv because vLLM pins its own CUDA torch build (~10 GB) and would
# clobber the measurement venv, which is deliberately light enough to install
# anywhere.
#
# The flashinfer removal is not optional here. vLLM pulls it in via
# requirements/cuda.txt, but flashinfer 0.6 does not import on Python 3.10 --
# it annotates a function with `array.array[int]`, subscriptable only from 3.11
# -- and vLLM's sampler probes for flashinfer *by importing it*, so the engine
# dies at startup with "TypeError: 'type' object is not subscriptable". Ubuntu
# 22.04 ships Python 3.10 and installing 3.11 needs root, so the package goes.
# VLLM_USE_FLASHINFER_SAMPLER=0 (set in configs/experiment.yaml) then stops the
# probe from importing it at all.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
VENV="${ROOT}/.venv-vllm"

[ -d "${VENV}" ] || python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -q --upgrade pip
"${VENV}/bin/python" -m pip install -r requirements-vllm.txt

echo "==> removing flashinfer (does not import on $(python3 -V 2>&1 | cut -d' ' -f2))"
"${VENV}/bin/python" -m pip uninstall -y -q flashinfer-python 2>/dev/null || true

echo
"${VENV}/bin/python" - <<'PY'
import importlib.metadata as m
print("vllm ", m.version("vllm"))
print("torch", m.version("torch"))
try:
    import flashinfer  # noqa: F401
    print("WARNING: flashinfer still importable; vllm serve will fail on py3.10")
except ImportError:
    print("flashinfer: absent, as required")
PY
echo
echo "vLLM also needs VLLM_WSL2_ENABLE_PIN_MEMORY=1 under WSL2 -- it disables"
echo "pinned memory there by default and then requires UVA, which needs it."
echo "Both variables are set for you in configs/experiment.yaml."
