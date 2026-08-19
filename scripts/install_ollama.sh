#!/usr/bin/env bash
# Install Ollama into vendor/ollama, without root.
#
# The upstream install.sh wants sudo and installs a systemd unit. A systemd
# Ollama would start on boot and hold the GPU, which is exactly what this bench
# must not have: bench/run.py refuses to start a runtime when its port is
# already taken, precisely so two runtimes can never share the card. A local,
# unmanaged install leaves the GPU idle unless the benchmark asks for it, and
# pins the version in-tree so the results stay attributable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/vendor/ollama"
PY="${ROOT}/.venv/bin/python"
VERSION="${OLLAMA_VERSION:-v0.32.14}"
URL="https://github.com/ollama/ollama/releases/download/${VERSION}/ollama-linux-amd64.tar.zst"

mkdir -p "${DEST}"
echo "==> downloading ollama ${VERSION}"
curl -fL --progress-bar "${URL}" -o "${DEST}/ollama.tar.zst"

# Releases are zstd-compressed and this box has neither the zstd binary nor a
# tar built against it; decompress through the venv instead of asking for root.
echo "==> decompressing"
"${PY}" - "${DEST}/ollama.tar.zst" "${DEST}/ollama.tar" <<'PY'
import sys
import zstandard
with open(sys.argv[1], "rb") as src, open(sys.argv[2], "wb") as dst:
    zstandard.ZstdDecompressor().copy_stream(src, dst)
PY
tar -xf "${DEST}/ollama.tar" -C "${DEST}"
rm -f "${DEST}/ollama.tar.zst" "${DEST}/ollama.tar"

echo
echo "installed: $("${DEST}/bin/ollama" --version 2>&1 | tail -1)"
echo "for interactive use:  export PATH=\"${DEST}/bin:\$PATH\""
