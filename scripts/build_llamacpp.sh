#!/usr/bin/env bash
# Build llama.cpp for this bench.
#
# The published llama.cpp Linux binaries are compiled with AVX2. This machine is
# an i5-2500K (Sandy Bridge, 2011): it has AVX and SSE4.2, but no AVX2, no FMA,
# no F16C and no BMI1/BMI2, so a stock binary dies with SIGILL. Hence a source
# build with the ISA extensions explicitly pinned to what the CPU actually has.
#
# GGML_BMI2 in particular defaults to ON and is easy to miss: BMI2 arrived with
# Haswell (2013), two generations after this CPU. Verified against /proc/cpuinfo.
#
# GGML_NATIVE=OFF stops ggml from probing the build host, which makes the
# resulting binary reproducible and the flags below authoritative.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/vendor/llama.cpp"
CMAKE="${ROOT}/.venv/bin/cmake"   # pip-installed, so no root needed
JOBS="${JOBS:-$(nproc)}"

# Ubuntu's packaged nvcc here is 11.5, whose C++ frontend cannot parse GCC 11's
# libstdc++ (<functional> fails with "parameter packs not expanded"). A newer
# toolkit is already installed under /usr/local; use it explicitly rather than
# whatever `nvcc` PATH happens to resolve to.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${CUDA_HOME}/bin/nvcc"
[ -x "${NVCC}" ] || { echo "no nvcc at ${NVCC}; set CUDA_HOME"; exit 1; }
echo "using $("${NVCC}" --version | tail -2 | head -1)"

[ -d "${SRC}" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "${SRC}"

"${CMAKE}" -B "${SRC}/build" -S "${SRC}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="${NVCC}" \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=ON \
    -DGGML_AVX2=OFF \
    -DGGML_FMA=OFF \
    -DGGML_F16C=OFF \
    -DGGML_BMI2=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF

"${CMAKE}" --build "${SRC}/build" --config Release -j "${JOBS}" \
    --target llama-server llama-cli llama-bench llama-quantize

echo
echo "built:"
ls -la "${SRC}/build/bin/" | grep -E "llama-(server|cli|bench|quantize)"
echo
echo "llama.cpp commit: $(git -C "${SRC}" rev-parse --short HEAD)"
