"""Prove that all three runtimes execute the same weights.

This is the check the whole comparison rests on. `ollama create` rewrites the
GGUF it imports -- our file and Ollama's blob differ at byte 1205 -- so a plain
sha256 of the two files says "different" and tells you nothing about whether
the difference matters. What matters is the tensor payload, so that is what
gets hashed: every tensor, in name order, name and bytes.

Run it after scripts/prepare_model.sh, and quote the digest in the README.
Requires the converter venv (.venv-convert), which is where `gguf` lives.

    .venv-convert/bin/python scripts/verify_weights.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
from gguf import GGUFReader


def tensor_digest(path: Path) -> tuple[str, int]:
    reader = GGUFReader(str(path))
    digest = hashlib.sha256()
    for tensor in sorted(reader.tensors, key=lambda t: t.name):
        digest.update(tensor.name.encode())
        digest.update(bytes(memoryview(tensor.data).cast("B")))
    return digest.hexdigest(), len(reader.tensors)


def metadata_diff(a: Path, b: Path) -> list[str]:
    ra, rb = GGUFReader(str(a)), GGUFReader(str(b))
    keys = set(ra.fields) | set(rb.fields)
    return sorted(
        k for k in keys
        if str(ra.fields[k].contents() if k in ra.fields else None)
        != str(rb.fields[k].contents() if k in rb.fields else None)
    )


def ollama_blob(models_dir: Path, tag: str) -> Path | None:
    """Locate the GGUF blob Ollama would load for `tag`, from the manifest.

    Read off disk rather than through `ollama show`, which needs a running
    server -- and starting one here would mean a second runtime touching the
    GPU, which is the one thing the benchmark must never do.
    """
    name, _, version = tag.partition(":")
    version = version or "latest"
    manifests = models_dir / "manifests"
    candidates = list(manifests.rglob(f"{name}/{version}")) if manifests.exists() else []
    if not candidates:
        return None
    manifest = json.loads(candidates[0].read_text())
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer["digest"].replace(":", "-")
            return models_dir / "blobs" / digest
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--ollama-models", default="vendor/ollama/models")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ours = Path(cfg["model"]["gguf"])
    if not ours.exists():
        raise SystemExit(f"{ours} not found -- run scripts/prepare_model.sh first")

    our_digest, n_tensors = tensor_digest(ours)
    print(f"llama.cpp GGUF : {ours}")
    print(f"  file sha256  : {hashlib.sha256(ours.read_bytes()).hexdigest()}")
    print(f"  {n_tensors} tensors, payload sha256: {our_digest}")

    blob = ollama_blob(Path(args.ollama_models), cfg["model"]["ollama_tag"])
    if blob is None or not blob.exists():
        print("\nOllama blob not found (is the model imported?) -- skipping comparison")
        return 1

    their_digest, their_n = tensor_digest(blob)
    print(f"\nOllama blob    : {blob}")
    print(f"  file sha256  : {hashlib.sha256(blob.read_bytes()).hexdigest()}")
    print(f"  {their_n} tensors, payload sha256: {their_digest}")

    diff = metadata_diff(ours, blob)
    print(f"\nmetadata fields differing: {diff or 'none'}")

    if our_digest == their_digest and n_tensors == their_n:
        print("\nOK: llama.cpp and Ollama execute bit-identical tensor data.")
        print("    (vLLM loads the safetensors these were converted from, at the")
        print("     same FP16 precision -- see README, Limitations.)")
        return 0
    print("\nFAIL: tensor payloads differ. The runtime comparison is not valid.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
