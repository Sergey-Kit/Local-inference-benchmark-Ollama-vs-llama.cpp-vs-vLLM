"""Draw docs/architecture.png -- the bench, and where the weights come from.

The diagram exists to make one thing obvious at a glance: all three runtimes
execute the same weights, and one client measures all three. Everything else on
the page is supporting detail.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SURFACE, INK, INK2, MUTED, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#7a7973", "#c9c8c3"
SERIES = {"llamacpp": "#2a78d6", "ollama": "#eb6834", "vllm": "#1baf7a"}


def box(ax, x, y, w, h, title, lines, colour=LINE, fill="#ffffff", lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=fill, edgecolor=colour, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h - 0.19, title, ha="center", va="top",
            fontsize=10, fontweight="bold", color=INK, zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.46 - i * 0.24, line, ha="center", va="top",
                fontsize=8, color=INK2, zorder=3)


def arrow(ax, xy, xytext, colour=MUTED, style="-|>", lw=1.3, dashed=False):
    ax.add_patch(FancyArrowPatch(xytext, xy, arrowstyle=style, mutation_scale=11,
                                 color=colour, linewidth=lw, zorder=1,
                                 linestyle="--" if dashed else "-",
                                 shrinkA=2, shrinkB=2))


def main() -> int:
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 7.4); ax.axis("off")

    ax.text(0.2, 7.15, "P1 - one model, three runtimes, one measuring client",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(0.2, 6.85, "GTX 1650 SUPER 4 GB (Turing sm75) - i5-2500K, AVX only - 8 GB RAM - WSL2 / Ubuntu 22.04",
            fontsize=9, color=MUTED)

    # weight provenance
    box(ax, 0.2, 5.05, 2.5, 1.35, "Qwen/Qwen3-0.6B",
        ["safetensors, bf16", "downloaded once"], fill="#f4f7fd")
    box(ax, 3.15, 5.05, 2.5, 1.35, "GGUF F16",
        ["convert_hf_to_gguf.py", "--outtype f16"], fill="#f4f7fd")
    box(ax, 6.10, 5.05, 2.5, 1.35, "Ollama import",
        ["Modelfile FROM <gguf>", "not `ollama pull`"], fill="#f4f7fd")
    arrow(ax, (3.15, 5.72), (2.70, 5.72))
    arrow(ax, (6.10, 5.72), (5.65, 5.72))
    ax.text(8.75, 5.72, "311 tensors\nsha256 6ecf0bbb…\nverified identical",
            fontsize=8, color=MUTED, va="center", fontweight="bold")

    # client
    box(ax, 3.15, 3.45, 4.2, 1.15, "bench/client.py  -  one measuring client",
        ["streaming, TTFT on first non-empty chunk, usage cross-checked",
         "preflight: every runtime must report 64 prompt tokens"],
        colour="#2a2a28", lw=1.8)

    # runtimes
    specs = [
        (0.2, "llamacpp", "llama-server", ["/v1/completions", "-ngl 99  --parallel N", "-c N x ctx_per_slot"]),
        (3.65, "ollama", "Ollama", ["/api/generate  raw=true", "OLLAMA_NUM_PARALLEL=N", "F16 import, not a pull"]),
        (7.10, "vllm", "vLLM", ["/v1/completions", "--dtype float16 --enforce-eager", "TRITON_ATTN, no prefix cache"]),
    ]
    for x, key, title, lines in specs:
        box(ax, x, 1.35, 3.2, 1.45, title, lines, colour=SERIES[key], lw=1.8)
        arrow(ax, (x + 1.6, 2.80), (5.25, 3.45), colour=SERIES[key])

    # Background patch: the arrows pass straight through this line.
    ax.text(5.25, 3.02, "one at a time - the port is checked before every launch",
            ha="center", va="center", fontsize=8, color=MUTED, style="italic", zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor="none"))

    # gpu
    box(ax, 3.15, 0.18, 4.2, 0.82, "GTX 1650 SUPER - 4096 MiB",
        ["peak VRAM = device used - baseline captured before each launch"],
        colour="#2a2a28", fill="#f6f6f4")
    for x, key, *_ in specs:
        arrow(ax, (5.25, 1.00), (x + 1.6, 1.35), colour=SERIES[key], dashed=True)

    fig.tight_layout()
    out = Path("docs/architecture.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
