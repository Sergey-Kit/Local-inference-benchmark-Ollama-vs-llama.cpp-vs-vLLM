"""Figures for the README, built from results/summary.csv.

Chart choices follow the job each one has to do:

* TTFT single-stream and peak VRAM are magnitude across a handful of named
  things -> horizontal bars, sorted, directly labelled.
* Throughput vs concurrency is change across an ordered scale with one line per
  runtime -> lines. This is the figure the project exists for: continuous
  batching is the reason the curves separate.
* TTFT short vs long is magnitude across two nested categories -> grouped bars.

Colour is assigned by identity (runtime), in a fixed slot order, so a runtime
keeps its colour across every figure. Every value is directly labelled: three
of the palette's light-mode slots sit below 3:1 contrast on a light surface, so
identity is never carried by colour alone. summary.csv is the table view.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

# Reference categorical palette, light mode, slots 1-3 in fixed order. Runtimes
# are mapped to slots explicitly so a runtime never changes colour when a filter
# or a missing run changes how many series are on screen.
SERIES = {
    "llamacpp": "#2a78d6",   # slot 1, blue
    "ollama": "#eb6834",     # slot 2, orange
    "vllm": "#1baf7a",       # slot 3, aqua
}
LABELS = {"llamacpp": "llama.cpp", "ollama": "Ollama", "vllm": "vLLM"}
ORDER = ["llamacpp", "ollama", "vllm"]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#7a7973"
GRID = "#e6e5e1"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.edgecolor": GRID,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "figure.dpi": 160,
    })


def recede(ax, axis: str = "x") -> None:
    """Grid and spines are reference, not content."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=axis, linewidth=0.8, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def rounded_hbar(ax, y: float, width: float, height: float, color: str, radius_frac=0.35):
    """Horizontal bar with the data end rounded and the baseline end square.

    Rounding only the data end keeps the bar visually anchored to zero, which a
    fully rounded rectangle does not.
    """
    r = min(abs(width) * radius_frac, height * radius_frac)
    if r <= 0 or width <= 0:
        return
    y0, y1 = y - height / 2, y + height / 2
    x0, x1 = 0.0, width
    verts = [
        (x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + r),
        (x1, y1 - r), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=2))


def _present(df: pd.DataFrame) -> list[str]:
    return [r for r in ORDER if r in set(df["runtime"])]


def fig_ttft_single_stream(df: pd.DataFrame, out: Path) -> None:
    sub = df[(df.scenario == "concurrency_sweep") & (df.concurrency == 1)]
    if sub.empty:
        print("  skip ttft_single_stream: no data")
        return
    runtimes = _present(sub)
    values = [float(sub[sub.runtime == r].ttft_ms_p50.iloc[0]) for r in runtimes]

    fig, ax = plt.subplots(figsize=(6.8, 0.78 * len(runtimes) + 1.7))
    for i, (rt, v) in enumerate(zip(runtimes, values)):
        rounded_hbar(ax, i, v, 0.40, SERIES[rt])
        ax.text(v, i, f"  {v:,.0f} ms", va="center", ha="left",
                color=TEXT_PRIMARY, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(runtimes)), [LABELS[r] for r in runtimes])
    ax.set_ylim(len(runtimes) - 0.45, -0.55)   # padding, so bars clear the frame
    ax.set_xlim(0, max(values) * 1.32)
    ax.set_xlabel("median TTFT, ms  (lower is better)")
    ax.set_title("Time to first token, one request at a time")
    recede(ax, "x")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_throughput_vs_concurrency(df: pd.DataFrame, out: Path) -> None:
    sub = df[df.scenario == "concurrency_sweep"]
    if sub.empty:
        print("  skip throughput_vs_concurrency: no data")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xmax = 0
    for rt in _present(sub):
        s = sub[sub.runtime == rt].sort_values("concurrency")
        ax.plot(s.concurrency, s.throughput_tok_s, color=SERIES[rt], linewidth=2.0,
                marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=3, label=LABELS[rt])
        last = s.iloc[-1]
        xmax = max(xmax, float(last.concurrency))
        # Direct label at the line end, in text ink rather than the series
        # colour; the coloured marker beside it carries the identity.
        ax.annotate(f" {LABELS[rt]}: {last.throughput_tok_s:,.0f}",
                    (last.concurrency, last.throughput_tok_s),
                    textcoords="offset points", xytext=(8, 0), va="center",
                    color=TEXT_PRIMARY, fontsize=9, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sub.concurrency.unique()),
                  [str(int(c)) for c in sorted(sub.concurrency.unique())])
    ax.set_xlim(right=xmax * 2.6)
    ax.set_xlabel("concurrent requests")
    ax.set_ylabel("output tokens / s  (higher is better)")
    ax.set_title("Throughput vs concurrency")
    ax.legend(frameon=False, loc="upper left", labelcolor=TEXT_SECONDARY)
    recede(ax, "y")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_peak_vram(df: pd.DataFrame, out: Path) -> None:
    sub = df[(df.scenario == "concurrency_sweep") & df.peak_vram_mib.notna()]
    if sub.empty:
        print("  skip peak_vram: no data")
        return
    runtimes = _present(sub)
    values = [float(sub[sub.runtime == r].peak_vram_mib.max()) for r in runtimes]

    fig, ax = plt.subplots(figsize=(6.8, 0.78 * len(runtimes) + 2.0))
    for i, (rt, v) in enumerate(zip(runtimes, values)):
        rounded_hbar(ax, i, v, 0.40, SERIES[rt])
        ax.text(v, i, f"  {v:,.0f} MiB", va="center", ha="left",
                color=TEXT_PRIMARY, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(runtimes)), [LABELS[r] for r in runtimes])
    ax.set_ylim(len(runtimes) - 0.45, -0.55)
    ax.set_xlim(0, max(values) * 1.34)
    ax.set_xlabel("peak VRAM above idle baseline, MiB")
    ax.set_title("Peak VRAM")
    recede(ax, "x")
    fig.tight_layout()
    # Figure-level so it cannot collide with the marks.
    fig.text(0.01, -0.015,
             "vLLM reserves its pool up front (--gpu-memory-utilization), so its bar is a "
             "reservation,\nnot a high-water mark. The two are not the same measurement.",
             fontsize=8, color=TEXT_MUTED, va="top", ha="left")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_prompt_length(df: pd.DataFrame, out: Path) -> None:
    sub = df[df.scenario == "prompt_length"]
    if sub.empty:
        print("  skip prompt_length: no data")
        return
    runtimes = _present(sub)
    sets = [s for s in ("short", "long") if s in set(sub.prompt_set)]
    tokens = {s: int(sub[sub.prompt_set == s].prompt_tokens.iloc[0]) for s in sets}
    # Two nested categories; the palette's fixed slots encode runtime, so prompt
    # length is encoded by position and by an explicit label, not by a hue.
    alpha = {"short": 1.0, "long": 0.55}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.34
    minor_x, minor_labels = [], []
    for j, pset in enumerate(sets):
        for i, rt in enumerate(runtimes):
            row = sub[(sub.runtime == rt) & (sub.prompt_set == pset)]
            if row.empty or pd.isna(row.ttft_ms_p50.iloc[0]):
                continue
            v = float(row.ttft_ms_p50.iloc[0])
            x = i + (j - (len(sets) - 1) / 2) * (width + 0.02)  # 2px-equivalent gap
            ax.bar(x, v, width=width, color=SERIES[rt], alpha=alpha[pset],
                   edgecolor=SURFACE, linewidth=2, zorder=2)
            ax.text(x, v, f"{v:,.0f}", ha="center", va="bottom",
                    color=TEXT_PRIMARY, fontsize=9, fontweight="bold")
            minor_x.append(x)
            minor_labels.append(pset)
    # Prompt length on the minor ticks, runtime on the major ones with extra
    # pad, so the two label rows cannot overlap.
    ax.set_xticks(minor_x, minor_labels, minor=True)
    ax.tick_params(axis="x", which="minor", labelsize=8, labelcolor=TEXT_MUTED, length=0)
    ax.set_xticks(range(len(runtimes)), [LABELS[r] for r in runtimes])
    ax.tick_params(axis="x", which="major", pad=18)
    ax.set_ylabel("median TTFT, ms")
    ax.set_title(
        f"TTFT is prefill-bound: {tokens.get('short', '?')}-token vs "
        f"{tokens.get('long', '?')}-token prompt"
    )
    recede(ax, "y")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--out-dir", default="analysis/figures")
    args = ap.parse_args(argv)

    path = Path(args.summary)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run bench/run.py first")
    df = pd.read_csv(path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    style()
    print(f"plotting from {path} ({len(df)} rows)")
    fig_ttft_single_stream(df, out_dir / "ttft_single_stream.png")
    fig_throughput_vs_concurrency(df, out_dir / "throughput_vs_concurrency.png")
    fig_peak_vram(df, out_dir / "peak_vram.png")
    fig_prompt_length(df, out_dir / "ttft_prompt_length.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
