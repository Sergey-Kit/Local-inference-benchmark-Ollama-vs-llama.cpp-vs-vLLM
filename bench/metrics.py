"""Latency/throughput math for the benchmark.

Deliberately free of I/O and of any runtime-specific knowledge: everything here
is a pure function over `RequestRecord`s, which is what makes it unit-testable
and what lets P2/P5 reuse it unchanged.

Definitions follow SPEC section 2:
    TTFT       time from sending the request to the first streamed token
    TPOT       (total_time - TTFT) / (n_output_tokens - 1)
    throughput total output tokens across a run / wall-clock of that run
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence


@dataclass
class RequestRecord:
    """One completed (or failed) request."""

    runtime: str
    scenario: str
    prompt_set: str
    concurrency: int
    run_index: int
    prompt_tokens: int
    output_tokens: int
    ttft_s: float | None
    total_s: float
    ok: bool = True
    error: str | None = None
    # Populated when the runtime's reported usage disagrees with our tokenizer.
    token_count_source: str = "usage"
    # Diagnostics: "the runtime stopped early" and "the runtime did not report
    # usage" produce the same low token count but are completely different
    # problems, and only finish_reason separates them.
    finish_reason: str | None = None
    n_chunks: int = 0
    # Relative gap between the runtime's usage and our own tokenizer count.
    token_drift: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunSummary:
    """Aggregate of one run (one pass over `requests_per_run` requests)."""

    runtime: str
    scenario: str
    prompt_set: str
    concurrency: int
    run_index: int
    n_requests: int
    n_failed: int
    wall_clock_s: float
    total_output_tokens: int
    throughput_tok_s: float
    ttft_ms_p50: float | None
    ttft_ms_p95: float | None
    tpot_ms_p50: float | None
    output_tokens_p50: float | None
    token_drift_max: float | None = None
    peak_vram_mib: float | None = None
    peak_rss_mib: float | None = None
    cold_start_s: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class MetricsError(ValueError):
    """Raised when a measurement is not trustworthy enough to report."""


def ttft_ms(record: RequestRecord) -> float | None:
    """Time to first token, in milliseconds."""
    if record.ttft_s is None:
        return None
    return record.ttft_s * 1000.0


def tpot_ms(record: RequestRecord) -> float | None:
    """Time per output token, in milliseconds.

    Returns None when the request produced fewer than two tokens: with a single
    token there is no inter-token interval to measure, and dividing by zero (or
    silently reporting the TTFT again) would quietly corrupt the aggregate.
    """
    if record.ttft_s is None or record.output_tokens < 2:
        return None
    decode_s = record.total_s - record.ttft_s
    if decode_s < 0:
        raise MetricsError(
            f"total_s ({record.total_s}) < ttft_s ({record.ttft_s}) -- clock went backwards"
        )
    return decode_s / (record.output_tokens - 1) * 1000.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile, matching numpy's default method.

    `p` is a percentage in [0, 100].
    """
    if not values:
        raise MetricsError("percentile of an empty sequence")
    if not 0.0 <= p <= 100.0:
        raise MetricsError(f"percentile p must be in [0, 100], got {p}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (p / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def median(values: Sequence[float]) -> float:
    """Median. Kept as a named function so call sites read as the SPEC does."""
    return percentile(values, 50.0)


def summarize_run(
    records: Iterable[RequestRecord],
    *,
    wall_clock_s: float,
    peak_vram_mib: float | None = None,
    peak_rss_mib: float | None = None,
    cold_start_s: float | None = None,
) -> RunSummary:
    """Collapse the requests of a single run into one RunSummary.

    Throughput is total output tokens over the run's wall clock -- deliberately
    including the failed requests' share of the clock, because a runtime that
    drops requests under load should not be rewarded for it.
    """
    records = list(records)
    if not records:
        raise MetricsError("cannot summarize a run with no requests")
    if wall_clock_s <= 0:
        raise MetricsError(f"wall_clock_s must be positive, got {wall_clock_s}")

    head = records[0]
    ok = [r for r in records if r.ok]
    n_failed = len(records) - len(ok)

    total_output = sum(r.output_tokens for r in ok)
    ttfts = [v for v in (ttft_ms(r) for r in ok) if v is not None]
    tpots = [v for v in (tpot_ms(r) for r in ok) if v is not None]

    notes: list[str] = []
    if n_failed:
        notes.append(f"{n_failed}/{len(records)} requests failed")
    sources = {r.token_count_source for r in ok}
    if sources - {"usage"}:
        notes.append(f"token counts from {sorted(sources)}")


    return RunSummary(
        runtime=head.runtime,
        scenario=head.scenario,
        prompt_set=head.prompt_set,
        concurrency=head.concurrency,
        run_index=head.run_index,
        n_requests=len(records),
        n_failed=n_failed,
        wall_clock_s=wall_clock_s,
        total_output_tokens=total_output,
        throughput_tok_s=total_output / wall_clock_s,
        ttft_ms_p50=percentile(ttfts, 50.0) if ttfts else None,
        ttft_ms_p95=percentile(ttfts, 95.0) if ttfts else None,
        tpot_ms_p50=percentile(tpots, 50.0) if tpots else None,
        # Without ignore_eos the runtimes choose their own output lengths.
        # That does not bias the rate metrics, but a runtime that consistently
        # generates less is doing less work, and that has to be visible.
        output_tokens_p50=percentile([r.output_tokens for r in ok], 50.0) if ok else None,
        token_drift_max=max((r.token_drift for r in ok if r.token_drift), default=None),
        peak_vram_mib=peak_vram_mib,
        peak_rss_mib=peak_rss_mib,
        cold_start_s=cold_start_s,
        notes=notes,
    )


def median_across_runs(summaries: Sequence[RunSummary]) -> dict:
    """Median of each metric across the >=3 runs of one measurement point.

    SPEC section 2 asks for the median of at least 3 runs; the warm-up run is
    dropped by the caller before it gets here.
    """
    if not summaries:
        raise MetricsError("cannot aggregate zero runs")
    head = summaries[0]
    key = (head.runtime, head.scenario, head.prompt_set, head.concurrency)
    for s in summaries:
        if (s.runtime, s.scenario, s.prompt_set, s.concurrency) != key:
            raise MetricsError(
                f"refusing to aggregate mismatched measurement points: {key} vs "
                f"{(s.runtime, s.scenario, s.prompt_set, s.concurrency)}"
            )

    def med(attr: str) -> float | None:
        vals = [getattr(s, attr) for s in summaries]
        vals = [v for v in vals if v is not None]
        return median(vals) if vals else None

    notes = sorted({n for s in summaries for n in s.notes})
    return {
        "runtime": head.runtime,
        "scenario": head.scenario,
        "prompt_set": head.prompt_set,
        "concurrency": head.concurrency,
        "n_runs": len(summaries),
        "n_failed_total": sum(s.n_failed for s in summaries),
        "throughput_tok_s": med("throughput_tok_s"),
        "ttft_ms_p50": med("ttft_ms_p50"),
        "ttft_ms_p95": med("ttft_ms_p95"),
        "tpot_ms_p50": med("tpot_ms_p50"),
        "output_tokens_p50": med("output_tokens_p50"),
        # Worst disagreement between the runtime's usage and our tokenizer,
        # across every run of this point. A percent or two is the expected cost
        # of the detokenise/retokenise round trip.
        "token_drift_max": max((s.token_drift_max for s in summaries
                                if s.token_drift_max is not None), default=None),
        "peak_vram_mib": med("peak_vram_mib"),
        "peak_rss_mib": med("peak_rss_mib"),
        "cold_start_s": med("cold_start_s"),
        "notes": "; ".join(notes),
    }
