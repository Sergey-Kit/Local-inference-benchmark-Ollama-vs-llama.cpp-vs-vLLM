"""Orchestrator: drives the runtime x concurrency x prompt-length matrix.

Owns the parts of the experiment that have to be identical across runtimes and
are easy to get subtly wrong by hand:

* exactly one runtime is alive at a time (3.5 GB of VRAM does not forgive two);
* the VRAM baseline is captured before each server starts, never reused across
  servers, because the Windows desktop's share of the card drifts by hundreds
  of MiB over a session;
* cold start is timed to a *completed generation*, not to a healthy port;
* warm-up traffic is discarded and every point is the median of n_runs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from bench.client import Prompt, RuntimeClient, Sampling
from bench.metrics import RunSummary, median_across_runs, summarize_run
from bench.monitor import ResourceMonitor


# --------------------------------------------------------------------------
# config plumbing
# --------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_prompts(path: str | Path) -> dict[str, list[Prompt]]:
    sets: dict[str, list[Prompt]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sets.setdefault(row["prompt_set"], []).append(
            Prompt(row["name"], row["prompt_set"], row["text"], row["n_tokens"])
        )
    if not sets:
        raise SystemExit(f"no prompts in {path}; run prompts/build_prompts.py first")
    for name, prompts in sets.items():
        lengths = {p.n_tokens for p in prompts}
        if len(lengths) != 1:
            raise SystemExit(f"prompt set '{name}' has mixed lengths {sorted(lengths)}")
    return sets


def substitution_vars(cfg: dict, profile: dict) -> dict[str, Any]:
    model = cfg["model"]
    return {
        **model,
        "parallel": profile["parallel"],
        "ctx_per_slot": profile["ctx_per_slot"],
        # llama-server's -c is the TOTAL context shared by all slots; passing
        # ctx_per_slot here would silently give each slot ctx/parallel.
        "ctx_total": profile["parallel"] * profile["ctx_per_slot"],
    }


def _fmt(value: Any, vars_: dict) -> str:
    return str(value).format(**vars_)


# --------------------------------------------------------------------------
# server lifecycle
# --------------------------------------------------------------------------


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


@dataclass
class ServerSession:
    runtime: str
    profile_name: str
    proc: subprocess.Popen
    log_path: Path
    cold_start_s: float


class ServerManager:
    """Launches one runtime at a time and guarantees it is gone afterwards."""

    def __init__(self, cfg: dict, runtime: str, log_dir: Path) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self.rt = cfg["runtimes"][runtime]
        self.log_dir = log_dir
        self.proc: subprocess.Popen | None = None
        self.log_file = None

    def launch(self, profile_name: str, profile: dict) -> subprocess.Popen:
        if self.proc is not None:
            raise RuntimeError("a server is already running under this manager")
        port = int(self.rt["port"])
        if port_is_open(port):
            raise SystemExit(
                f"port {port} is already in use -- refusing to start {self.runtime}. "
                "Two runtimes sharing the GPU would make every number meaningless. "
                f"Stop whatever is listening (e.g. `systemctl --user stop ollama`) and retry."
            )

        vars_ = substitution_vars(self.cfg, profile)
        cmd = [_fmt(part, vars_) for part in self.rt["command"]]
        env = dict(os.environ)
        for key, value in (self.rt.get("env") or {}).items():
            env[key] = _fmt(value, vars_)

        log_path = self.log_dir / f"{self.runtime}_{profile_name}.log"
        self.log_file = log_path.open("w")
        self.log_file.write(f"# {' '.join(cmd)}\n")
        self.log_file.flush()
        print(f"    launching: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group, so children die too
        )
        self._log_path = log_path
        return self.proc

    @property
    def log_path(self) -> Path:
        return self._log_path

    def check_alive(self) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            tail = "\n".join(self._log_path.read_text().splitlines()[-25:])
            raise SystemExit(
                f"{self.runtime} exited with code {self.proc.returncode} before "
                f"becoming ready.\n--- last lines of {self._log_path} ---\n{tail}"
            )

    def terminate(self, grace_s: float = 20.0) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + grace_s
            while self.proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.2)
            if self.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=10)
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        self.proc = None
        # The GPU does not always release immediately after the process dies.
        time.sleep(2.0)


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------


@dataclass
class Point:
    """One measurement point: a cell of the runtime x concurrency matrix."""

    scenario: str
    profile: str
    prompt_set: str
    concurrency: int


def expand_scenarios(cfg: dict) -> list[Point]:
    points: list[Point] = []
    for scenario in cfg["scenarios"]:
        sets = scenario["prompt_set"]
        sets = [sets] if isinstance(sets, str) else sets
        for prompt_set in sets:
            for conc in scenario["concurrency"]:
                points.append(Point(scenario["name"], scenario["profile"], prompt_set, conc))
    return points


def requests_for(measurement: dict, concurrency: int) -> int:
    """Enough requests that the server reaches steady state at this concurrency."""
    return max(
        int(measurement["requests_per_run_min"]),
        int(measurement["requests_per_run_factor"]) * concurrency,
    )


async def measure_point(
    client: RuntimeClient,
    point: Point,
    prompts: list[Prompt],
    sampling: Sampling,
    monitor: ResourceMonitor,
    measurement: dict,
    cold_start_s: float | None,
) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    n_requests = requests_for(measurement, point.concurrency)
    for run_index in range(int(measurement["n_runs"])):
        monitor.start()
        records, wall = await client.run_batch(
            prompts,
            sampling,
            scenario=point.scenario,
            concurrency=point.concurrency,
            n_requests=n_requests,
            run_index=run_index,
            prompt_offset=run_index * n_requests,
        )
        res = monitor.stop()
        summary = summarize_run(
            records,
            wall_clock_s=wall,
            peak_vram_mib=res.peak_vram_mib,
            peak_rss_mib=res.peak_rss_mib,
            # Cold start belongs to the server, not the run; attached to run 0
            # so it survives into the aggregate without being counted n_runs times.
            cold_start_s=cold_start_s if run_index == 0 else None,
        )
        summary.notes.extend(res.notes)
        summaries.append(summary)
        summary._records = records  # type: ignore[attr-defined]
        print(
            f"      run {run_index}: {summary.throughput_tok_s:7.2f} tok/s  "
            f"TTFT p50 {summary.ttft_ms_p50 or float('nan'):7.1f} ms  "
            f"TPOT p50 {summary.tpot_ms_p50 or float('nan'):6.2f} ms  "
            f"peakVRAM {summary.peak_vram_mib or float('nan'):7.1f} MiB"
            + (f"  [{'; '.join(summary.notes)}]" if summary.notes else "")
        )
    return summaries


async def run_runtime(
    cfg: dict,
    runtime: str,
    prompt_sets: dict[str, list[Prompt]],
    points: list[Point],
    out_dir: Path,
    raw_writer,
    on_point=None,
) -> list[dict]:
    rt = cfg["runtimes"][runtime]
    sampling = Sampling(**cfg["sampling"])
    measurement = cfg["measurement"]
    aggregates: list[dict] = []

    from transformers import AutoTokenizer  # local: keeps import cost off --dry-run

    tok_src = cfg["model"]["local_dir"]
    if not Path(tok_src).exists():
        tok_src = cfg["model"]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    # Group by server profile: one launch per profile, not per point.
    by_profile: dict[str, list[Point]] = {}
    for point in points:
        by_profile.setdefault(point.profile, []).append(point)

    for profile_name, profile_points in by_profile.items():
        profile = cfg["server_profiles"][profile_name]
        vars_ = substitution_vars(cfg, profile)
        print(f"\n  [{runtime}] profile '{profile_name}' "
              f"(parallel={profile['parallel']}, ctx/slot={profile['ctx_per_slot']})")

        manager = ServerManager(cfg, runtime, out_dir)
        monitor = ResourceMonitor(
            poll_interval_s=float(cfg["monitor"]["poll_interval_s"]),
            baseline_settle_s=float(cfg["monitor"]["baseline_settle_s"]),
        )
        baseline = monitor.capture_baseline()
        print(f"    VRAM baseline: {baseline:.0f} MiB" if baseline else "    VRAM baseline: n/a")

        try:
            proc = manager.launch(profile_name, profile)
            monitor.track_pid(proc.pid)

            async with RuntimeClient(
                runtime,
                rt["base_url"],
                _fmt(rt["served_model"], vars_),
                health_path=rt.get("health_path", "/health"),
                supports_ignore_eos=bool(rt.get("supports_ignore_eos", True)),
                timeout_s=float(measurement["request_timeout_s"]),
                count_tokens=count_tokens,
            ) as client:
                monitor.start()
                try:
                    cold_start_s = await client.wait_ready(timeout_s=600.0)
                finally:
                    load_res = monitor.stop()
                manager.check_alive()
                print(f"    cold start: {cold_start_s:.1f} s "
                      f"(load peak {load_res.peak_vram_mib or float('nan'):.0f} MiB over baseline)")

                # Warm up from the tail of the list, past every window the
                # measured runs will touch, so it cannot prime the prompt cache
                # for a prompt that is about to be measured.
                warm_set = prompt_sets[profile_points[0].prompt_set]
                reserve = max(
                    requests_for(measurement, p.concurrency) * int(measurement["n_runs"])
                    for p in profile_points
                )
                await client.warmup(warm_set, sampling,
                                    n=int(measurement["warmup_requests"]),
                                    prompt_offset=reserve)

                for point in profile_points:
                    print(f"    {point.scenario} / {point.prompt_set} / concurrency {point.concurrency} "
                          f"({requests_for(measurement, point.concurrency)} requests/run)")
                    summaries = await measure_point(
                        client, point, prompt_sets[point.prompt_set], sampling,
                        monitor, measurement, cold_start_s,
                    )
                    for summary in summaries:
                        for record in getattr(summary, "_records", []):
                            raw_writer(record.as_dict())
                    agg = median_across_runs(summaries)
                    agg["profile"] = profile_name
                    agg["prompt_tokens"] = prompt_sets[point.prompt_set][0].n_tokens
                    agg["max_tokens"] = sampling.max_tokens
                    agg["requests_per_run"] = requests_for(measurement, point.concurrency)
                    agg["vram_baseline_mib"] = baseline
                    aggregates.append(agg)
                    if on_point is not None:
                        on_point(agg)
        finally:
            manager.terminate()
            monitor.close()

    return aggregates


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "runtime", "scenario", "profile", "prompt_set", "prompt_tokens", "max_tokens",
    "concurrency", "requests_per_run", "n_runs", "n_failed_total", "throughput_tok_s", "ttft_ms_p50",
    "ttft_ms_p95", "tpot_ms_p50", "output_tokens_p50", "peak_vram_mib", "vram_baseline_mib",
    "peak_rss_mib", "cold_start_s", "notes",
]


POINT_KEY = ("runtime", "scenario", "prompt_set", "concurrency")


def merge_summary(new_rows: Iterable[dict], path: Path, quiet: bool = False) -> None:
    """Write summary.csv, replacing rows for the points just measured.

    Merging rather than overwriting matters twice over: the file is rewritten
    after every measurement point, so a run that is interrupted still leaves
    usable aggregates behind; and runtimes can be benchmarked in separate
    invocations -- which they have to be, since only one may hold the GPU --
    without the later run erasing the earlier one's results.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[tuple, dict] = {}
    if path.exists():
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                rows[tuple(str(row.get(k, "")) for k in POINT_KEY)] = row
    for row in new_rows:
        rows[tuple(str(row.get(k, "")) for k in POINT_KEY)] = row

    ordered = sorted(rows.values(), key=lambda r: (str(r.get("runtime")), str(r.get("scenario")),
                                                   str(r.get("prompt_set")), int(float(r.get("concurrency") or 0))))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    if not quiet:
        print(f"\nsummary.csv now holds {len(ordered)} measurement points -> {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--prompts", default="prompts/prompts.jsonl")
    ap.add_argument("--runtime", action="append", help="restrict to these runtimes (repeatable)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true", help="print the matrix, launch nothing")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    points = expand_scenarios(cfg)
    runtimes = args.runtime or list(cfg["runtimes"])
    unknown = set(runtimes) - set(cfg["runtimes"])
    if unknown:
        raise SystemExit(f"unknown runtime(s): {sorted(unknown)}")

    if args.dry_run:
        print(f"{len(runtimes)} runtime(s) x {len(points)} point(s) "
              f"x {cfg['measurement']['n_runs']} run(s) = "
              f"{len(runtimes) * len(points) * cfg['measurement']['n_runs']} runs")
        for runtime in runtimes:
            print(f"\n{runtime}:")
            for point in points:
                print(f"  {point.scenario:20s} profile={point.profile:12s} "
                      f"{point.prompt_set:6s} concurrency={point.concurrency:2d} "
                      f"requests/run={requests_for(cfg['measurement'], point.concurrency)}")
        return 0

    prompt_sets = load_prompts(args.prompts)
    out_dir = Path(args.out)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    summary_path = out_dir / "summary.csv"
    all_aggregates: list[dict] = []

    def on_point(agg: dict) -> None:
        # Flush after every point: this machine is slow enough that a full
        # matrix outlives most patience, and a partial result that survives is
        # worth more than a complete one that does not.
        merge_summary([agg], summary_path, quiet=True)

    for runtime in runtimes:
        raw_path = out_dir / "raw" / f"{runtime}_{stamp}.jsonl"
        print(f"\n=== {runtime} === (raw -> {raw_path})")
        with raw_path.open("w") as raw_fh:
            def raw_writer(row: dict, _fh=raw_fh) -> None:
                _fh.write(json.dumps(row) + "\n")
                _fh.flush()

            all_aggregates.extend(
                asyncio.run(run_runtime(cfg, runtime, prompt_sets, points, out_dir,
                                        raw_writer, on_point=on_point))
            )

    merge_summary([], summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
