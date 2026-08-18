"""Background sampler for GPU memory and process RSS.

Two things about this bench force the design:

1. WSL2 does not expose per-process GPU memory. `nvidia-smi
   --query-compute-apps=pid,used_memory` returns an empty table even while a
   CUDA process is resident, so there is no way to ask "how much VRAM did the
   server use". We therefore sample device-wide `used` and subtract a baseline
   captured before the server starts. The desktop compositor sits on ~544 MiB
   and drifts, which is why the baseline is a median over several samples and
   why its spread is reported alongside the result.

2. One `nvidia-smi` invocation costs ~100 ms on this machine, so the 100 ms
   polling interval the SPEC asks for would peg a core with process spawns.
   NVML is queried through a persistent handle instead (microseconds); the
   nvidia-smi path exists only as a fallback.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field

try:  # pynvml is optional so the harness still imports on a CPU-only box
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - exercised only without nvidia-ml-py
    pynvml = None

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None


MIB = 1024 * 1024


class GpuSampler:
    """Device-wide GPU memory in MiB, via NVML when available."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.backend = "unavailable"
        self._handle = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self.backend = "nvml"
            except Exception:
                self._handle = None
        if self._handle is None and shutil.which("nvidia-smi"):
            self.backend = "nvidia-smi"

    def used_mib(self) -> float | None:
        if self.backend == "nvml":
            try:
                return pynvml.nvmlDeviceGetMemoryInfo(self._handle).used / MIB
            except Exception:
                return None
        if self.backend == "nvidia-smi":
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.device_index}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return float(out.stdout.strip().splitlines()[0])
            except Exception:
                return None
        return None

    def close(self) -> None:
        if self.backend == "nvml":
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _tree_rss_mib(pid: int) -> float | None:
    """RSS of a process and its children, in MiB.

    llama.cpp with partial offload and Ollama's spawned runner both put real
    weight in host memory, so the server's own RSS alone would understate it.
    """
    if psutil is None:
        return None
    try:
        proc = psutil.Process(pid)
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                continue
        return total / MIB
    except psutil.Error:
        return None


@dataclass
class MonitorResult:
    gpu_backend: str
    baseline_vram_mib: float | None
    baseline_spread_mib: float | None
    peak_vram_mib: float | None
    peak_vram_total_mib: float | None
    peak_rss_mib: float | None
    n_samples: int
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "gpu_backend": self.gpu_backend,
            "baseline_vram_mib": self.baseline_vram_mib,
            "baseline_spread_mib": self.baseline_spread_mib,
            "peak_vram_mib": self.peak_vram_mib,
            "peak_vram_total_mib": self.peak_vram_total_mib,
            "peak_rss_mib": self.peak_rss_mib,
            "n_samples": self.n_samples,
            "notes": "; ".join(self.notes),
        }


class ResourceMonitor:
    """Usage:

        mon = ResourceMonitor()
        mon.capture_baseline()      # BEFORE the server starts
        proc = launch_server()
        mon.track_pid(proc.pid)
        mon.start()
        ...                          # run the benchmark
        result = mon.stop()
    """

    def __init__(
        self,
        poll_interval_s: float = 0.1,
        baseline_settle_s: float = 2.0,
        device_index: int = 0,
    ) -> None:
        self.poll_interval_s = poll_interval_s
        self.baseline_settle_s = baseline_settle_s
        self.gpu = GpuSampler(device_index)
        self._pid: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._vram_samples: list[float] = []
        self._rss_samples: list[float] = []
        self._notes: list[str] = []
        self.baseline_mib: float | None = None
        self.baseline_spread_mib: float | None = None

    def capture_baseline(self) -> float | None:
        """Sample the idle GPU so the server's share can be isolated later."""
        samples: list[float] = []
        deadline = time.monotonic() + self.baseline_settle_s
        while time.monotonic() < deadline:
            v = self.gpu.used_mib()
            if v is not None:
                samples.append(v)
            time.sleep(self.poll_interval_s)
        if not samples:
            self._notes.append("no GPU baseline available")
            return None
        self.baseline_mib = statistics.median(samples)
        self.baseline_spread_mib = max(samples) - min(samples)
        if self.baseline_spread_mib > 64:
            # Something other than the benchmark is actively using the card.
            self._notes.append(
                f"noisy GPU baseline: spread {self.baseline_spread_mib:.0f} MiB"
            )
        return self.baseline_mib

    def track_pid(self, pid: int) -> None:
        self._pid = pid

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("monitor already started")
        self._stop.clear()
        self._vram_samples.clear()
        self._rss_samples.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="monitor")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self.gpu.used_mib()
            if v is not None:
                self._vram_samples.append(v)
            if self._pid is not None:
                r = _tree_rss_mib(self._pid)
                if r is not None:
                    self._rss_samples.append(r)
            self._stop.wait(self.poll_interval_s)

    def stop(self) -> MonitorResult:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None

        peak_total = max(self._vram_samples) if self._vram_samples else None
        peak_delta = None
        if peak_total is not None and self.baseline_mib is not None:
            peak_delta = peak_total - self.baseline_mib
            if peak_delta < 0:
                # The desktop released memory after the baseline was taken.
                self._notes.append("peak below baseline; clamped to 0")
                peak_delta = 0.0

        return MonitorResult(
            gpu_backend=self.gpu.backend,
            baseline_vram_mib=self.baseline_mib,
            baseline_spread_mib=self.baseline_spread_mib,
            peak_vram_mib=peak_delta,
            peak_vram_total_mib=peak_total,
            peak_rss_mib=max(self._rss_samples) if self._rss_samples else None,
            n_samples=len(self._vram_samples),
            notes=list(self._notes),
        )

    def close(self) -> None:
        self.gpu.close()
