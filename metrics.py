"""Measurement helpers for the research record: GPU memory, device telemetry,
statistics, and honest wall-clock timing.

reproduces (arXiv:2511.20639) reports exactly three numbers per cell
**Acc. / Token / Speed**, and two of those are only trustworthy if the
measurement around them is careful:

O(1) in how much the agents thought" is a claim about bytes and has to be
"""

import math
import subprocess
import threading
import time
from typing import Dict, List, Optional, Sequence

try:
    import torch
except Exception:  # pragma: no cover - torch is a hard dep everywhere else
    torch = None


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def cuda_sync(device=None) -> None:
    """Block until queued CUDA work has finished.
    """
    if torch is None or not torch.cuda.is_available():
        return
    if device is not None and torch.device(device).type != "cuda":
        return
    torch.cuda.synchronize()


class Phase:
    """Context manager returning the synced wall time of a block.
    Context manager returning the synced wall time of a block.

        with Phase(device) as p:
            ...
        p.sec
    """

    __slots__ = ("device", "sec", "_t0")

    def __init__(self, device=None) -> None:
        self.device = device
        self.sec = 0.0
        self._t0 = 0.0

    def __enter__(self) -> "Phase":
        cuda_sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        cuda_sync(self.device)
        self.sec = time.perf_counter() - self._t0
        return False  # never suppress


# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------

def reset_peak_memory() -> None:
    """Zero the peak-memory watermark for every visible device."""
    if torch is None or not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def memory_snapshot() -> Optional[Dict]:
    """Current and peak allocator figures, in MiB.
    Current and peak allocator figures, in MiB.

    `allocated` is what tensors hold; `reserved` is what the caching allocator
    has taken from the driver and is the number that decides whether a run
    fits on a 24 GB L4. Both are reported because they answer different
    questions and the gap between them is fragmentation.
    """
    if torch is None or not torch.cuda.is_available():
        return None
    mib = 2 ** 20
    out = {}
    for i in range(torch.cuda.device_count()):
        out[f"cuda:{i}"] = {
            "allocated_mib": round(torch.cuda.memory_allocated(i) / mib, 1),
            "reserved_mib": round(torch.cuda.memory_reserved(i) / mib, 1),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(i) / mib, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(i) / mib, 1),
        }
    return out


def peak_reserved_mib() -> Optional[float]:
    """Worst-case reserved memory over all devices, the 'does it fit' number."""
    snap = memory_snapshot()
    if not snap:
        return None
    return max(d["peak_reserved_mib"] for d in snap.values())


# ---------------------------------------------------------------------------
# Device description and live telemetry
# ---------------------------------------------------------------------------

def _nvidia_smi(query: str) -> Optional[List[List[str]]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=10,
        )
    except Exception:
        return None
    rows = []
    for line in out.decode(errors="replace").strip().splitlines():
        rows.append([c.strip() for c in line.split(",")])
    return rows or None


def device_details() -> Optional[Dict]:
    """Static device facts that change how fast the same code runs.
    Static device facts that change how fast the same code runs.

    An L4 and an A100 differ by more than their name: compute capability
    decides which attention kernels are eligible, the power cap decides
    whether sustained throughput matches the first batch, and the driver
    version moves kernel selection between releases. All of it belongs in the
    manifest next to the Speed number it explains.
    """
    if torch is None or not torch.cuda.is_available():
        return None
    info: Dict = {"devices": []}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append({
            "index": i,
            "name": props.name,
            "vram_gib": round(props.total_memory / 2 ** 30, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        })
    smi = _nvidia_smi("driver_version,power.limit,clocks.max.sm,clocks.max.mem")
    if smi:
        for i, row in enumerate(smi):
            if i < len(info["devices"]) and len(row) >= 4:
                info["devices"][i].update({
                    "driver_version": row[0],
                    "power_limit_w": row[1],
                    "max_sm_clock_mhz": row[2],
                    "max_mem_clock_mhz": row[3],
                })
    try:
        info["torch_cuda_build"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
        info["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        pass
    return info


class TelemetrySampler:
    """Background sampler for power / temperature / clocks (opt-in).
    """

    def __init__(self, path: str, interval: float = 5.0) -> None:
        self.path = path
        self.interval = max(1.0, float(interval))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples = 0

    def start(self) -> bool:
        if _nvidia_smi("index") is None:
            return False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        import json
        query = ("index,utilization.gpu,utilization.memory,memory.used,"
                 "temperature.gpu,power.draw,clocks.sm,clocks.mem")
        fields = query.split(",")[1:]
        while not self._stop.wait(self.interval):
            rows = _nvidia_smi(query)
            if not rows:
                continue
            ts = time.time()
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    for row in rows:
                        if len(row) < len(fields) + 1:
                            continue
                        rec = {"ts": round(ts, 2), "gpu": row[0]}
                        rec.update({k: row[i + 1] for i, k in enumerate(fields)})
                        f.write(json.dumps(rec) + "\n")
                self.samples += 1
            except Exception:
                return  # a telemetry failure must never take the run down

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_interval(correct: int, n: int, z: float = 1.96):
    """95% Wilson score interval for an accuracy.

    width near ±2.5 points at 90% accuracy, so an unqualified 1-point gap
    between `both` and `state_only` is not a result. Wilson rather than the
    normal approximation because it stays inside [0,1] and behaves at the
    small sample counts of AIME (30 questions).
    """
    if n <= 0:
        return None
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {
        "low": round(max(0.0, center - half), 4),
        "high": round(min(1.0, center + half), 4),
        "half_width": round(half, 4),
        "z": z,
    }


def summarize(values: Sequence[float], digits: int = 4) -> Optional[Dict]:
    """mean / median / p90 / min / max over a sample.
    mean / median / p90 / min / max over a sample.

    A mean alone cannot distinguish a run that was uniformly slow from one
    that was fast until the card got hot, and cannot show the cold first
    batch. The spread is the diagnostic.
    """
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None

    def pct(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = q * (len(vals) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), digits),
        "median": round(pct(0.5), digits),
        "p90": round(pct(0.9), digits),
        "min": round(vals[0], digits),
        "max": round(vals[-1], digits),
    }


__all__ = [
    "cuda_sync", "Phase", "reset_peak_memory", "memory_snapshot",
    "peak_reserved_mib", "device_details", "TelemetrySampler",
    "wilson_interval", "summarize",
]
