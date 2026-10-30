"""Dependency-free Prometheus-style metrics."""

from __future__ import annotations

import os
import threading
from typing import Dict, Iterable, List, Sequence, Tuple

LATENCY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
Labels = Tuple[Tuple[str, str], ...]


def _labels(values: Dict[str, str]) -> Labels:
    return tuple(sorted(values.items()))


def _render(name: str, labels: Labels, value: float) -> str:
    if not labels:
        return f"{name} {value:g}"
    inner = ",".join(
        '{}="{}"'.format(k, str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n"))
        for k, v in labels
    )
    return f"{name}{{{inner}}} {value:g}"


class Metrics:
    """Thread-safe counters and latency histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[Tuple[str, Labels], float] = {}
        self._histograms: Dict[Tuple[str, Labels], List[float]] = {}
        self._help: Dict[str, Tuple[str, str]] = {}

    def describe(self, name: str, kind: str, text: str) -> None:
        self._help[name] = (kind, text)

    def inc(self, name: str, amount: float = 1.0, **labels: str) -> None:
        key = (name, _labels(labels))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + amount

    def observe(self, name: str, seconds: float, **labels: str) -> None:
        key = (name, _labels(labels))
        with self._lock:
            state = self._histograms.setdefault(key, [0.0] * (len(LATENCY_BUCKETS) + 2))
            for i, bound in enumerate(LATENCY_BUCKETS):
                if seconds <= bound:
                    state[i] += 1
            state[-2] += 1  # +Inf bucket / count
            state[-1] += seconds  # sum

    def quantile(self, name: str, q: float, **labels: str) -> float | None:
        """Bucket-resolution quantile estimate, for diagnostics and tests."""
        with self._lock:
            state = self._histograms.get((name, _labels(labels)))
            if not state or state[-2] == 0:
                return None
            target = q * state[-2]
            for i, bound in enumerate(LATENCY_BUCKETS):
                if state[i] >= target:
                    return bound
            return float("inf")

    def render(self, gauges: Iterable[Tuple[str, Dict[str, str], float]] = ()) -> str:
        lines: List[str] = []
        with self._lock:
            counters = sorted(self._counters.items())
            histograms = sorted(self._histograms.items())
        seen = set()

        def header(name: str, kind: str) -> None:
            if name not in seen:
                seen.add(name)
                text = self._help.get(name, (kind, name))[1]
                lines.append(f"# HELP {name} {text}")
                lines.append(f"# TYPE {name} {kind}")

        for (name, labels), value in counters:
            header(name, "counter")
            lines.append(_render(name, labels, value))
        for (name, labels), state in histograms:
            header(name, "histogram")
            for bound, count in zip(LATENCY_BUCKETS, state, strict=False):
                lines.append(_render(f"{name}_bucket", labels + (("le", f"{bound:g}"),), count))
            lines.append(_render(f"{name}_bucket", labels + (("le", "+Inf"),), state[-2]))
            lines.append(_render(f"{name}_sum", labels, state[-1]))
            lines.append(_render(f"{name}_count", labels, state[-2]))
        for name, gauge_labels, value in gauges:
            header(name, "gauge")
            lines.append(_render(name, _labels(gauge_labels), value))
        return "\n".join(lines) + "\n"


def process_memory_bytes() -> Dict[str, float]:
    """Resident and peak memory of this process (Linux; zeros elsewhere)."""
    result = {"rss": 0.0, "peak": 0.0}
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    result["rss"] = float(line.split()[1]) * 1024
                elif line.startswith("VmHWM:"):
                    result["peak"] = float(line.split()[1]) * 1024
    except OSError:
        try:
            import resource

            result["peak"] = result["rss"] = float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
        except Exception:
            pass
    return result


def open_file_descriptors() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return 0


def histogram_buckets() -> Sequence[float]:
    return LATENCY_BUCKETS
