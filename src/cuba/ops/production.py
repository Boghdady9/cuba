"""
Production helpers: health/metrics, CPU tuning, and re-exports for streaming and load balancing.
"""
from __future__ import annotations

import os
import time
from typing import Any, Literal, Sequence

import torch

from cuba.engine.batching import ContinuousBatchingEngine

__all__ = [
    "InferenceMonitor",
    "apply_cpu_optimizations",
    "cpu_optimization_info",
]


def apply_cpu_optimizations(
    num_threads: int | None = None,
    interop_threads: int | None = None,
) -> dict[str, int]:
    """
    Apply conservative CPU-oriented PyTorch settings for inference throughput.
    Thread counts can be overridden via CUBA_TORCH_NUM_THREADS / CUBA_TORCH_INTEROP_THREADS
    (integer environment variables) or the function arguments.
    """
    default_interop = max(1, (os.cpu_count() or 2) // 4)
    default_intra = int(
        os.environ.get("CUBA_TORCH_NUM_THREADS", str(min(8, (os.cpu_count() or 4) - default_interop)))
    )
    n = int(num_threads) if num_threads is not None else default_intra
    i = (
        int(interop_threads)
        if interop_threads is not None
        else int(os.environ.get("CUBA_TORCH_INTEROP_THREADS", str(default_interop)))
    )
    n = max(1, n)
    i = max(1, i)
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(i)
    except (RuntimeError, ValueError):
        pass
    return {"intraop": n, "interop": i}


def cpu_optimization_info() -> dict[str, Any]:
    """Introspect current CPU / thread related settings (for monitoring)."""
    out: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    if torch.cuda.is_available():
        out["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
        }
    if hasattr(torch.backends, "mkl") and torch.backends.mkl.is_available():
        out["mkl"] = True
    return out


class InferenceMonitor:
    """Lightweight health and aggregate metrics; combines engine state with optional GPU memory."""

    def __init__(self, engine: ContinuousBatchingEngine, metrics_window: int = 200):
        self.engine = engine
        self.metrics_window = max(1, metrics_window)
        self._t0 = time.perf_counter()
        self._last_total_gen = 0
        self._last_tp_check = time.perf_counter()
        self._tp_initialized = False
        self.metrics: dict[str, int | float] = {
            "requests_processed": 0,
            "total_tokens_generated": 0,
            "average_latency": 0.0,
            "throughput_tokens_per_sec": 0.0,
        }

    def _rolling_latencies(self) -> list[float]:
        seq: Sequence[dict[str, Any]] = self.engine.completed_sequences
        if not seq:
            return []
        take = self.metrics_window
        out: list[float] = []
        for r in list(seq)[-take:]:
            t0, t1 = r.get("t_started"), r.get("t_completed")
            if t0 is not None and t1 is not None:
                out.append(float(t1) - float(t0))
        return out

    def get_health_status(self) -> dict[str, Any]:
        completed = self.engine.completed_sequences
        n = len(completed)
        toks = sum(int(c.get("generated_tokens", 0)) for c in completed)
        lats = self._rolling_latencies()
        avg_lat = (sum(lats) / len(lats)) if lats else 0.0
        now = time.perf_counter()
        if not self._tp_initialized:
            self._last_total_gen = toks
            self._last_tp_check = now
            self._tp_initialized = True
        else:
            window = now - self._last_tp_check
            if window >= 0.5:
                dt = toks - self._last_total_gen
                self._last_total_gen = toks
                self._last_tp_check = now
                self.metrics["throughput_tokens_per_sec"] = float(dt / window) if window > 0 else 0.0
        self.metrics["requests_processed"] = n
        self.metrics["total_tokens_generated"] = toks
        self.metrics["average_latency"] = avg_lat
        load = len(self.engine.running_sequences) + len(self.engine.pending_requests)
        mem = self._get_memory_usage()
        status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        if load > 2 * self.engine.max_batch_size:
            status = "degraded"
        return {
            "status": status,
            "active_requests": len(self.engine.running_sequences),
            "pending_requests": len(self.engine.pending_requests),
            "max_batch_size": self.engine.max_batch_size,
            "memory_usage": mem,
            "metrics": dict(self.metrics),
        }

    def _get_memory_usage(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if torch.cuda.is_available():
            out["gpu_allocated"] = float(torch.cuda.memory_allocated() / 1e9)
            out["gpu_reserved"] = float(torch.cuda.memory_reserved() / 1e9)
        return out
