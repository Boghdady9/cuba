"""
In-process request metrics and Prometheus text exposition.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import Any

_PROM_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Monotonic per-process start for uptime
_START = time.perf_counter()


def _content_type() -> str:
    return _PROM_CONTENT_TYPE


def _process_rss_bytes() -> int | None:
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS: bytes; Linux: kilobytes
        if sys.platform == "darwin":
            return int(r)
        return int(r) * 1024
    except Exception:
        return None


def _gpu_memory_bytes() -> dict[str, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "cuda_allocated": float(torch.cuda.memory_allocated()),
            "cuda_reserved": float(torch.cuda.memory_reserved()),
        }
    except Exception:
        return None


class MetricsCollector:
    """
    Counters, duration observations, and batch-size samples with optional RSS / CUDA gauges.
    Thread-safe for multi-worker; use one instance per process.
    """

    def __init__(self, max_samples: int = 10_000) -> None:
        self._lock = threading.RLock()
        self._max_samples = max(100, int(max_samples))
        self._requests_total = 0
        self._tokens_total = 0
        self._prompt_tokens_total = 0
        self._decode_tokens_total = 0
        # bounded deques to avoid unbounded memory
        self._durations: deque[float] = deque(maxlen=self._max_samples)
        self._ttft_seconds: deque[float] = deque(maxlen=self._max_samples)
        self._batch_sizes: deque[int] = deque(maxlen=self._max_samples)
        self._memory_bytes: deque[int] = deque(maxlen=self._max_samples)
        self._queue_depths: deque[int] = deque(maxlen=self._max_samples)
        self._active_sequences: deque[int] = deque(maxlen=self._max_samples)
        self._scheduler_loop_seconds: deque[float] = deque(maxlen=self._max_samples)

    def record_request(
        self,
        duration: float,
        tokens_generated: int,
        batch_size: int = 1,
        *,
        memory_bytes: int | None = None,
        prompt_tokens: int = 0,
        ttft_seconds: float | None = None,
    ) -> None:
        """Record one completed request (wall-clock ``duration`` in seconds)."""
        with self._lock:
            self._requests_total += 1
            self._tokens_total += max(0, int(tokens_generated))
            self._decode_tokens_total += max(0, int(tokens_generated))
            self._prompt_tokens_total += max(0, int(prompt_tokens))
            self._durations.append(float(duration))
            self._batch_sizes.append(max(0, int(batch_size)))
            if ttft_seconds is not None:
                self._ttft_seconds.append(float(ttft_seconds))
            if memory_bytes is not None:
                self._memory_bytes.append(int(memory_bytes))

    def record_scheduler_batch(
        self,
        *,
        batch_size: int,
        prompt_tokens: int,
        decode_tokens: int,
    ) -> None:
        with self._lock:
            self._batch_sizes.append(max(0, int(batch_size)))
            self._prompt_tokens_total += max(0, int(prompt_tokens))
            self._decode_tokens_total += max(0, int(decode_tokens))

    def record_scheduler_state(
        self,
        *,
        queue_depth: int,
        active_sequences: int,
        scheduler_loop_seconds: float,
    ) -> None:
        with self._lock:
            self._queue_depths.append(max(0, int(queue_depth)))
            self._active_sequences.append(max(0, int(active_sequences)))
            self._scheduler_loop_seconds.append(float(scheduler_loop_seconds))

    def record_memory(self, memory_bytes: int) -> None:
        """Record a process memory sample (e.g. RSS) for the rolling buffer."""
        with self._lock:
            self._memory_bytes.append(int(memory_bytes))

    def as_dict(self) -> dict[str, Any]:
        """Structured snapshot (not Prometheus format)."""
        with self._lock:
            return {
                "requests_total": self._requests_total,
                "tokens_generated_total": self._tokens_total,
                "prompt_tokens_total": self._prompt_tokens_total,
                "decode_tokens_total": self._decode_tokens_total,
                "request_duration_seconds": list(self._durations),
                "ttft_seconds": list(self._ttft_seconds),
                "batch_size_histogram": list(self._batch_sizes),
                "memory_usage_bytes": list(self._memory_bytes),
                "queue_depth": list(self._queue_depths),
                "active_sequences": list(self._active_sequences),
                "scheduler_loop_seconds": list(self._scheduler_loop_seconds),
            }

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus text format (0.0.4)."""
        with self._lock:
            req = self._requests_total
            tok = self._tokens_total
            prompt_tok = self._prompt_tokens_total
            decode_tok = self._decode_tokens_total
            durations = list(self._durations)
            ttfts = list(self._ttft_seconds)
            batches = list(self._batch_sizes)
            mems = list(self._memory_bytes)
            queue_depths = list(self._queue_depths)
            active_sequences = list(self._active_sequences)
            scheduler_loops = list(self._scheduler_loop_seconds)

        n = len(durations)
        dur_sum = float(sum(durations)) if n else 0.0
        batch_sum = int(sum(batches)) if batches else 0
        le_dur = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf")]
        buck_d: list[int] = []
        for upper in le_dur:
            if upper == float("inf"):
                buck_d.append(n)
            else:
                buck_d.append(sum(1 for v in durations if v <= upper))

        lines: list[str] = []
        lines.append("# HELP cuba_up Process is up (always 1 when scraped).")
        lines.append("# TYPE cuba_up gauge")
        lines.append("cuba_up 1")

        lines.append("# HELP cuba_process_uptime_seconds Time since this process started.")
        lines.append("# TYPE cuba_process_uptime_seconds gauge")
        lines.append(f"cuba_process_uptime_seconds {time.perf_counter() - _START:.6f}")

        lines.append("# HELP cuba_requests_total Total chat completion requests observed.")
        lines.append("# TYPE cuba_requests_total counter")
        lines.append(f"cuba_requests_total {req}")

        lines.append("# HELP cuba_tokens_generated_total Total (rough) completion tokens recorded.")
        lines.append("# TYPE cuba_tokens_generated_total counter")
        lines.append(f"cuba_tokens_generated_total {tok}")
        lines.append("# HELP cuba_prompt_tokens_total Total prompt tokens observed.")
        lines.append("# TYPE cuba_prompt_tokens_total counter")
        lines.append(f"cuba_prompt_tokens_total {prompt_tok}")
        lines.append("# HELP cuba_decode_tokens_total Total decode tokens observed by the scheduler.")
        lines.append("# TYPE cuba_decode_tokens_total counter")
        lines.append(f"cuba_decode_tokens_total {decode_tok}")

        lines.append("# HELP cuba_request_duration_seconds Wall time of chat handler (seconds).")
        lines.append("# TYPE cuba_request_duration_seconds histogram")
        for b, c in zip(le_dur, buck_d, strict=False):
            le = str(b) if b != float("inf") else "+Inf"
            lines.append(f'cuba_request_duration_seconds_bucket{{le="{le}"}} {c}')
        lines.append(f"cuba_request_duration_seconds_sum {dur_sum}")
        lines.append(f"cuba_request_duration_seconds_count {n}")

        if ttfts:
            lines.append("# HELP cuba_time_to_first_token_seconds Time to first streamed token.")
            lines.append("# TYPE cuba_time_to_first_token_seconds summary")
            lines.append(f"cuba_time_to_first_token_seconds_sum {float(sum(ttfts))}")
            lines.append(f"cuba_time_to_first_token_seconds_count {len(ttfts)}")

        le_b = [1, 2, 4, 8, 16, 32, 64, float("inf")]
        bn = len(batches)
        buck_b: list[int] = []
        for upper in le_b:
            if upper == float("inf"):
                buck_b.append(bn)
            else:
                buck_b.append(sum(1 for b in batches if b <= int(upper)))

        lines.append("# HELP cuba_request_batch_size Batch size (API calls default to 1).")
        lines.append("# TYPE cuba_request_batch_size histogram")
        for b, c in zip(le_b, buck_b, strict=False):
            le = str(int(b)) if b != float("inf") else "+Inf"
            lines.append(f'cuba_request_batch_size_bucket{{le="{le}"}} {c}')
        lines.append(f"cuba_request_batch_size_sum {float(batch_sum)}")
        lines.append(f"cuba_request_batch_size_count {bn}")

        rss = _process_rss_bytes()
        if rss is not None:
            lines.append("# HELP cuba_process_resident_memory_bytes Process RSS (best-effort).")
            lines.append("# TYPE cuba_process_resident_memory_bytes gauge")
            lines.append(f"cuba_process_resident_memory_bytes {rss}")

        if mems:
            lines.append("# HELP cuba_recorded_memory_bytes_last Last recorded memory samples (rolling).")
            lines.append("# TYPE cuba_recorded_memory_bytes_last gauge")
            lines.append(f"cuba_recorded_memory_bytes_last {float(mems[-1])}")

        if queue_depths:
            lines.append("# HELP cuba_scheduler_queue_depth Current queued requests sample.")
            lines.append("# TYPE cuba_scheduler_queue_depth gauge")
            lines.append(f"cuba_scheduler_queue_depth {float(queue_depths[-1])}")
        if active_sequences:
            lines.append("# HELP cuba_scheduler_active_sequences Active sequences sample.")
            lines.append("# TYPE cuba_scheduler_active_sequences gauge")
            lines.append(f"cuba_scheduler_active_sequences {float(active_sequences[-1])}")
        if scheduler_loops:
            lines.append("# HELP cuba_scheduler_loop_seconds Scheduler loop duration sample.")
            lines.append("# TYPE cuba_scheduler_loop_seconds gauge")
            lines.append(f"cuba_scheduler_loop_seconds {float(scheduler_loops[-1])}")

        gpu = _gpu_memory_bytes()
        if gpu is not None:
            lines.append("# HELP cuba_cuda_memory_allocated_bytes torch.cuda memory allocated.")
            lines.append("# TYPE cuba_cuda_memory_allocated_bytes gauge")
            lines.append(f"cuba_cuda_memory_allocated_bytes {gpu['cuda_allocated']}")
            lines.append("# HELP cuba_cuda_memory_reserved_bytes torch.cuda memory reserved.")
            lines.append("# TYPE cuba_cuda_memory_reserved_bytes gauge")
            lines.append(f"cuba_cuda_memory_reserved_bytes {gpu['cuda_reserved']}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def content_type() -> str:
        return _content_type()
