from __future__ import annotations

from cuba.metrics import MetricsCollector


def test_metrics_include_scheduler_fields() -> None:
    metrics = MetricsCollector()
    metrics.record_scheduler_state(queue_depth=3, active_sequences=2, scheduler_loop_seconds=0.01)
    metrics.record_scheduler_batch(batch_size=2, prompt_tokens=5, decode_tokens=4)
    metrics.record_request(0.2, 4, prompt_tokens=5, ttft_seconds=0.05)
    text = metrics.get_prometheus_metrics()
    assert "cuba_scheduler_queue_depth" in text
    assert "cuba_time_to_first_token_seconds" in text
    assert "cuba_prompt_tokens_total" in text
