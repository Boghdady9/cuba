"""
Cuba: continuous batching, CPU ops, and OpenAI-style HTTP serving.

- ``cuba.engine`` — :class:`ContinuousBatchingEngine`
- ``cuba.ops`` — streaming, load balancing, dynamic batching, monitoring
- ``cuba.openai`` — FastAPI OpenAI Chat Completions API
"""
from __future__ import annotations

from cuba.engine.batching import ContinuousBatchingEngine
from cuba.metrics import MetricsCollector
from cuba.openai.api import create_app, default_app
from cuba.openai.backend import (
    ContinuousBatchingOpenAIBackend,
    OpenAIInferenceBackend,
    StubOpenAIBackend,
    messages_to_prompt,
)
from cuba.runtime import RuntimeSettings, build_runtime
from cuba.ops import (
    CPUDynamicBatching,
    CPUMemoryPool,
    InferenceMonitor,
    LoadBalanceStrategy,
    LoadBalancer,
    StreamingInference,
    apply_cpu_optimizations,
    cpu_optimization_info,
)

__all__ = [
    "CPUDynamicBatching",
    "CPUMemoryPool",
    "ContinuousBatchingEngine",
    "ContinuousBatchingOpenAIBackend",
    "InferenceMonitor",
    "LoadBalanceStrategy",
    "LoadBalancer",
    "MetricsCollector",
    "OpenAIInferenceBackend",
    "RuntimeSettings",
    "StreamingInference",
    "StubOpenAIBackend",
    "apply_cpu_optimizations",
    "build_runtime",
    "cpu_optimization_info",
    "create_app",
    "default_app",
    "messages_to_prompt",
]
