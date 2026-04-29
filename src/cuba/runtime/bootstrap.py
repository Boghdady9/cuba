from __future__ import annotations

from dataclasses import dataclass

import torch

from cuba.engine.batching import ContinuousBatchingEngine, EngineConfig
from cuba.metrics import MetricsCollector
from cuba.openai.backend import ContinuousBatchingOpenAIBackend, StubOpenAIBackend
from cuba.ops.production import apply_cpu_optimizations, cpu_optimization_info


@dataclass(slots=True)
class RuntimeSettings:
    backend_mode: str = "engine"
    model_id: str = "cuba"
    model_path: str | None = None
    max_batch_size: int = 16
    max_wait_ms: int = 10
    max_concurrent_sequences: int = 32
    scheduler_tick_ms: int = 5
    num_threads: int | None = None
    interop_threads: int | None = None
    device: str = "cpu"


@dataclass(slots=True)
class RuntimeBundle:
    backend: object
    metrics: MetricsCollector
    engine: ContinuousBatchingEngine | None
    runtime_info: dict[str, object]


def build_runtime(settings: RuntimeSettings) -> RuntimeBundle:
    metrics = MetricsCollector()
    thread_info = apply_cpu_optimizations(
        num_threads=settings.num_threads,
        interop_threads=settings.interop_threads,
    )
    runtime_info: dict[str, object] = {
        "backend_mode": settings.backend_mode,
        "device": settings.device,
        "threads": thread_info,
        "torch": cpu_optimization_info(),
        "model_path": settings.model_path,
    }
    if settings.backend_mode == "stub":
        return RuntimeBundle(
            backend=StubOpenAIBackend(model_id=settings.model_id),
            metrics=metrics,
            engine=None,
            runtime_info=runtime_info,
        )

    # Load model and tokenizer
    model_path = str(settings.model_path).strip() if settings.model_path else None
    if not model_path:
        raise ValueError(
            "model_path is required. Set via --model-path, CUBA_MODEL_PATH, or MODEL_PATH env var. "
            "Example: CUBA_MODEL_PATH=gpt2"
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    # Set pad token if not already set (needed for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=settings.device if settings.device != "cpu" else None,
    )
    if settings.device == "cpu":
        model = model.to(settings.device)
    print("Model loaded successfully", flush=True)
    model.eval()
    config = EngineConfig(
        max_batch_size=settings.max_batch_size,
        max_wait_ms=settings.max_wait_ms,
        max_concurrent_sequences=settings.max_concurrent_sequences,
        scheduler_tick_ms=settings.scheduler_tick_ms,
        device=settings.device,
        forbid_early_eos=False,
    )
    engine = ContinuousBatchingEngine(
        model,
        tokenizer,
        config=config,
        metrics=metrics,
    )
    backend = ContinuousBatchingOpenAIBackend(engine)
    runtime_info["engine"] = {
        "max_batch_size": config.max_batch_size,
        "max_wait_ms": config.max_wait_ms,
        "max_concurrent_sequences": config.max_concurrent_sequences,
        "scheduler_tick_ms": config.scheduler_tick_ms,
    }
    return RuntimeBundle(
        backend=backend,
        metrics=metrics,
        engine=engine,
        runtime_info=runtime_info,
    )
