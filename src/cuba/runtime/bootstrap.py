from __future__ import annotations

from dataclasses import dataclass

import torch

from cuba.engine.batching import ContinuousBatchingEngine, EngineConfig
from cuba.metrics import MetricsCollector
from cuba.openai.backend import ContinuousBatchingOpenAIBackend, StubOpenAIBackend
from cuba.ops.production import apply_cpu_optimizations, cpu_optimization_info


def _auto_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(device: str) -> torch.dtype:
    if device == "cuda":
        # bfloat16 on Ampere+ (A100, RTX 30/40xx); float16 on older.
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        return torch.bfloat16 if cap >= (8, 0) else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32  # CPU — no hardware fp16/bf16 on x86


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
    device: str = "auto"  # auto | cpu | cuda | mps | mlx


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

    # Resolve "auto" to a concrete device.
    device = settings.device
    if device == "auto":
        # MLX is only available on Apple Silicon; check that first.
        try:
            import mlx.core  # noqa: F401
            import platform
            if platform.machine() == "arm64":
                device = "mlx"
            else:
                device = _auto_device()
        except ImportError:
            device = _auto_device()

    runtime_info: dict[str, object] = {
        "backend_mode": settings.backend_mode,
        "device": device,
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

    model_path = str(settings.model_path).strip() if settings.model_path else None
    if not model_path:
        raise ValueError(
            "model_path is required. Set via --model-path, CUBA_MODEL_PATH, or MODEL_PATH env var. "
            "Example: CUBA_MODEL_PATH=gpt2"
        )

    # --- MLX backend (Apple Silicon only) ---
    if device == "mlx":
        return _build_mlx_runtime(model_path, settings, metrics, runtime_info)

    # --- PyTorch backend (CPU / CUDA / MPS) ---
    return _build_torch_runtime(model_path, device, settings, metrics, runtime_info)


def _build_torch_runtime(
    model_path: str,
    device: str,
    settings: RuntimeSettings,
    metrics: MetricsCollector,
    runtime_info: dict[str, object],
) -> RuntimeBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _dtype_for_device(device)

    # HF device_map: "auto" works for CUDA (multi-GPU); MPS/CPU need manual .to().
    hf_device_map = "auto" if device == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        device_map=hf_device_map,
    )
    if hf_device_map is None:
        model = model.to(device)

    print("Model loaded successfully", flush=True)
    model.eval()

    # Dynamic INT8 quantization: CPU only (not supported on MPS/CUDA without bitsandbytes).
    if device == "cpu":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("Dynamic INT8 quantization applied", flush=True)

    # torch.compile: CPU and CUDA only — MPS support is incomplete in PyTorch 2.x.
    if device in ("cpu", "cuda"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile applied (reduce-overhead)", flush=True)
        except Exception:
            pass

    runtime_info["dtype"] = str(dtype)
    config = EngineConfig(
        max_batch_size=settings.max_batch_size,
        max_wait_ms=settings.max_wait_ms,
        max_concurrent_sequences=settings.max_concurrent_sequences,
        scheduler_tick_ms=settings.scheduler_tick_ms,
        device=device,
        forbid_early_eos=False,
    )
    engine = ContinuousBatchingEngine(model, tokenizer, config=config, metrics=metrics)
    backend = ContinuousBatchingOpenAIBackend(engine)
    runtime_info["engine"] = {
        "max_batch_size": config.max_batch_size,
        "max_wait_ms": config.max_wait_ms,
        "max_concurrent_sequences": config.max_concurrent_sequences,
        "scheduler_tick_ms": config.scheduler_tick_ms,
    }
    return RuntimeBundle(backend=backend, metrics=metrics, engine=engine, runtime_info=runtime_info)


def _build_mlx_runtime(
    model_path: str,
    settings: RuntimeSettings,
    metrics: MetricsCollector,
    runtime_info: dict[str, object],
) -> RuntimeBundle:
    try:
        from mlx_lm import load as mlx_load
    except ImportError as e:
        raise ImportError(
            "mlx-lm is required for MLX backend. Install it with:\n"
            "  uv add mlx-lm\n"
            "or: pip install mlx-lm"
        ) from e

    from cuba.openai.backend import MLXOpenAIBackend

    print(f"Loading model from {model_path} with MLX...", flush=True)
    model, tokenizer = mlx_load(model_path)
    print("MLX model loaded successfully", flush=True)

    runtime_info["dtype"] = "mlx-default"
    backend = MLXOpenAIBackend(model, tokenizer)
    return RuntimeBundle(backend=backend, metrics=metrics, engine=None, runtime_info=runtime_info)
