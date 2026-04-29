"""Pytest configuration and fixtures for Cuba tests."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from cuba.engine.batching import ContinuousBatchingEngine, EngineConfig
from cuba.metrics import MetricsCollector
from cuba.runtime.bootstrap import RuntimeBundle


class TinyTokenizer:
    """Minimal tokenizer for testing engine behavior (not real model quality)."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        if not text:
            return [self.bos_token_id]
        return [min(255, b) for b in text.encode("utf-8", errors="ignore")] or [self.bos_token_id]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        out = []
        for tok in token_ids:
            if skip_special_tokens and int(tok) <= 2:
                continue
            out.append(chr(int(tok) % 128))
        return "".join(out)


class TinyModel(torch.nn.Module):
    """Minimal causal LM for testing engine scheduling (not real inference)."""

    def __init__(self, vocab_size: int = 256) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 32)
        self.proj = torch.nn.Linear(32, vocab_size, bias=False)

        with torch.no_grad():
            torch.manual_seed(42)
            for param in self.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> object:
        use_cache = kwargs.get("use_cache", False)
        past_kv = kwargs.get("past_key_values", None)
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        if use_cache:
            bsz, seq = input_ids.shape
            past_len = past_kv[0][0].shape[2] if past_kv is not None else 0
            k = torch.zeros(bsz, 1, past_len + seq, 1, device=input_ids.device)
            v = torch.zeros(bsz, 1, past_len + seq, 1, device=input_ids.device)
            return type("Output", (), {"logits": logits, "past_key_values": ((k, v),)})()
        return type("Output", (), {"logits": logits})()


@pytest.fixture
def tiny_runtime() -> RuntimeBundle:
    """Provide a tiny in-memory runtime with toy model for testing engine logic.

    Tests that verify engine scheduling behavior (not real model quality) use this
    fixture to avoid downloading real models. The toy model is deterministic so
    tests can verify exact token counts and streaming behavior.
    """
    tokenizer = TinyTokenizer()
    model = TinyModel(vocab_size=tokenizer.vocab_size)
    model.eval()

    config = EngineConfig(
        max_batch_size=4,
        max_wait_ms=5,
        max_concurrent_sequences=8,
        scheduler_tick_ms=2,
        device="cpu",
        forbid_early_eos=True,  # Force exact max_tokens for test assertions
    )

    engine = ContinuousBatchingEngine(model, tokenizer, config=config)
    metrics = MetricsCollector()

    # Create a minimal backend that uses the engine
    class TinyBackend:
        async def start(self) -> None:
            await engine.start()

        async def stop(self) -> None:
            await engine.stop()

        def health(self) -> dict[str, object]:
            return {"ready": True, "backend": "engine"}

        async def complete_chat(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            record = await engine.submit(prompt, max_tokens=max_tokens, temperature=temperature)
            return await engine.wait_for_completion(record.id)

        async def stream_chat(
            self, prompt: str, *, max_tokens: int, temperature: float
        ) -> AsyncIterator[str]:
            record = await engine.submit(prompt, max_tokens=max_tokens, temperature=temperature)
            async for chunk in engine.stream_request(record.id):
                yield chunk

    # Return a proper RuntimeBundle for test compatibility
    return RuntimeBundle(
        backend=TinyBackend(),
        metrics=metrics,
        engine=engine,
        runtime_info={
            "backend_mode": "engine",
            "device": "cpu",
            "model_path": None,
        },
    )


@pytest.fixture
def event_loop() -> Any:
    """Provide event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
