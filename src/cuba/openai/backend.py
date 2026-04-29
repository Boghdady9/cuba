"""
OpenAI-style chat backends: stub for local testing and
:class:`cuba.engine.batching.ContinuousBatchingEngine` for real runs.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from cuba.engine.batching import ContinuousBatchingEngine


def messages_to_prompt(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


class OpenAIInferenceBackend(Protocol):
    async def complete_chat(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> str: ...

    def stream_chat(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def health(self) -> dict[str, object]: ...


class StubOpenAIBackend:
    def __init__(self, model_id: str = "cuba") -> None:
        self.model_id = model_id

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def health(self) -> dict[str, object]:
        return {"ready": True, "backend": "stub"}

    async def complete_chat(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> str:
        return (
            f"[{self.model_id}] stub response (max_tokens={max_tokens}, temp={temperature}). "
            f"Input chars: {len(prompt)}."
        )

    async def stream_chat(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        text = await self.complete_chat(
            prompt, max_tokens=max_tokens, temperature=temperature
        )
        step = max(1, min(32, max(1, len(text) // 8)))
        for i in range(0, len(text), step):
            yield text[i : i + step]


class ContinuousBatchingOpenAIBackend:
    def __init__(self, engine: ContinuousBatchingEngine) -> None:
        self._engine = engine

    async def start(self) -> None:
        await self._engine.start()

    async def stop(self) -> None:
        await self._engine.stop()

    def health(self) -> dict[str, object]:
        stats = self._engine.scheduler_stats
        return {
            "ready": True,
            "backend": "engine",
            "scheduler": stats,
        }

    async def complete_chat(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> str:
        record = await self._engine.submit(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return await self._engine.wait_for_completion(record.id)

    async def stream_chat(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        record = await self._engine.submit(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        async for chunk in self._engine.stream_request(record.id):
            yield chunk
