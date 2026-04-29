"""Token streaming over a :class:`cuba.engine.batching.ContinuousBatchingEngine`."""
from __future__ import annotations

from typing import Any, AsyncIterator

from cuba.engine.batching import ContinuousBatchingEngine


class StreamingInference:
    def __init__(self, engine: ContinuousBatchingEngine):
        self.engine = engine

    async def generate_stream(
        self, prompt: str, max_tokens: int = 100, **kwargs: Any
    ) -> AsyncIterator[str]:
        record = await self.engine.submit(
            prompt,
            max_tokens=max_tokens,
            temperature=float(kwargs.get("temperature", 0.7)),
        )
        async for chunk in self.engine.stream_request(record.id):
            yield chunk

    async def complete_text(self, prompt: str, max_tokens: int = 100, **kwargs: Any) -> str:
        parts: list[str] = []
        async for chunk in self.generate_stream(prompt, max_tokens, **kwargs):
            parts.append(chunk)
        return "".join(parts)
