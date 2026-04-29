"""Load distribution across multiple :class:`ContinuousBatchingEngine` instances."""
from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

from cuba.engine.batching import ContinuousBatchingEngine


class LoadBalanceStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"


class LoadBalancer:
    """
    Distribute work across several ContinuousBatchingEngine instances.
    Use ``least_loaded`` to favor engines with the smallest (running + pending) queue.
    """

    def __init__(
        self,
        engines: Sequence[ContinuousBatchingEngine],
        strategy: LoadBalanceStrategy | str = LoadBalanceStrategy.ROUND_ROBIN,
    ) -> None:
        if not engines:
            raise ValueError("LoadBalancer requires at least one engine")
        self.engines: list[ContinuousBatchingEngine] = list(engines)
        s = (
            strategy
            if isinstance(strategy, LoadBalanceStrategy)
            else LoadBalanceStrategy(str(strategy))
        )
        self._strategy: LoadBalanceStrategy = s
        self._current = 0

    def _pick(self) -> ContinuousBatchingEngine:
        if self._strategy is LoadBalanceStrategy.LEAST_LOADED:
            return min(
                self.engines,
                key=lambda e: len(e.running_sequences) + len(e.pending_requests),
            )
        e = self.engines[self._current]
        self._current = (self._current + 1) % len(self.engines)
        return e

    async def route_request(self, prompt: str, **kwargs: Any):
        return await self._pick().add_request(prompt, **kwargs)

    def get_engine_status(self) -> list[dict[str, int]]:
        return [
            {
                "engine_id": i,
                "active_requests": len(engine.running_sequences),
                "pending_requests": len(engine.pending_requests),
            }
            for i, engine in enumerate(self.engines)
        ]
