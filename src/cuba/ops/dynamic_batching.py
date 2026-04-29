"""Time- and size-bounded dynamic micro-batching for CPU (async, future-based)."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")

ProcessBatch = Callable[[list[T]], Awaitable[Sequence[R]]]


class CPUDynamicBatching(Generic[T, R]):
    """
    Buffers requests and invokes ``process_batch`` when either
    ``max_batch_size`` is reached or ``max_wait_time`` elapses.
    Each ``add_to_batch`` / ``submit`` call waits for that item’s result.
    """

    def __init__(
        self,
        process_batch: ProcessBatch[T, R],
        max_batch_size: int = 16,
        max_wait_time: float = 0.1,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_wait_time < 0:
            raise ValueError("max_wait_time must be non-negative")
        self._process_batch_fn = process_batch
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self._lock = asyncio.Lock()
        self._queue: list[tuple[T, asyncio.Future[R]]] = []
        self._flush_task: asyncio.Task[None] | None = None
        self.last_batch_time: float = time.time()

    async def submit(self, request: T) -> R:
        return await self.add_to_batch(request)

    async def add_to_batch(self, request: T) -> R:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[R] = loop.create_future()
        async with self._lock:
            self._queue.append((request, fut))
            if len(self._queue) >= self.max_batch_size:
                await self._drain_unlocked()
            elif self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._delayed_flush())
        return await fut

    async def _delayed_flush(self) -> None:
        try:
            await asyncio.sleep(self.max_wait_time)
            async with self._lock:
                await self._drain_unlocked()
        except asyncio.CancelledError:
            return

    async def _drain_unlocked(self) -> None:
        if not self._queue:
            return
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = None

        batch = self._queue
        self._queue = []
        self.last_batch_time = time.time()

        items = [b[0] for b in batch]
        futs: list[asyncio.Future[R]] = [b[1] for b in batch]
        try:
            results = await self._process_batch_fn(items)
            if len(results) != len(futs):
                raise RuntimeError(
                    f"process_batch returned {len(results)} results for {len(futs)} requests"
                )
            for fut, r in zip(futs, results, strict=True):
                if not fut.done():
                    fut.set_result(r)
        except Exception as e:  # noqa: BLE001
            for fut in futs:
                if not fut.done():
                    fut.set_exception(e)

    async def flush_queue(self) -> None:
        """
        Drain the buffer immediately (does not wait for ``max_wait_time``).
        """
        async with self._lock:
            await self._drain_unlocked()
