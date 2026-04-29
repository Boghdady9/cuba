from __future__ import annotations

import asyncio

import pytest


@pytest.mark.anyio
async def test_engine_completes_request(tiny_runtime: object) -> None:
    runtime = tiny_runtime
    assert runtime.engine is not None
    await runtime.engine.start()
    try:
        record = await runtime.engine.submit("hello", max_tokens=4, temperature=0)
        text = await runtime.engine.wait_for_completion(record.id)
        assert isinstance(text, str)
        assert len(text) == 4
        completed = runtime.engine.get_request(record.id)
        assert completed is not None
        assert completed.generated_tokens == 4
    finally:
        await runtime.engine.stop()


@pytest.mark.anyio
async def test_engine_streams_tokens(tiny_runtime: object) -> None:
    runtime = tiny_runtime
    assert runtime.engine is not None
    await runtime.engine.start()
    try:
        record = await runtime.engine.submit("stream me", max_tokens=3, temperature=0)
        parts: list[str] = []
        async for chunk in runtime.engine.stream_request(record.id):
            parts.append(chunk)
        assert len(parts) == 3
        assert await runtime.engine.wait_for_completion(record.id) == "".join(parts)
    finally:
        await runtime.engine.stop()


@pytest.mark.anyio
async def test_engine_handles_concurrency(tiny_runtime: object) -> None:
    runtime = tiny_runtime
    assert runtime.engine is not None
    await runtime.engine.start()
    try:
        tasks = [
            asyncio.create_task(runtime.backend.complete_chat(f"prompt {i}", max_tokens=2, temperature=0))
            for i in range(6)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 6
        assert all(len(result) == 2 for result in results)
        stats = runtime.engine.scheduler_stats
        assert int(stats["iterations"]) > 0
    finally:
        await runtime.engine.stop()
