#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def run_once(
    client: httpx.AsyncClient,
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
) -> float:
    t0 = time.perf_counter()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for _ in resp.aiter_text():
                pass
    else:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
    return time.perf_counter() - t0


async def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Cuba OpenAI benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="cuba")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--prompt", default="Summarize the scheduler.")
    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    limits = httpx.Limits(max_keepalive_connections=args.concurrency, max_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=60, limits=limits) as client:
        sem = asyncio.Semaphore(args.concurrency)

        async def worker() -> float:
            async with sem:
                return await run_once(
                    client,
                    url=url,
                    model=args.model,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    stream=args.stream,
                )

        started = time.perf_counter()
        durations = await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(args.requests)])
        elapsed = time.perf_counter() - started

    p95_index = max(0, min(len(durations) - 1, round(0.95 * (len(durations) - 1))))
    sorted_durations = sorted(durations)
    print(f"requests={args.requests}")
    print(f"concurrency={args.concurrency}")
    print(f"stream={args.stream}")
    print(f"total_elapsed_s={elapsed:.4f}")
    print(f"throughput_rps={args.requests / elapsed:.2f}")
    print(f"latency_avg_s={statistics.mean(durations):.4f}")
    print(f"latency_p95_s={sorted_durations[p95_index]:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
