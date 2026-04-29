#!/usr/bin/env python3
"""
Start the OpenAI-compatible API server (FastAPI + uvicorn).

Example::

    PYTHONPATH=src uv run python cli_openai.py --host 0.0.0.0 --port 8000
    # or from repo root with ``uv run python cli_openai.py`` if PYTHONPATH is set in the environment
"""
from __future__ import annotations

import argparse
import errno
import os
import sys


def main() -> None:
    # Prepend `src` so ``import cuba`` works without an editable install (optional).
    root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(root, "src")
    if src not in sys.path and os.path.isdir(src):
        sys.path.insert(0, src)
    p = argparse.ArgumentParser(description="Cuba OpenAI-compatible API server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--backend",
        default=os.environ.get("CUBA_BACKEND", "engine"),
        choices=["engine", "stub"],
        help="Serving backend. Defaults to engine; use stub for smoke tests.",
    )
    p.add_argument(
        "--model-name",
        default=os.environ.get("CUBA_MODEL_NAME", "cuba"),
        help="Model id in /v1/models and completion payloads (default: cuba, or CUBA_MODEL_NAME)",
    )
    p.add_argument(
        "--model-path",
        default=(
            os.environ.get("CUBA_MODEL_PATH")
            or os.environ.get("MODEL_PATH")
        ),
        help=(
            "HuggingFace-style id (e.g. google/gemma-3-270m). Env: CUBA_MODEL_PATH or MODEL_PATH. "
            "If set, the public model id defaults to the last path segment unless CUBA_MODEL_NAME / --model-name override."
        ),
    )
    p.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="For operators / health (stub uses it in /health). Env: MAX_BATCH_SIZE or CUBA_MAX_BATCH_SIZE.",
    )
    p.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn autoreload (dev only)",
    )
    p.add_argument("--max-wait-ms", type=int, default=int(os.environ.get("CUBA_MAX_WAIT_MS", "10")))
    p.add_argument(
        "--max-concurrent-sequences",
        type=int,
        default=int(os.environ.get("CUBA_MAX_CONCURRENT_SEQUENCES", "32")),
    )
    p.add_argument(
        "--scheduler-tick-ms",
        type=int,
        default=int(os.environ.get("CUBA_SCHEDULER_TICK_MS", "5")),
    )
    p.add_argument("--torch-num-threads", type=int, default=None)
    p.add_argument("--torch-interop-threads", type=int, default=None)
    args = p.parse_args()

    model_id = str(args.model_name)
    if args.model_path and "CUBA_MODEL_NAME" not in os.environ and model_id == "cuba":
        tail = str(args.model_path).rstrip("/").split("/")[-1] or "cuba"
        model_id = tail

    max_bs = args.max_batch_size
    if max_bs is None:
        for key in ("MAX_BATCH_SIZE", "CUBA_MAX_BATCH_SIZE"):
            raw = os.environ.get(key)
            if raw is not None and raw.strip() != "":
                try:
                    max_bs = int(raw)
                except ValueError:
                    pass
                break

    import uvicorn
    from cuba.openai.api import create_app
    from cuba.runtime import RuntimeSettings, build_runtime

    runtime = build_runtime(
        RuntimeSettings(
            backend_mode=str(args.backend),
            model_id=model_id,
            model_path=str(args.model_path) if args.model_path else None,
            max_batch_size=max_bs or 16,
            max_wait_ms=int(args.max_wait_ms),
            max_concurrent_sequences=int(args.max_concurrent_sequences),
            scheduler_tick_ms=int(args.scheduler_tick_ms),
            num_threads=args.torch_num_threads,
            interop_threads=args.torch_interop_threads,
        )
    )
    app = create_app(
        runtime.backend,
        model_id=model_id,
        model_path=str(args.model_path) if args.model_path else None,
        max_batch_size=max_bs,
        metrics=runtime.metrics,
        runtime_info=runtime.runtime_info,
    )
    def _addr_in_use(exc: OSError) -> bool:
        if exc.errno == errno.EADDRINUSE:
            return True
        if sys.platform == "win32" and getattr(exc, "winerror", None) == 10048:
            return True
        return "address already in use" in str(exc).lower()

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except OSError as e:
        if _addr_in_use(e):
            print(
                f"Port {args.port} is already in use on {args.host}.\n"
                f"  Use another port:  --port 8001\n"
                f"  Or see what is listening:  lsof -nP -iTCP:{args.port} -sTCP:LISTEN\n"
                f"  Then stop that process (e.g. old cli_openai) or:  kill <PID>",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        raise


if __name__ == "__main__":
    main()
