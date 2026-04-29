"""
OpenAI Chat Completions-compatible HTTP API (FastAPI).
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from cuba.metrics import MetricsCollector
from cuba.openai.backend import OpenAIInferenceBackend, StubOpenAIBackend, messages_to_prompt

__all__ = [
    "create_app",
    "ChatCompletionRequest",
    "ChatMessage",
    "StubOpenAIBackend",
    "default_app",
]


def _rough_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=100, ge=1)
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool | None = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


def create_app(
    backend: OpenAIInferenceBackend,
    *,
    title: str = "Cuba OpenAI API",
    model_id: str = "cuba",
    model_path: str | None = None,
    max_batch_size: int | None = None,
    metrics: MetricsCollector | None = None,
    runtime_info: dict[str, object] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await backend.start()
        yield
        await backend.stop()

    app = FastAPI(title=title, version="1.0.0", lifespan=lifespan)
    app.state.backend = backend
    app.state.model_id = model_id
    app.state.model_path = model_path
    app.state.max_batch_size = max_batch_size
    app.state.metrics = metrics
    app.state.runtime_info = runtime_info or {}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        backend_health = backend.health()
        out: dict[str, Any] = {
            "status": "healthy" if backend_health.get("ready") else "degraded",
            "engine": "cuba",
            "model": str(app.state.model_id),
            "backend": backend_health,
            "runtime": app.state.runtime_info,
        }
        if app.state.model_path is not None:
            out["model_path"] = str(app.state.model_path)
        if app.state.max_batch_size is not None:
            out["max_batch_size"] = int(app.state.max_batch_size)
        out["metrics"] = app.state.metrics is not None
        return out

    @app.get("/ready")
    async def readiness() -> dict[str, Any]:
        backend_health = backend.health()
        if not backend_health.get("ready"):
            raise HTTPException(status_code=503, detail="backend not ready")
        return {"ready": True, "backend": backend_health}

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        m: MetricsCollector | None = app.state.metrics
        if m is None:
            raise HTTPException(status_code=404, detail="Metrics not enabled for this app")
        return Response(
            content=m.get_prometheus_metrics(),
            media_type=MetricsCollector.content_type(),
        )

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        mid = str(app.state.model_id)
        return {
            "object": "list",
            "data": [
                {
                    "id": mid,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "cuba",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest) -> Any:
        backend_o: OpenAIInferenceBackend = app.state.backend
        msg_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
        tokenizer = getattr(backend_o, "tokenizer", None)
        prompt = messages_to_prompt(msg_dicts, tokenizer=tokenizer)
        model = request.model
        max_t = int(request.max_tokens or 100)
        temp = float(request.temperature if request.temperature is not None else 0.7)
        metrics_c: MetricsCollector | None = app.state.metrics
        t0 = time.perf_counter()

        if request.stream:
            async def event_stream() -> AsyncIterator[str]:
                rid = f"chatcmpl-{uuid.uuid4()}"
                created = int(time.time())
                n_chars = 0
                first_chunk_at: float | None = None
                try:
                    start_payload = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(start_payload, ensure_ascii=False)}\n\n"
                    async for chunk in backend_o.stream_chat(prompt, max_tokens=max_t, temperature=temp):
                        if first_chunk_at is None:
                            first_chunk_at = time.perf_counter()
                        n_chars += len(chunk)
                        payload = {
                            "id": rid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": chunk}, "logprobs": None, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    final = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:  # noqa: BLE001
                    err = {"error": {"message": str(exc), "type": "internal_error"}}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                finally:
                    if metrics_c is not None:
                        elapsed = time.perf_counter() - t0
                        metrics_c.record_request(
                            elapsed,
                            max(0, n_chars // 4),
                            batch_size=1,
                            prompt_tokens=_rough_tokens(prompt),
                            ttft_seconds=(first_chunk_at - t0) if first_chunk_at is not None else None,
                        )

            return StreamingResponse(event_stream(), media_type="text/event-stream; charset=utf-8")

        try:
            text = await backend_o.complete_chat(prompt, max_tokens=max_t, temperature=temp)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        pt = _rough_tokens(prompt)
        ct = _rough_tokens(text)
        if metrics_c is not None:
            elapsed = time.perf_counter() - t0
            metrics_c.record_request(
                elapsed,
                ct,
                batch_size=1,
                prompt_tokens=pt,
                ttft_seconds=elapsed,
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=model,
            choices=[{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            usage={"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        )

    return app


def default_app() -> FastAPI:
    return create_app(StubOpenAIBackend())
