from __future__ import annotations

from fastapi.testclient import TestClient

from cuba.openai.api import create_app
from cuba.runtime import RuntimeSettings, build_runtime


def test_stub_app_health() -> None:
    runtime = build_runtime(RuntimeSettings(backend_mode="stub"))
    with TestClient(
        create_app(
            runtime.backend,
            model_id="cuba",
            metrics=runtime.metrics,
            runtime_info=runtime.runtime_info,
        )
    ) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"]["backend"] == "stub"


def test_engine_completion_endpoint(tiny_runtime: object) -> None:
    runtime = tiny_runtime
    with TestClient(
        create_app(
            runtime.backend,
            model_id="cuba",
            metrics=runtime.metrics,
            runtime_info=runtime.runtime_info,
        )
    ) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cuba",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 3,
                "temperature": 0,
                "stream": False,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "cuba"
        assert len(body["choices"][0]["message"]["content"]) == 3


def test_engine_streaming_endpoint(tiny_runtime: object) -> None:
    runtime = tiny_runtime
    with TestClient(
        create_app(
            runtime.backend,
            model_id="cuba",
            metrics=runtime.metrics,
            runtime_info=runtime.runtime_info,
        )
    ) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "cuba",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
                "temperature": 0,
                "stream": True,
            },
        ) as resp:
            text = "".join(resp.iter_text())
        assert resp.status_code == 200
        assert "data: [DONE]" in text
