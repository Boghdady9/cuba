"""
Interactive terminal chat against a Cuba OpenAI-compatible server (streaming).

  uv run python examples/chat_terminal.py

If ``CUBA_CHAT_URL`` is not set, the client probes ``/health`` on (by default)::

  http://127.0.0.1:8000  then  http://127.0.0.1:8080

Set explicitly when needed::

  CUBA_CHAT_URL=http://127.0.0.1:9000/v1/chat/completions
  CUBA_MODEL=my-model
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any
from urllib.parse import urlparse

import httpx

_EXPLICIT_FAIL = (
    "If the URL is correct, start the process or port-forward, then retry.\n"
    "  uv run python cli_openai.py --host 127.0.0.1 --port 8000"
)


def _discovery_fail_text() -> str:
    ports = ", ".join(_CANDIDATE_PORTS)
    return (
        f"No server responded to GET /health (2xx) on host {_DISCOVERY_HOST}, ports [{ports}].\n"
        "\n"
        "Start the Cuba API in another terminal, then run this client again:\n"
        f"  uv run python cli_openai.py --host {_DISCOVERY_HOST} --port 8000\n"
        "\n"
        "To reach the cluster (after pods are up):\n"
        "  kubectl -n cuba port-forward svc/cuba-api 8080:80\n"
        "\n"
        f"If the API is on a different port, set a full URL:\n"
        f"  CUBA_CHAT_URL=http://{_DISCOVERY_HOST}:<port>/v1/chat/completions "
        f"uv run python examples/chat_terminal.py\n"
    )


_CANDIDATE_PORTS = [
    p.strip() for p in os.environ.get("CUBA_CHAT_DISCOVERY_PORTS", "8000,8080").split(",") if p.strip()
]
_DISCOVERY_HOST = os.environ.get("CUBA_CHAT_DISCOVERY_HOST", "127.0.0.1")
_MAX_TOKENS = int(os.environ.get("CUBA_MAX_TOKENS", "256"))
_TEMPERATURE = float(os.environ.get("CUBA_TEMPERATURE", "0.7"))
_API_KEY = os.environ.get("CUBA_OPENAI_API_KEY", "dummy")


def _chat_to_base(url: str) -> str:
    if "/v1/" in url:
        return url.split("/v1/", 1)[0].rstrip("/")
    u = url.rsplit("/", 1)
    return u[0] if len(u) > 1 else url


def _discover_base() -> str:
    for port in _CANDIDATE_PORTS:
        base = f"http://{_DISCOVERY_HOST}:{port}"
        try:
            r = httpx.get(f"{base}/health", timeout=2.0)
        except httpx.RequestError:
            continue
        if r.is_success:
            return base
    return ""


def _resolve_endpoint(client: httpx.Client) -> tuple[str, str]:
    explicit = os.environ.get("CUBA_CHAT_URL", "").strip()
    if explicit:
        base = _chat_to_base(explicit)
        try:
            h = client.get(f"{base}/health", timeout=3.0)
        except httpx.RequestError as e:
            print(f"Cannot reach {base}/health: {e}\n", file=sys.stderr)
            print(_EXPLICIT_FAIL, file=sys.stderr)
            raise SystemExit(1) from e
        if not h.is_success:
            print(f"{base}/health returned HTTP {h.status_code}.", file=sys.stderr)
            raise SystemExit(1)
        return explicit, _resolve_model(client, base, prefer_env=True)

    base = _discover_base()
    if not base:
        print(_discovery_fail_text(), file=sys.stderr)
        raise SystemExit(1)

    return f"{base}/v1/chat/completions", _resolve_model(client, base, prefer_env=True)


def _resolve_model(client: httpx.Client, base: str, *, prefer_env: bool) -> str:
    if prefer_env and os.environ.get("CUBA_MODEL", "").strip():
        return os.environ["CUBA_MODEL"].strip()
    try:
        r = client.get(f"{base}/v1/models", timeout=3.0)
        if r.is_success:
            data: Any = r.json()
            items = data.get("data") or []
            if items and isinstance(items, list) and "id" in items[0]:
                return str(items[0]["id"])
    except (httpx.RequestError, TypeError, KeyError, ValueError):
        pass
    return os.environ.get("CUBA_MODEL", "cuba").strip() or "cuba"


def _stream_response(client: httpx.Client, url: str, headers: dict[str, str], payload: dict[str, Any]) -> str:
    """Stream SSE response, printing tokens as they arrive. Returns full assistant text.

    Suppresses <think>...</think> blocks (Qwen3 chain-of-thought) from the display.
    The full text including think blocks is kept in conversation history.
    """
    collected: list[str] = []
    buf = ""          # rolling buffer for tag boundary detection
    in_think = False  # currently inside a <think> block
    OPEN, CLOSE = "<think>", "</think>"

    with client.stream("POST", url, headers=headers, json=payload, timeout=300.0) as resp:
        if resp.is_error:
            body = resp.read().decode()
            print(f"\nHTTP {resp.status_code}: {body}", file=sys.stderr)
            return ""
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw.strip() == "[DONE]":
                break
            try:
                text = json.loads(raw)["choices"][0]["delta"].get("content") or ""
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if not text:
                continue
            collected.append(text)
            buf += text

            # Drain buffer: emit visible text, drop think blocks.
            visible: list[str] = []
            while buf:
                if not in_think:
                    idx = buf.find(OPEN)
                    if idx == -1:
                        # No opening tag in buffer — safe to flush all but last few chars
                        # (guards against a tag split across two chunks)
                        safe_end = max(0, len(buf) - len(OPEN) + 1)
                        visible.append(buf[:safe_end])
                        buf = buf[safe_end:]
                        break
                    visible.append(buf[:idx])
                    buf = buf[idx + len(OPEN):]
                    in_think = True
                else:
                    idx = buf.find(CLOSE)
                    if idx == -1:
                        buf = ""  # discard buffered think content
                        break
                    buf = buf[idx + len(CLOSE):].lstrip("\n")
                    in_think = False

            text_out = "".join(visible).lstrip("\n") if len(collected) <= 1 else "".join(visible)
            if text_out:
                print(text_out, end="", flush=True)

    if buf and not in_think:
        print(buf, end="", flush=True)
    print()
    return "".join(collected)


def main() -> None:
    messages: list[dict[str, str]] = []
    headers = {"Authorization": f"Bearer {_API_KEY}"} if _API_KEY else {}

    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        url, model = _resolve_endpoint(client)
        base = _chat_to_base(url)
        parsed = urlparse(base)
        port = parsed.port or (80 if parsed.scheme == "http" else 443)
        print(f"Connected to {url}")
        if not os.environ.get("CUBA_CHAT_URL", "").strip():
            print(f"  (auto-discovered on port {port})")
        print(f"model={model}  max_tokens={_MAX_TOKENS}  streaming=on")
        print("Type a message and press Enter. 'q' to quit.\n")

        while True:
            try:
                line = input("> ")
            except EOFError:
                print()
                break
            user = line.strip()
            if not user:
                continue
            if user.lower() == "q":
                break

            messages.append({"role": "user", "content": user})
            print(" ", end="", flush=True)  # indent assistant reply
            try:
                reply = _stream_response(
                    client, url, headers,
                    {
                        "model": model,
                        "messages": messages,
                        "max_tokens": _MAX_TOKENS,
                        "temperature": _TEMPERATURE,
                        "stream": True,
                    },
                )
            except httpx.ConnectError as e:
                print(f"\nConnectError: {e}", file=sys.stderr)
                messages.pop()
                continue
            except httpx.RequestError as e:
                print(f"\nRequest failed: {e}", file=sys.stderr)
                messages.pop()
                continue

            if reply:
                messages.append({"role": "assistant", "content": reply})
            else:
                messages.pop()
            print()


if __name__ == "__main__":
    main()
