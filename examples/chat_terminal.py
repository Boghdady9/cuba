"""
Interactive terminal chat against a Cuba OpenAI-compatible server.

  uv run python examples/chat_terminal.py

If ``CUBA_CHAT_URL`` is not set, the client probes ``/health`` on (by default)::

  http://127.0.0.1:8000  then  http://127.0.0.1:8080

Whichever answers first is used; 8000 matches ``cli_openai.py`` default, 8080 matches
a typical ``kubectl port-forward ... 8080:80``.

Set explicitly when needed::

  CUBA_CHAT_URL=http://127.0.0.1:9000/v1/chat/completions
  CUBA_MODEL=my-model
"""
from __future__ import annotations

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
        f"If the API is on a different port, set a full URL (or expand discovery):\n"
        f"  CUBA_CHAT_URL=http://{_DISCOVERY_HOST}:<port>/v1/chat/completions "
        f"uv run python examples/chat_terminal.py\n"
        f"  CUBA_CHAT_DISCOVERY_PORTS=8000,8080,9000   # try more ports in order\n"
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
    """Return API base (scheme://host:port) that responds to GET /health with 2xx."""
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
    """Return (chat_completions_url, model_id)."""
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
            print(
                f"{base}/health returned HTTP {h.status_code}. Fix the server or URL, then retry.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return explicit, _resolve_model(client, base, prefer_env=True)

    base = _discover_base()
    if not base:
        print(_discovery_fail_text(), file=sys.stderr)
        raise SystemExit(1)

    url = f"{base}/v1/chat/completions"
    return url, _resolve_model(client, base, prefer_env=True)


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


def main() -> None:
    messages: list[dict[str, str]] = []
    headers = {"Authorization": f"Bearer {_API_KEY}"} if _API_KEY else {}

    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        url, model = _resolve_endpoint(client)
        base = _chat_to_base(url)
        parsed = urlparse(base)
        port = parsed.port or (80 if parsed.scheme == "http" else 443)
        print(f"Using {url}")
        if not os.environ.get("CUBA_CHAT_URL", "").strip():
            print(f"  (auto-discovered via /health on port {port} among {_CANDIDATE_PORTS})")
        print(f"model={model}\nType a line, Enter to send. 'q' to exit.\n")

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
            try:
                r = client.post(
                    url,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": _MAX_TOKENS,
                        "temperature": _TEMPERATURE,
                        "stream": False,
                    },
                )
            except httpx.ConnectError as e:
                print(f"ConnectError: {e}\n\n{_discovery_fail_text()}\n", file=sys.stderr)
                messages.pop()
                continue
            except httpx.RequestError as e:
                print(f"Request failed: {e}\n", file=sys.stderr)
                messages.pop()
                continue

            if r.is_error:
                print(r.status_code, r.text, file=sys.stderr)
                messages.pop()
                continue

            try:
                data = r.json()
            except ValueError:
                print("Invalid JSON:", r.text[:500], file=sys.stderr)
                messages.pop()
                continue

            try:
                text = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                print("Bad response:", data, file=sys.stderr)
                messages.pop()
                continue

            print(f"{text}\n")
            messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()
