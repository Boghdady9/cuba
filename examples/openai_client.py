"""
Use the official OpenAI Python SDK against a local Cuba server (same contract as OpenAI).

Install::

    uv sync --extra client
    # or: pip install "openai"

Run the server in another shell::

    uv run python cli_openai.py --host 127.0.0.1 --port 8000 --model-path google/gemma-3-270m

Imports use the ``cuba`` package (``src/cuba/``); install with ``uv sync`` for ``import cuba``.

Then::

    python examples/openai_client.py
"""
from __future__ import annotations

import os
import sys

# Must match a model id returned by ``GET /v1/models`` (default: last segment of --model-path).
DEFAULT_MODEL = os.environ.get("CUBA_EXAMPLE_MODEL", "gemma-3-270m")
BASE = os.environ.get("CUBA_OPENAI_BASE", "http://127.0.0.1:8000/v1")


def main() -> None:
    try:
        from openai import OpenAI
    except ImportError as e:
        print("Install the client: uv sync --extra client  (or pip install openai)", file=sys.stderr)
        raise e

    client = OpenAI(
        base_url=BASE,
        api_key=os.environ.get("CUBA_OPENAI_API_KEY", "dummy"),
    )

    # 2) Chat completion (non-streaming)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "user", "content": "Tell me a joke about programming"},
        ],
        max_tokens=100,
        temperature=0.7,
    )
    if response.choices[0].message.content is not None:
        print("--- non-streaming ---\n", response.choices[0].message.content)

    # 3) Streaming
    print("\n--- streaming ---\n", end="")
    stream = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "user", "content": "Write a short story"},
        ],
        max_tokens=200,
        stream=True,
    )
    for chunk in stream:
        c = chunk.choices[0].delta
        if c and c.content:
            print(c.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
