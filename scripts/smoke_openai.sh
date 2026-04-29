#!/usr/bin/env bash
# Sync deps, run API tests, then (if a server is up) hit /health, /ready, /v1/models, and one chat.
# With the OpenAI server already running (e.g. cli_openai on 8000)::
#   ./scripts/smoke_openai.sh
# Override base URL::
#   CUBA_SMOKE_BASE=http://127.0.0.1:9000 ./scripts/smoke_openai.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BASE="${CUBA_SMOKE_BASE:-http://127.0.0.1:8000}"

echo "== uv sync =="
uv sync -q
echo "== pytest tests/ =="
uv run pytest tests/ -q
echo

if ! curl -sfS "$BASE/health" -o /dev/null; then
  echo "No server at $BASE (GET /health failed). Start the API, then re-run, e.g.:"
  echo "  uv run python cli_openai.py --host 127.0.0.1 --port 8000 --backend engine --model-path google/gemma-3-270m"
  exit 0
fi

echo "== server $BASE =="
curl -sfS "$BASE/health" | head -c 300
echo
echo
curl -sfS "$BASE/ready" | head -c 300
echo
echo

export SMOKE_BASE="$BASE"
echo "== chat completion =="
uv run python <<'PY'
import os
import httpx

base = os.environ["SMOKE_BASE"]
r = httpx.get(f"{base}/v1/models", timeout=10.0)
r.raise_for_status()
mid = r.json()["data"][0]["id"]
p = httpx.post(
    f"{base}/v1/chat/completions",
    json={
        "model": mid,
        "messages": [{"role": "user", "content": "Reply with one short sentence."}],
        "max_tokens": 64,
        "stream": False,
    },
    timeout=300.0,
)
p.raise_for_status()
print(p.json()["choices"][0]["message"]["content"])
PY
echo
echo "== chat_terminal (quit immediately) =="
printf 'q\n' | uv run python examples/chat_terminal.py 2>&1 | head -12
echo
echo "Done. Interactive REPL: uv run python examples/chat_terminal.py"
