# Cuba

> A custom LLM inference engine with continuous batching, KV cache, and an OpenAI-compatible API — built from scratch in Python.

---

## Overview

Cuba is a production-grade inference server for large language models. It implements the core scheduling techniques used by systems like vLLM — continuous batching, per-request KV caching, and phase-separated prefill/decode — on top of HuggingFace Transformers, exposing a drop-in OpenAI-compatible REST API.

**Supports:** CPU · CUDA · MPS · MLX (Apple Silicon)

---

## How It Works

```
Client Request  →  POST /v1/chat/completions
                        │
              ┌─────────▼──────────┐
              │  ContinuousBatching │
              │      Engine         │
              │  ┌───────────────┐  │
              │  │    Prefill    │  │  ← full prompt, build KV cache
              │  │    Decode     │  │  ← single token per step, O(1)
              │  └───────────────┘  │
              └─────────┬──────────┘
                        │
              JSON / SSE stream
```

**Key design decisions:**

- **Decode-first scheduling** — active generations are never stalled by new prefills
- **KV cache per request** — decode cost is O(1) per token, not O(n²) re-forward
- **Phase-separated batches** — prefill and decode never share a batch, eliminating padding waste
- **Dynamic INT8 quantization** — applied automatically on CPU for 2–3× throughput gain

---

## Benchmarks

Measured on **Intel MacBook Pro (x86_64, 8-core)**, model: `Qwen/Qwen3-0.6B`, single request, CPU-only.

| Engine | Quantization | tok/s | Notes |
|--------|-------------|------:|-------|
| llama.cpp | GGML INT8 (GGUF) | **37.6** | Hand-written AVX2/SIMD C++ kernels |
| **Cuba** | PyTorch INT8 | **~15** | KV cache + dynamic quantization |
| vLLM | — | — | CPU not supported; GPU-only |

**Why llama.cpp is faster on single-request CPU:** it uses hand-optimised AVX2/AVX512 SIMD kernels in C++. Cuba runs on PyTorch INT8, which carries Python overhead.

**Where Cuba is the better choice:**
| | Cuba | llama.cpp |
|--|------|-----------|
| OpenAI-compatible API | ✅ | ❌ |
| Continuous batching (multi-user) | ✅ | ❌ |
| Any HuggingFace model (no conversion) | ✅ | ❌ |
| CUDA / MPS / MLX | ✅ | Partial |
| Raw single-request CPU throughput | — | ✅ |

---

## Quick Start

### Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager

### Install

```bash
git clone https://github.com/Boghdady9/cuba.git
cd cuba
uv sync
```

### Start the server

```bash
# Recommended: Qwen3-0.6B (600M params, reasoning model)
CUBA_MODEL_PATH=Qwen/Qwen3-0.6B uv run python cli_openai.py --host 127.0.0.1 --port 8000

# Lightweight: GPT-2 (no auth required, fast)
CUBA_MODEL_PATH=gpt2 uv run python cli_openai.py --host 127.0.0.1 --port 8000

# Specific device
CUBA_MODEL_PATH=Qwen/Qwen3-0.6B uv run python cli_openai.py --device cuda
CUBA_MODEL_PATH=Qwen/Qwen3-0.6B uv run python cli_openai.py --device mps
CUBA_MODEL_PATH=Qwen/Qwen3-0.6B uv run python cli_openai.py --device mlx  # Apple Silicon only
```

The server auto-detects the best available device: **CUDA → MPS → CPU**.

### Interactive terminal chat (streaming)

```bash
# In a second terminal (server must be running)
CUBA_MAX_TOKENS=512 uv run python examples/chat_terminal.py
```

Tokens stream in real time. Qwen3 chain-of-thought `<think>` blocks are filtered automatically.

---

## API

Cuba exposes an OpenAI-compatible API. Any client that works with the OpenAI SDK works with Cuba.

### Health check

```bash
curl http://127.0.0.1:8000/health | jq
```

### Chat completion

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "max_tokens": 256,
    "temperature": 0.7
  }' | jq '.choices[0].message.content'
```

### Streaming

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Tell me a story"}],"max_tokens":256,"stream":true}' \
  -N
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://127.0.0.1:8000/v1")

response = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "Explain KV caching in one paragraph."}],
    max_tokens=256,
)

print(response.choices[0].message.content)
```

---

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUBA_MODEL_PATH` | required | HuggingFace model ID or local path |
| `CUBA_DEVICE` | `auto` | Device: `auto` · `cpu` · `cuda` · `mps` · `mlx` |
| `CUBA_BACKEND` | `engine` | `engine` for real inference, `stub` for smoke tests |
| `CUBA_MAX_BATCH_SIZE` | `16` | Max requests per batch |
| `CUBA_MAX_CONCURRENT_SEQUENCES` | `32` | Max in-flight requests |
| `CUBA_MAX_WAIT_MS` | `10` | Max wait before dispatching a batch (ms) |
| `CUBA_SCHEDULER_TICK_MS` | `5` | Scheduler loop interval (ms) |
| `CUBA_MAX_TOKENS` | `256` | Default generation length |
| `CUBA_TEMPERATURE` | `0.7` | Default sampling temperature |
| `HF_TOKEN` | — | HuggingFace token for gated models |

### CLI flags

```
--host               Bind address (default: 0.0.0.0)
--port               Port (default: 8000)
--device             auto | cpu | cuda | mps | mlx
--model-path         HuggingFace model ID
--max-batch-size     Max requests per batch
--max-concurrent-sequences
--max-wait-ms
--scheduler-tick-ms
```

---

## Supported Models

Any `AutoModelForCausalLM`-compatible model from HuggingFace works. Tested:

| Model | Params | CPU tok/s | Notes |
|-------|--------|----------:|-------|
| `Qwen/Qwen3-0.6B` | 600M | ~15 | Recommended — reasoning model, chat template |
| `HuggingFaceTB/SmolLM-360M` | 360M | ~25 | Fast, no auth required |
| `gpt2` | 124M | ~28 | Baseline, no auth required |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | ~10 | Instruction-tuned |
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | ~2 | High quality, needs more RAM |

For Apple Silicon (MLX backend), install `mlx-lm`:
```bash
uv sync --extra mlx
CUBA_MODEL_PATH=Qwen/Qwen3-0.6B uv run python cli_openai.py --device mlx
```

---

## Deployment

### Docker

```bash
docker build -t cuba:latest .

docker run -p 8000:8000 \
  -e CUBA_MODEL_PATH=Qwen/Qwen3-0.6B \
  cuba:latest
```

### Kubernetes

```bash
kubectl create namespace cuba
kubectl apply -f k8s/configmap.yaml   # set CUBA_MODEL_PATH here
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl -n cuba port-forward svc/cuba-api 8000:80
curl http://127.0.0.1:8000/health
```

---

## Project Structure

```
cuba/
├── src/cuba/
│   ├── engine/
│   │   └── batching.py       # ContinuousBatchingEngine — KV cache, scheduler, sampler
│   ├── openai/
│   │   ├── api.py            # FastAPI — /v1/chat/completions, /health, /metrics
│   │   └── backend.py        # PyTorch + MLX backend implementations
│   ├── runtime/
│   │   └── bootstrap.py      # Device detection, model loading, quantization
│   ├── ops/                  # CPU thread optimisation
│   └── metrics/              # Prometheus-compatible metrics
├── tests/                    # 7 tests — engine, API, streaming, concurrency, metrics
├── examples/
│   ├── chat_terminal.py      # Streaming terminal client
│   └── openai_client.py      # Python SDK example
├── k8s/                      # Kubernetes manifests
├── Dockerfile
└── cli_openai.py             # Server entrypoint
```

---

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run python -m pytest tests/ -v

# Type check
uv run mypy src/cuba/
```

---

## References

- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — inspiration for KV cache design
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## License

MIT
