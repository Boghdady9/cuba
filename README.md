# Cuba: Continuous Batching Engine with OpenAI-Compatible API

A high-performance inference engine for serving large language models with continuous batching, dynamic scheduling, and OpenAI-compatible chat completions API.

**Status**: Production-ready ✅ | Tests: 7/7 passing | Devices: CPU · CUDA · MPS · MLX

## Features

- **Continuous Batching Engine**: Dynamically groups requests for efficient GPU/CPU utilization
- **KV Cache**: Per-request key-value cache — O(n) decode instead of O(n²) re-forward
- **OpenAI Chat Completions API**: Drop-in replacement for `POST /v1/chat/completions` with streaming support
- **HuggingFace Integration**: Load any causal language model from HuggingFace Hub (Qwen3, GPT-2, Mistral, Llama, etc.)
- **Multi-device**: Auto-detects CUDA → MPS → CPU; MLX backend for Apple Silicon
- **Production Ready**: FastAPI + Uvicorn, comprehensive test coverage, Docker + Kubernetes deployment

## Quick Start

### Prerequisites
- Python 3.12+
- PyTorch 2.2+ (or 2.6+ on non-macOS platforms)
- `uv` package manager

### Installation

```bash
git clone https://github.com/yourusername/cuba.git
cd cuba
uv sync
```

### Running the Server

```bash
# Start with GPT-2 (small, fast, no auth required)
CUBA_MODEL_PATH=gpt2 uv run python cli_openai.py --host 127.0.0.1 --port 8000

# With Mistral (7B, higher quality)
CUBA_MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.2 uv run python cli_openai.py --host 0.0.0.0 --port 8000

# With gated models (requires HuggingFace token)
export HF_TOKEN=hf_YOUR_TOKEN_HERE
CUBA_MODEL_PATH=google/gemma-3-270m uv run python cli_openai.py --host 0.0.0.0 --port 8000
```

### Interactive Chat

```bash
uv run python examples/chat_terminal.py

# Type prompts:
> Hello! How are you?
 I'm doing well, thank you for asking!

> What is machine learning?
 Machine learning is a subset of artificial intelligence...

# Press 'q' to exit
```

## API Usage

### Health Check

```bash
curl http://127.0.0.1:8000/health | jq
```

### Chat Completion (Non-Streaming)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq '.choices[0].message.content'
```

### Streaming Chat Completion

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "max_tokens": 200,
    "temperature": 0.7,
    "stream": true
  }' -N | grep -o '"content":"[^"]*"' | cut -d'"' -f4
```

### Python Client (OpenAI API)

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # Cuba doesn't require auth
    base_url="http://127.0.0.1:8000/v1",
)

response = client.chat.completions.create(
    model="gpt2",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)

print(response.choices[0].message.content)
```

## Architecture

### Core Components

- **`ContinuousBatchingEngine`** (`src/cuba/engine/batching.py`)
  - Dynamic batch scheduling with prefill/decode phases
  - Request prioritization (decode-first to keep generations moving)
  - Concurrent request handling with per-request KV cache simulation
  - Built-in metrics collection

- **OpenAI API** (`src/cuba/openai/api.py`)
  - FastAPI application with OpenAI-compatible endpoints
  - Streaming via Server-Sent Events (SSE)
  - Health checks, model listing, metrics export

- **Runtime Bootstrap** (`src/cuba/runtime/bootstrap.py`)
  - Loads any HuggingFace causal language model
  - Automatic tokenizer setup
  - Engine configuration and initialization

### Request Lifecycle

```
User Request
    ↓
[OpenAI API Handler] /v1/chat/completions
    ↓
[ContinuousBatchingEngine]
    ├─ Tokenize input (HuggingFace tokenizer)
    ├─ Prefill phase (prompt encoding)
    ├─ Decode phase (generate tokens 1-by-1)
    └─ Detokenize output
    ↓
[JSON Response] (streaming or complete)
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUBA_MODEL_PATH` | Required | HuggingFace model ID (e.g., `gpt2`, `mistralai/Mistral-7B-Instruct-v0.2`) |
| `CUBA_BACKEND` | `engine` | Backend: `engine` (real model) or `stub` (dummy responses) |
| `CUBA_MAX_BATCH_SIZE` | `16` | Maximum requests per batch |
| `CUBA_MAX_WAIT_MS` | `10` | Max milliseconds to wait before scheduling batch |
| `CUBA_MAX_CONCURRENT_SEQUENCES` | `32` | Maximum concurrent requests |
| `CUBA_SCHEDULER_TICK_MS` | `5` | Scheduler loop frequency (milliseconds) |
| `CUBA_MAX_TOKENS` | `256` | Default max tokens for generation |
| `CUBA_TEMPERATURE` | `0.7` | Default sampling temperature |
| `HF_TOKEN` | - | HuggingFace API token (for gated models) |

### CLI Arguments

```bash
uv run python cli_openai.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model-path gpt2 \
  --max-batch-size 16 \
  --max-wait-ms 10 \
  --max-concurrent-sequences 32 \
  --scheduler-tick-ms 5
```

## Testing

### Run All Tests

```bash
uv run python -m pytest tests/ -v
```

**Test Coverage**: 7 tests covering stub backend, engine endpoints, streaming, concurrency, and metrics.

Tests use a minimal in-memory toy model (see `tests/conftest.py`) for fast, deterministic testing without downloading real models.

## Deployment

### Docker

```bash
# Build
docker build -t cuba:latest .

# Run with GPT-2
docker run -p 8000:8000 \
  -e CUBA_MODEL_PATH=gpt2 \
  cuba:latest

# Run with custom model and HuggingFace token
docker run -p 8000:8000 \
  -e CUBA_MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.2 \
  -e HF_TOKEN=hf_YOUR_TOKEN \
  cuba:latest
```

### Kubernetes

```bash
# Create namespace
kubectl create namespace cuba

# Update k8s/configmap.yaml with your MODEL_PATH, then apply:
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Port forward
kubectl -n cuba port-forward svc/cuba-api 8000:80

# Test
curl http://127.0.0.1:8000/health
```


## Benchmarks

Measured on **Intel MacBook Pro (x86_64, 8-core)**, model: `Qwen/Qwen3-0.6B`, single request, CPU-only.

| Engine | Format | Quantization | tok/s | Notes |
|--------|--------|-------------|-------|-------|
| **llama.cpp** | GGUF Q8_0 | GGML INT8 | **37.6** | Hand-written AVX2/SIMD C++ kernels |
| **Cuba** | HuggingFace | PyTorch INT8 | **~15** | KV cache + dynamic quantization |
| **vLLM** | HuggingFace | — | — | CPU not supported (GPU-only) |

**Why llama.cpp is faster on CPU:**
llama.cpp uses hand-written AVX2/AVX512 SIMD kernels in C++ — the tightest possible CPU implementation. Cuba uses PyTorch INT8 dynamic quantization which has Python overhead and less aggressive SIMD usage.

**Where Cuba wins:**
- **Any HuggingFace model** — no GGUF conversion needed
- **Continuous batching** — throughput scales with concurrent users (llama.cpp is single-request)
- **OpenAI-compatible API** — drop-in replacement, streaming, chat templates
- **Multi-device** — same code runs on CPU, CUDA, MPS, and MLX
- **KV cache** — linear decode scaling, handles long sequences efficiently

## Performance Tuning

### For High Throughput
```bash
# Larger batches, more concurrency
CUBA_MAX_BATCH_SIZE=32 \
CUBA_MAX_CONCURRENT_SEQUENCES=64 \
CUBA_MAX_WAIT_MS=50 \
uv run python cli_openai.py --host 0.0.0.0 --port 8000
```

### For Low Latency
```bash
# Small batches, quick scheduling
CUBA_MAX_BATCH_SIZE=8 \
CUBA_MAX_CONCURRENT_SEQUENCES=16 \
CUBA_MAX_WAIT_MS=5 \
uv run python cli_openai.py --host 0.0.0.0 --port 8000
```

### For Resource-Constrained Environments
```bash
# Smaller model, fewer concurrent requests
CUBA_MODEL_PATH=distilgpt2 \
CUBA_MAX_CONCURRENT_SEQUENCES=8 \
CUBA_MAX_BATCH_SIZE=4 \
uv run python cli_openai.py --host 127.0.0.1 --port 8000
```

## Supported Models

Cuba works with any causal language model from HuggingFace:

| Model | Size | Speed | Quality | Notes |
|-------|------|-------|---------|-------|
| `gpt2` | 548MB | Fast | Fair | Great for testing |
| `distilgpt2` | 336MB | Very Fast | Good | Smaller, faster GPT-2 |
| `TinyLlama/TinyLlama-1.1B` | 1.1B | Fast | Good | Instruction-tuned |
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | Medium | Excellent | High-quality, open |
| `meta-llama/Llama-2-7b-hf` | 7B | Medium | Excellent | Needs HuggingFace auth |
| `google/gemma-2-2b-it` | 2B | Fast | Good | Gated, needs HF_TOKEN |
| `google/gemma-3-270m` | 270M | Very Fast | Fair | Gated, needs HF_TOKEN |

## Troubleshooting

### "model_path is required" Error
```bash
# CUBA_MODEL_PATH environment variable is missing
export CUBA_MODEL_PATH=gpt2
uv run python cli_openai.py --host 127.0.0.1 --port 8000
```

### "Cannot access gated repo" Error
```bash
# Model requires authentication
export HF_TOKEN=hf_YOUR_TOKEN_HERE
export CUBA_MODEL_PATH=google/gemma-3-270m
uv run python cli_openai.py --host 0.0.0.0 --port 8000
```

### Port Already in Use
```bash
# Kill existing process or use different port
kill -9 $(lsof -nP -iTCP:8000 -sTCP:LISTEN | tail -1 | awk '{print $2}')
# Or use port 8001 instead
uv run python cli_openai.py --host 127.0.0.1 --port 8001
```

### Slow First Request
First request downloads the model (several hundred MB to several GB). Use smaller models for testing:
```bash
CUBA_MODEL_PATH=gpt2          # 548 MB
CUBA_MODEL_PATH=distilgpt2    # 336 MB
```

## Project Structure

```
cuba/
├── src/cuba/
│   ├── engine/              # Batching engine
│   │   └── batching.py      # ContinuousBatchingEngine
│   ├── openai/              # OpenAI API
│   │   ├── api.py           # FastAPI app
│   │   └── backend.py       # Backend protocol
│   ├── runtime/             # Runtime setup
│   │   └── bootstrap.py     # Model loading, engine config
│   ├── ops/                 # Utilities
│   ├── metrics/             # Monitoring & metrics
│   └── __init__.py
├── tests/                   # Test suite (all passing ✅)
│   ├── conftest.py          # Toy model fixture
│   ├── test_engine.py
│   ├── test_api.py
│   └── test_metrics.py
├── examples/
│   ├── chat_terminal.py     # Interactive CLI
│   └── openai_client.py     # Python client example
├── k8s/                     # Kubernetes manifests
├── Dockerfile
├── cli_openai.py            # Server entry point
├── pyproject.toml
└── README.md
```

## Development

### Install Dev Dependencies

```bash
uv sync --all-extras  # or manually: pip install -e '.[dev]'
```

### Run Tests

```bash
# All tests
uv run python -m pytest tests/ -v

# Single test
uv run python -m pytest tests/test_engine.py::test_engine_completes_request -v

# With coverage
uv run python -m pytest tests/ --cov=src/cuba --cov-report=html
```

### Type Checking

```bash
uv run mypy src/cuba/
```

## References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [FastAPI](https://fastapi.tiangelo.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat)
- [PyTorch](https://pytorch.org/)

## License

MIT (or your chosen license)
