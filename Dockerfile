FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src/cuba /app/src/cuba
COPY cli_openai.py /app/
COPY scripts /app/scripts

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "httpx>=0.28.1" \
    "transformers==4.39.3" \
    "sentencepiece" \
    && pip install --no-cache-dir -e /app

EXPOSE 8000

ENV CUBA_BACKEND=engine \
    MAX_BATCH_SIZE=16 \
    CUBA_MAX_WAIT_MS=10 \
    CUBA_MAX_CONCURRENT_SEQUENCES=32 \
    CUBA_SCHEDULER_TICK_MS=5

# Set MODEL_PATH at runtime: -e MODEL_PATH=google/gemma-3-270m (or any HuggingFace model)

CMD ["python", "cli_openai.py", "--host", "0.0.0.0", "--port", "8000"]
