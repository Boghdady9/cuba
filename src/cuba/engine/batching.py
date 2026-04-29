from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import torch

from cuba.metrics import MetricsCollector


@dataclass(slots=True)
class EngineConfig:
    max_batch_size: int = 16
    max_wait_ms: int = 10
    max_concurrent_sequences: int = 32
    scheduler_tick_ms: int = 5
    device: str = "cpu"
    # If True, mask EOS until `max_tokens` (toy in-memory model tests only; real models should stop at EOS).
    forbid_early_eos: bool = False


@dataclass(slots=True)
class RequestRecord:
    id: str
    prompt: str
    max_tokens: int
    temperature: float
    prompt_tokens: list[int]
    arrival_time: float
    token_queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    completion: asyncio.Future[str] | None = None
    generated_token_ids: list[int] = field(default_factory=list)
    generated_text_parts: list[str] = field(default_factory=list)
    # Cumulative decoded assistant text (handles BPE: single-id decode is wrong).
    last_stream_text: str = ""
    generated_tokens: int = 0
    status: str = "pending"
    prefill_done: bool = False
    first_token_at: float | None = None
    completed_at: float | None = None
    error: str | None = None

    @property
    def total_tokens(self) -> list[int]:
        return self.prompt_tokens + self.generated_token_ids


class ContinuousBatchingEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        config: EngineConfig | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        self.metrics = metrics

        self.max_batch_size = self.config.max_batch_size
        self.device = self.config.device

        self.pending_prefill: deque[RequestRecord] = deque()
        self.pending_decode: deque[RequestRecord] = deque()
        self.active_requests: dict[str, RequestRecord] = {}
        self.completed_sequences: list[dict[str, Any]] = []
        self._recent_completed: deque[RequestRecord] = deque(maxlen=512)

        self._request_index: dict[str, RequestRecord] = {}
        self._wake_event = asyncio.Event()
        self._scheduler_task: asyncio.Task[None] | None = None
        self._closed = False
        self._scheduler_iterations = 0
        self._last_scheduler_duration = 0.0

    @property
    def pending_requests(self) -> list[RequestRecord]:
        return list(self.pending_prefill) + list(self.pending_decode)

    @property
    def running_sequences(self) -> list[RequestRecord]:
        return list(self.active_requests.values())

    @property
    def scheduler_stats(self) -> dict[str, float | int]:
        return {
            "iterations": self._scheduler_iterations,
            "last_loop_seconds": self._last_scheduler_duration,
            "queue_depth": len(self.pending_prefill) + len(self.pending_decode),
            "active_sequences": len(self.active_requests),
        }

    async def start(self) -> None:
        if self._scheduler_task is None or self._scheduler_task.done():
            self._closed = False
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        self._closed = True
        self._wake_event.set()
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
        self._scheduler_task = None
        for req in list(self._request_index.values()):
            if req.completion is not None and not req.completion.done():
                req.completion.set_exception(RuntimeError("engine stopped"))
            await self._emit_terminal(req, error="engine stopped")
        self.active_requests.clear()
        self.pending_prefill.clear()
        self.pending_decode.clear()

    async def submit(
        self,
        prompt: str,
        max_tokens: int = 100,
        *,
        temperature: float = 0.7,
    ) -> RequestRecord:
        if self._closed:
            raise RuntimeError("engine is stopped")
        loop = asyncio.get_running_loop()
        request_id = str(uuid.uuid4())
        record = RequestRecord(
            id=request_id,
            prompt=prompt,
            max_tokens=max(1, int(max_tokens)),
            temperature=float(temperature),
            prompt_tokens=list(self.tokenizer.encode(prompt)),
            arrival_time=time.perf_counter(),
            completion=loop.create_future(),
        )
        if not record.prompt_tokens:
            record.prompt_tokens = [self.tokenizer.bos_token_id]
        self.pending_prefill.append(record)
        self._request_index[record.id] = record
        self._wake_event.set()
        return record

    async def add_request(self, prompt: str, max_tokens: int = 100, **kwargs: Any) -> str:
        record = await self.submit(
            prompt,
            max_tokens=max_tokens,
            temperature=float(kwargs.get("temperature", 0.7)),
        )
        return record.id

    def get_request(self, request_id: str) -> RequestRecord | None:
        return self._request_index.get(request_id)

    async def stream_request(self, request_id: str) -> AsyncIterator[str]:
        record = self._request_index[request_id]
        while True:
            item = await record.token_queue.get()
            if item is None:
                break
            yield item

    async def wait_for_completion(self, request_id: str) -> str:
        record = self._request_index[request_id]
        assert record.completion is not None
        return await record.completion

    async def process_batch(self) -> None:
        await self._run_one_iteration()

    async def _scheduler_loop(self) -> None:
        tick = max(0.001, self.config.scheduler_tick_ms / 1000.0)
        while not self._closed:
            if not self.pending_prefill and not self.pending_decode and not self.active_requests:
                self._wake_event.clear()
                await self._wake_event.wait()
                if self._closed:
                    break
            started = time.perf_counter()
            await self._run_one_iteration()
            self._scheduler_iterations += 1
            self._last_scheduler_duration = time.perf_counter() - started
            if self.metrics is not None:
                self.metrics.record_scheduler_state(
                    queue_depth=len(self.pending_prefill) + len(self.pending_decode),
                    active_sequences=len(self.active_requests),
                    scheduler_loop_seconds=self._last_scheduler_duration,
                )
            await asyncio.sleep(tick)

    async def _run_one_iteration(self) -> None:
        self._promote_pending()
        batch = self._build_batch()
        if not batch:
            return
        inputs = self._build_batch_inputs(batch)
        try:
            outputs = await self._forward_batch(inputs)
            await self._apply_outputs(batch, outputs)
        except Exception as exc:  # noqa: BLE001
            for req in batch:
                await self._fail_request(req, exc)

    def _promote_pending(self) -> None:
        while (
            self.pending_prefill
            and len(self.active_requests) < self.config.max_concurrent_sequences
        ):
            req = self.pending_prefill.popleft()
            req.status = "running"
            self.active_requests[req.id] = req

    def _build_batch(self) -> list[RequestRecord]:
        batch: list[RequestRecord] = []

        # Decode-first so long prompts do not starve active generations.
        while self.pending_decode and len(batch) < self.config.max_batch_size:
            req = self.pending_decode.popleft()
            if req.id in self.active_requests:
                batch.append(req)

        if len(batch) >= self.config.max_batch_size:
            return batch

        prefill_candidates = [
            req for req in self.active_requests.values() if not req.prefill_done and req not in batch
        ]
        prefill_candidates.sort(key=lambda req: len(req.prompt_tokens), reverse=True)
        for req in prefill_candidates:
            if len(batch) >= self.config.max_batch_size:
                break
            batch.append(req)
        return batch

    def _build_batch_inputs(self, batch: list[RequestRecord]) -> list[list[int]]:
        # Causal LM needs full context each step (or KV cache; we use full re-forward).
        inputs: list[list[int]] = []
        for req in batch:
            if not req.prefill_done:
                inputs.append(req.prompt_tokens)
            else:
                inputs.append(req.prompt_tokens + req.generated_token_ids)
        return inputs

    async def _forward_batch(self, batch_inputs: list[list[int]]) -> Any:
        max_len = max(len(seq) for seq in batch_inputs)
        padded_inputs = [seq + [self.tokenizer.pad_token_id] * (max_len - len(seq)) for seq in batch_inputs]
        input_tensor = torch.tensor(padded_inputs, device=self.device, dtype=torch.long)
        with torch.inference_mode():
            return self.model(input_tensor)

    def _decode_generated(self, token_ids: list[int]) -> str:
        dec = self.tokenizer.decode
        try:
            return str(dec(token_ids, skip_special_tokens=True))  # type: ignore[call-arg]
        except TypeError:
            return str(dec(token_ids))

    async def _apply_outputs(self, batch: list[RequestRecord], outputs: Any) -> None:
        now = time.perf_counter()
        prompt_token_total = 0
        decode_token_total = 0
        for index, req in enumerate(batch):
            logits = outputs.logits[index, -1]
            next_token = self._sample_token(logits, req)

            req.prefill_done = True
            req.generated_token_ids.append(next_token)
            full_text = self._decode_generated(req.generated_token_ids)
            if full_text.startswith(req.last_stream_text):
                chunk = full_text[len(req.last_stream_text) :]
            else:
                # Rare tokenizer edge: emit full new string (re-sync stream)
                chunk = full_text
            req.last_stream_text = full_text
            req.generated_text_parts.append(chunk)
            req.generated_tokens += 1
            if req.first_token_at is None:
                req.first_token_at = now
            await req.token_queue.put(chunk)

            if req.generated_tokens >= req.max_tokens or next_token == self.tokenizer.eos_token_id:
                req.status = "completed"
                req.completed_at = now
                if req.completion is not None and not req.completion.done():
                    req.completion.set_result(self._decode_generated(req.generated_token_ids))
                await self._emit_terminal(req)
                self.active_requests.pop(req.id, None)
                self._recent_completed.append(req)
                self.completed_sequences.append(self._snapshot_request(req))
            else:
                self.pending_decode.append(req)

            if len(req.generated_token_ids) == 1:
                prompt_token_total += len(req.prompt_tokens)
            else:
                decode_token_total += 1

        if self.metrics is not None:
            self.metrics.record_scheduler_batch(
                batch_size=len(batch),
                prompt_tokens=prompt_token_total,
                decode_tokens=decode_token_total,
            )

    async def _emit_terminal(self, req: RequestRecord, error: str | None = None) -> None:
        if error is not None:
            req.error = error
        await req.token_queue.put(None)

    async def _fail_request(self, req: RequestRecord, exc: Exception) -> None:
        req.status = "failed"
        req.completed_at = time.perf_counter()
        if req.completion is not None and not req.completion.done():
            req.completion.set_exception(exc)
        await self._emit_terminal(req, error=str(exc))
        self.active_requests.pop(req.id, None)
        self.completed_sequences.append(self._snapshot_request(req))

    def _snapshot_request(self, req: RequestRecord) -> dict[str, Any]:
        return {
            "id": req.id,
            "prompt": req.prompt,
            "tokens": req.total_tokens,
            "generated_tokens": req.generated_tokens,
            "status": req.status,
            "t_started": req.arrival_time,
            "t_first_token": req.first_token_at,
            "t_completed": req.completed_at,
            "error": req.error,
        }

    def _mask_unsampleable(self, logits: torch.Tensor, req: RequestRecord) -> torch.Tensor:
        """Block pad / bos. Optionally block EOS (toy engine) so runs hit ``max_tokens``."""
        out = logits.clone()
        m = torch.finfo(out.dtype).min
        tok = self.tokenizer
        for name in ("pad_token_id", "bos_token_id"):
            tid = getattr(tok, name, None)
            if tid is not None and 0 <= int(tid) < out.shape[0]:
                out[int(tid)] = m
        eos = getattr(tok, "eos_token_id", None)
        if (
            self.config.forbid_early_eos
            and eos is not None
            and 0 <= int(eos) < out.shape[0]
            and len(req.generated_token_ids) < req.max_tokens
        ):
            out[int(eos)] = m
        return out

    def _sample_token(self, logits: torch.Tensor, req: RequestRecord) -> int:
        logits = self._mask_unsampleable(logits, req)
        temperature = float(req.temperature)
        if temperature <= 0:
            return int(torch.argmax(logits).item())
        scaled = logits / max(temperature, 1e-5)
        probs = torch.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, 1).item())
