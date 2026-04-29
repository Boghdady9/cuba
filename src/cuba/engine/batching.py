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
import torch.nn.functional as F

from cuba.metrics import MetricsCollector


@dataclass(slots=True)
class EngineConfig:
    max_batch_size: int = 16
    max_wait_ms: int = 10
    max_concurrent_sequences: int = 32
    scheduler_tick_ms: int = 5
    device: str = "cpu"
    # If True, mask EOS until `max_tokens` (toy in-memory model tests only).
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
    # KV cache state (populated after prefill)
    past_key_values: Any = field(default=None)
    cache_len: int = 0
    last_token_id: int | None = None

    @property
    def total_tokens(self) -> list[int]:
        return self.prompt_tokens + self.generated_token_ids


def _pad_and_stack_kv(batch: list[RequestRecord], max_past: int, device: str) -> Any:
    """Left-pad per-request KV caches to max_past and stack into a batched tensor.

    HF shape per layer: (1, num_heads, seq_len, head_dim).
    Returns: tuple of (key, value) per layer with shape (bsz, num_heads, max_past, head_dim).
    """
    if not batch or batch[0].past_key_values is None:
        return None
    num_layers = len(batch[0].past_key_values)
    result = []
    for layer_i in range(num_layers):
        keys, vals = [], []
        for req in batch:
            k, v = req.past_key_values[layer_i]  # (1, heads, cache_len, head_dim)
            pad = max_past - req.cache_len
            if pad > 0:
                # F.pad args go from last dim inward: (head_dim_r, head_dim_l, seq_r, seq_l)
                k = F.pad(k, (0, 0, pad, 0))
                v = F.pad(v, (0, 0, pad, 0))
            keys.append(k)
            vals.append(v)
        result.append((torch.cat(keys, dim=0), torch.cat(vals, dim=0)))
    return tuple(result)


def _slice_kv(past: Any, req_idx: int, new_cache_len: int) -> Any:
    """Extract one request's KV cache from a batched output, keeping last new_cache_len tokens."""
    result = []
    for k, v in past:
        result.append((
            k[req_idx : req_idx + 1, :, -new_cache_len:, :],
            v[req_idx : req_idx + 1, :, -new_cache_len:, :],
        ))
    return tuple(result)


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
            # Only yield to event loop when idle — avoids adding tick latency per token.
            if not self.pending_prefill and not self.pending_decode:
                await asyncio.sleep(tick)

    async def _run_one_iteration(self) -> None:
        self._promote_pending()
        # Decode-first: drain decode queue before processing new prefills.
        if self.pending_decode:
            batch = self._build_decode_batch()
            if batch:
                await self._process_decode_batch(batch)
        else:
            batch = self._build_prefill_batch()
            if batch:
                await self._process_prefill_batch(batch)

    def _promote_pending(self) -> None:
        while (
            self.pending_prefill
            and len(self.active_requests) < self.config.max_concurrent_sequences
        ):
            req = self.pending_prefill.popleft()
            req.status = "running"
            self.active_requests[req.id] = req

    def _build_decode_batch(self) -> list[RequestRecord]:
        batch: list[RequestRecord] = []
        while self.pending_decode and len(batch) < self.config.max_batch_size:
            req = self.pending_decode.popleft()
            if req.id in self.active_requests:
                batch.append(req)
        return batch

    def _build_prefill_batch(self) -> list[RequestRecord]:
        candidates = [r for r in self.active_requests.values() if not r.prefill_done]
        candidates.sort(key=lambda r: len(r.prompt_tokens), reverse=True)
        return candidates[: self.config.max_batch_size]

    async def _process_prefill_batch(self, batch: list[RequestRecord]) -> None:
        """Run prefill for each request individually (prompts differ in length)."""
        now = time.perf_counter()
        prompt_token_total = 0
        for req in batch:
            try:
                input_ids = torch.tensor(
                    [req.prompt_tokens], device=self.device, dtype=torch.long
                )
                attn_mask = torch.ones_like(input_ids)
                with torch.inference_mode():
                    out = self.model(input_ids, attention_mask=attn_mask, use_cache=True)

                logits = out.logits[0, -1]
                next_token = self._sample_token(logits, req)

                req.past_key_values = out.past_key_values
                req.cache_len = len(req.prompt_tokens)
                req.last_token_id = next_token
                req.prefill_done = True

                req.generated_token_ids.append(next_token)
                full_text = self._decode_generated(req.generated_token_ids)
                chunk = full_text[len(req.last_stream_text):]
                req.last_stream_text = full_text
                req.generated_text_parts.append(chunk)
                req.generated_tokens += 1
                req.first_token_at = now
                await req.token_queue.put(chunk)
                prompt_token_total += len(req.prompt_tokens)

                if req.generated_tokens >= req.max_tokens or (
                    not self.config.forbid_early_eos
                    and next_token == self.tokenizer.eos_token_id
                ):
                    await self._complete_request(req, now)
                else:
                    self.pending_decode.append(req)

            except Exception as exc:  # noqa: BLE001
                await self._fail_request(req, exc)

        if self.metrics is not None:
            self.metrics.record_scheduler_batch(
                batch_size=len(batch),
                prompt_tokens=prompt_token_total,
                decode_tokens=0,
            )

    async def _process_decode_batch(self, batch: list[RequestRecord]) -> None:
        """Run one decode step for all requests, batched with per-request KV caches."""
        now = time.perf_counter()
        bsz = len(batch)
        max_past = max(req.cache_len for req in batch)

        # (bsz, 1) — each request contributes its most recently sampled token
        input_ids = torch.tensor(
            [[req.last_token_id] for req in batch],
            device=self.device, dtype=torch.long,
        )
        # Attention mask: zeros for left-padding, ones for valid positions
        attn_mask = torch.zeros((bsz, max_past + 1), dtype=torch.long, device=self.device)
        for i, req in enumerate(batch):
            attn_mask[i, max_past - req.cache_len :] = 1

        try:
            past = _pad_and_stack_kv(batch, max_past, self.device)
            with torch.inference_mode():
                out = self.model(
                    input_ids,
                    attention_mask=attn_mask,
                    past_key_values=past,
                    use_cache=True,
                )
            for i, req in enumerate(batch):
                req.past_key_values = _slice_kv(out.past_key_values, i, req.cache_len + 1)
                req.cache_len += 1
            logits_batch = out.logits[:, -1, :]  # (bsz, vocab)
        except Exception:  # noqa: BLE001 — fall back to sequential per-request decode
            logits_list = []
            for req in batch:
                in_single = torch.tensor(
                    [[req.last_token_id]], device=self.device, dtype=torch.long
                )
                mask_single = torch.ones(
                    (1, req.cache_len + 1), dtype=torch.long, device=self.device
                )
                with torch.inference_mode():
                    out = self.model(
                        in_single,
                        attention_mask=mask_single,
                        past_key_values=req.past_key_values,
                        use_cache=True,
                    )
                req.past_key_values = out.past_key_values
                req.cache_len += 1
                logits_list.append(out.logits[0, -1])
            logits_batch = torch.stack(logits_list)

        for i, req in enumerate(batch):
            next_token = self._sample_token(logits_batch[i], req)
            req.last_token_id = next_token
            req.generated_token_ids.append(next_token)
            full_text = self._decode_generated(req.generated_token_ids)
            chunk = (
                full_text[len(req.last_stream_text):]
                if full_text.startswith(req.last_stream_text)
                else full_text
            )
            req.last_stream_text = full_text
            req.generated_text_parts.append(chunk)
            req.generated_tokens += 1
            await req.token_queue.put(chunk)

            if req.generated_tokens >= req.max_tokens or (
                not self.config.forbid_early_eos
                and next_token == self.tokenizer.eos_token_id
            ):
                await self._complete_request(req, now)
            else:
                self.pending_decode.append(req)

        if self.metrics is not None:
            self.metrics.record_scheduler_batch(
                batch_size=len(batch),
                prompt_tokens=0,
                decode_tokens=len(batch),
            )

    async def _complete_request(self, req: RequestRecord, now: float) -> None:
        req.status = "completed"
        req.completed_at = now
        final_text = self._decode_generated(req.generated_token_ids)
        if req.completion is not None and not req.completion.done():
            req.completion.set_result(final_text)
        await self._emit_terminal(req)
        self.active_requests.pop(req.id, None)
        self._recent_completed.append(req)
        self.completed_sequences.append(self._snapshot_request(req))

    def _decode_generated(self, token_ids: list[int]) -> str:
        dec = self.tokenizer.decode
        try:
            return str(dec(token_ids, skip_special_tokens=True))  # type: ignore[call-arg]
        except TypeError:
            return str(dec(token_ids))

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
        """Block pad / bos. Optionally block EOS (toy engine) so runs hit max_tokens."""
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
