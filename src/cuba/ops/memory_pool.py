"""Pre-allocated CPU tensor pool to reuse buffers and reduce alloc churn."""
from __future__ import annotations

from collections import deque
from collections.abc import Hashable
import torch


class CPUMemoryPool:
    def __init__(
        self,
        pool_size: int = 1000,
        tensor_shape: tuple[int, ...] = (32, 128),
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        self.pool_size = pool_size
        self.tensor_shape = tensor_shape
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        self.tensor_pool: list[torch.Tensor] = [
            torch.zeros(tensor_shape, dtype=self.dtype, device=self.device)
            for _ in range(pool_size)
        ]
        self._available: deque[int] = deque(range(pool_size))
        self.allocated_tensors: dict[Hashable, int] = {}

    def allocate_tensor(self, request_id: Hashable) -> torch.Tensor:
        if not self._available:
            raise RuntimeError("No available tensors in pool")
        if request_id in self.allocated_tensors:
            raise KeyError(f"request_id {request_id!r} already has an allocated tensor")
        tensor_id = self._available.popleft()
        self.allocated_tensors[request_id] = tensor_id
        return self.tensor_pool[tensor_id]

    def free_tensor(self, request_id: Hashable) -> None:
        if request_id not in self.allocated_tensors:
            return
        tensor_id = self.allocated_tensors.pop(request_id)
        self._available.append(tensor_id)
