"""Paged CPU key/value storage for independent inference requests.

This storage utility is not an HTTP server or a fused attention backend.
Absolute positions are exposed so callers can align rotary/causal positions.
"""

import numpy as np

from .tensor import Tensor


class PagedKVCache:
    def __init__(self, num_pages, page_size, num_heads, head_dim, dtype=np.float64):
        if any(type(n) is not int or n < 1 for n in (num_pages, page_size, num_heads, head_dim)):
            raise ValueError("Cache dimensions must be positive integers.")
        if np.dtype(dtype).kind != "f":
            raise TypeError("Cache storage must be floating point.")
        self.page_size, self.num_heads, self.head_dim = page_size, num_heads, head_dim
        self.keys = np.empty((num_pages, num_heads, page_size, head_dim), dtype=dtype)
        self.values = np.empty_like(self.keys)
        self.free = list(reversed(range(num_pages)))
        self.requests = {}

    def append(self, request_id, keys, values):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string.")
        k = keys.numpy() if isinstance(keys, Tensor) else np.asarray(keys)
        v = values.numpy() if isinstance(values, Tensor) else np.asarray(values)
        if (
            k.ndim != 3
            or k.shape != v.shape
            or k.shape[0] != self.num_heads
            or k.shape[-1] != self.head_dim
            or k.shape[1] == 0
        ):
            raise ValueError("Cache entries must have shape (heads,nonempty_time,head_dim).")
        if not np.isfinite(k).all() or not np.isfinite(v).all():
            raise ValueError("Cache entries must be finite.")
        if k.dtype.kind not in "fiu" or v.dtype.kind not in "fiu":
            raise TypeError("Cache entries must be real numbers.")
        with np.errstate(over="ignore", invalid="ignore"):
            k, v = k.astype(self.keys.dtype), v.astype(self.values.dtype)
        if not np.isfinite(k).all() or not np.isfinite(v).all():
            raise ValueError("Entries overflow the cache dtype.")
        state = self.requests.get(request_id, {"start": 0, "end": 0, "pages": {}})
        end = state["end"] + k.shape[1]
        blocks = range(state["end"] // self.page_size, (end - 1) // self.page_size + 1)
        needed = [block for block in blocks if block not in state["pages"]]
        if len(needed) > len(self.free):
            raise MemoryError("No free cache pages; evict or release a request first.")
        self.requests[request_id] = state
        for block in needed:
            state["pages"][block] = self.free.pop()
        for index in range(k.shape[1]):
            position = state["end"] + index
            page = state["pages"][position // self.page_size]
            offset = position % self.page_size
            self.keys[page, :, offset, :] = k[:, index, :]
            self.values[page, :, offset, :] = v[:, index, :]
        state["end"] = end

    def get(self, request_id):
        state = self.requests[request_id]
        positions = np.arange(state["start"], state["end"])
        shape = (self.num_heads, len(positions), self.head_dim)
        keys, values = (
            np.empty(shape, dtype=self.keys.dtype),
            np.empty(shape, dtype=self.values.dtype),
        )
        for index, position in enumerate(positions):
            page = state["pages"][int(position) // self.page_size]
            offset = int(position) % self.page_size
            keys[:, index] = self.keys[page, :, offset]
            values[:, index] = self.values[page, :, offset]
        return Tensor(keys, dtype=keys.dtype), Tensor(values, dtype=values.dtype), positions

    def evict(self, request_id, keep_last):
        if type(keep_last) is not int or keep_last < 0:
            raise ValueError("keep_last must be nonnegative.")
        state = self.requests[request_id]
        state["start"] = max(state["start"], state["end"] - keep_last)
        for block in list(state["pages"]):
            if state["start"] == state["end"] or (block + 1) * self.page_size <= state["start"]:
                self.free.append(state["pages"].pop(block))

    def release(self, request_id):
        state = self.requests.pop(request_id)
        self.free.extend(state["pages"].values())
