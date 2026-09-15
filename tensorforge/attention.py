"""CPU attention memory experiments and position-bias helpers.

The tiled implementation recomputes probabilities in backward. It is not a
CUDA FlashAttention kernel and makes no GPU-throughput claim.
"""

import math

import numpy as np

from .nn.modules import Module, Parameter
from .tensor import Tensor


def alibi_bias(num_heads, query_length, key_length=None, query_offset=0):
    if any(type(n) is not int or n < 1 for n in (num_heads, query_length)):
        raise ValueError("Head count and query length must be positive integers.")
    key_length = query_length if key_length is None else key_length
    if (
        type(key_length) is not int
        or key_length < 1
        or type(query_offset) is not int
        or query_offset < 0
    ):
        raise ValueError("Invalid key length or query offset.")

    def slopes(n):
        if n & (n - 1) == 0:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start**i for i in range(1, n + 1)]
        power = 2 ** int(math.log2(n))
        return slopes(power) + slopes(2 * power)[::2][: n - power]

    distance = np.abs(
        np.arange(query_length)[:, None] + query_offset - np.arange(key_length)[None, :]
    )
    return Tensor(-np.asarray(slopes(num_heads))[:, None, None] * distance)


class RelativePositionBias(Module):
    """Learned per-head biases with clipped signed relative distances."""

    def __init__(self, num_heads, max_distance=16):
        super().__init__()
        if (
            type(num_heads) is not int
            or type(max_distance) is not int
            or num_heads < 1
            or max_distance < 1
        ):
            raise ValueError("num_heads and max_distance must be positive integers.")
        self.max_distance = max_distance
        self.weight = Parameter(np.zeros((num_heads, 2 * max_distance + 1)))

    def forward(self, query_length, key_length=None, query_offset=0):
        key_length = query_length if key_length is None else key_length
        if (
            any(type(n) is not int or n < 1 for n in (query_length, key_length))
            or type(query_offset) is not int
            or query_offset < 0
        ):
            raise ValueError("Invalid sequence lengths or offset.")
        distance = np.arange(key_length)[None, :] - (
            np.arange(query_length)[:, None] + query_offset
        )
        buckets = np.clip(distance, -self.max_distance, self.max_distance) + self.max_distance
        return self.weight[:, buckets]


def memory_efficient_attention(
    query, key, value, block_size=32, is_causal=False, window_size=None, mask=None, query_offset=0
):
    """Exact attention computed in query tiles, without retaining full scores.

    Leading batch/head shapes must match. Window attention is causal and keeps
    the current key and at most window_size-1 earlier keys. True masks block.
    """
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be positive.")
    if window_size is not None and (
        type(window_size) is not int or window_size < 1 or not is_causal
    ):
        raise ValueError("A positive window_size requires causal attention.")
    if (
        min(query.ndim, key.ndim, value.ndim) < 2
        or query.shape[:-2] != key.shape[:-2]
        or key.shape[:-2] != value.shape[:-2]
        or query.shape[-1] != key.shape[-1]
        or key.shape[-2] != value.shape[-2]
        or query.shape[-1] == 0
        or query.shape[-2] == 0
        or key.shape[-2] == 0
    ):
        raise ValueError("Attention shapes are incompatible or empty.")
    if type(query_offset) is not int or query_offset < 0:
        raise ValueError("query_offset must be nonnegative.")
    q, k, v = query._data, key._data, value._data
    if any(a.dtype.kind != "f" for a in (q, k, v)):
        raise TypeError("Attention inputs must be floating point.")
    if not all(np.isfinite(a).all() for a in (q, k, v)):
        raise ValueError("Attention inputs must be finite.")
    full_shape = q.shape[:-1] + (k.shape[-2],)
    if mask is not None:
        mask = mask.numpy() if isinstance(mask, Tensor) else np.asarray(mask)
        if mask.dtype.kind != "b":
            raise TypeError("mask must be boolean.")
        mask = np.broadcast_to(mask.copy(), full_shape)
    scale = 1 / np.sqrt(q.shape[-1])

    def probabilities(start, stop):
        scores = (q[..., start:stop, :] @ np.swapaxes(k, -2, -1)) * scale
        blocked = np.zeros(scores.shape, dtype=bool)
        if mask is not None:
            blocked |= mask[..., start:stop, :]
        positions = np.arange(start, stop)[:, None] + query_offset
        keys = np.arange(k.shape[-2])[None, :]
        if is_causal:
            blocked |= keys > positions
        if window_size is not None:
            blocked |= keys <= positions - window_size
        scores = np.where(blocked, -np.inf, scores)
        maximum = scores.max(axis=-1, keepdims=True)
        maximum = np.where(np.isfinite(maximum), maximum, 0)
        p = np.exp(scores - maximum)
        denominator = p.sum(axis=-1, keepdims=True)
        return p / np.where(denominator == 0, 1, denominator)

    output = np.empty(q.shape[:-1] + (v.shape[-1],), dtype=np.result_type(q, k, v))
    for start in range(0, q.shape[-2], block_size):
        stop = min(start + block_size, q.shape[-2])
        output[..., start:stop, :] = probabilities(start, stop) @ v

    def backward(g):
        dq, dk, dv = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
        for start in range(0, q.shape[-2], block_size):
            stop = min(start + block_size, q.shape[-2])
            p = probabilities(start, stop)
            incoming = g[..., start:stop, :]
            dp = incoming @ np.swapaxes(v, -2, -1)
            ds = p * (dp - (dp * p).sum(axis=-1, keepdims=True))
            dq[..., start:stop, :] = (ds @ k) * scale
            dk += (np.swapaxes(ds, -2, -1) @ q[..., start:stop, :]) * scale
            dv += np.swapaxes(p, -2, -1) @ incoming
        return dq, dk, dv

    return Tensor._op(output, (query, key, value), backward)
