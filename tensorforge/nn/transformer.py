"""Batch-first attention and Transformer blocks.

All boolean masks use True for positions that must be blocked. Cached
attention is an inference-only API: call eval() and enter no_grad().
"""

import importlib

import numpy as np

from ..tensor import Tensor, arange, cat, stack
from .functional import scaled_dot_product_attention
from .modules import (
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    RMSNorm,
    _positive_int,
)

_core = importlib.import_module("tensorforge.tensor")


class SinusoidalPositionEncoding(Module):
    def __init__(self, features, max_length=2048):
        super().__init__()
        _positive_int(features, "features")
        _positive_int(max_length, "max_length")
        positions = np.arange(max_length)[:, None]
        frequencies = np.exp(-np.log(10000) * np.arange(0, features, 2) / features)
        table = np.zeros((max_length, features))
        table[:, 0::2] = np.sin(positions * frequencies)
        table[:, 1::2] = np.cos(positions * frequencies[: features // 2])
        self.register_buffer("table", Tensor(table))

    def forward(self, x, offset=0):
        if (
            x.ndim != 3
            or x.shape[-1] != self.table.shape[-1]
            or type(offset) is not int
            or offset < 0
            or offset + x.shape[1] > len(self.table)
        ):
            raise ValueError("Position encoding input shape or offset is invalid.")
        return x + self.table[offset : offset + x.shape[1]]


class LearnedPositionEncoding(Module):
    def __init__(self, features, max_length=2048):
        super().__init__()
        self.embedding = Embedding(max_length, features)

    def forward(self, x, offset=0):
        if (
            x.ndim != 3
            or x.shape[-1] != self.embedding.weight.shape[1]
            or type(offset) is not int
            or offset < 0
            or offset + x.shape[1] > self.embedding.weight.shape[0]
        ):
            raise ValueError("Position encoding input shape or offset is invalid.")
        return x + self.embedding(arange(offset, offset + x.shape[1], dtype=np.int64))


class RotaryEmbedding(Module):
    """Rotate adjacent feature pairs using their absolute token positions."""

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        _positive_int(head_dim, "head_dim")
        if head_dim % 2 or not np.isfinite(base) or base <= 0:
            raise ValueError("RoPE requires even head_dim and a finite positive base.")
        self.head_dim = head_dim
        self.register_buffer("frequencies", Tensor(base ** (-np.arange(0, head_dim, 2) / head_dim)))

    def forward(self, x, offset=0):
        if x.ndim != 4 or x.shape[-1] != self.head_dim or type(offset) is not int or offset < 0:
            raise ValueError("RoPE expects (batch, heads, time, head_dim) and nonnegative offset.")
        angles = np.arange(offset, offset + x.shape[-2])[:, None] * self.frequencies._data
        cosine, sine = Tensor(np.cos(angles)), Tensor(np.sin(angles))
        even, odd = x[..., 0::2], x[..., 1::2]
        return stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1).reshape(
            x.shape
        )


class MultiheadAttention(Module):
    """Self/cross-attention with optional grouped key/value heads and RoPE.

    Inputs and outputs are (batch, time, embed_dim). The result is a Tensor;
    use_cache=True returns (result, (keys, values)). Cache shapes use the
    unrepeated key/value head count. General masks broadcast to (B,H,Q,K);
    key_padding_mask has shape (B,K). Masked query rows produce zero attention
    before the output projection, whose bias can make final rows nonzero.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, num_kv_heads=None, rope=False, bias=True):
        super().__init__()
        _positive_int(embed_dim, "embed_dim")
        _positive_int(num_heads, "num_heads")
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        _positive_int(num_kv_heads, "num_kv_heads")
        if embed_dim % num_heads or num_heads % num_kv_heads:
            raise ValueError("embed_dim must divide by heads; heads must divide by kv heads.")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.num_kv_heads, self.head_dim = num_kv_heads, embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim) if rope else None

    def forward(
        self,
        query,
        key=None,
        value=None,
        mask=None,
        key_padding_mask=None,
        is_causal=False,
        cache=None,
        use_cache=False,
    ):
        self_attention = key is None and value is None
        key = query if key is None else key
        value = key if value is None else value
        if any(x.ndim != 3 or x.shape[-1] != self.embed_dim for x in (query, key, value)):
            raise ValueError("Attention inputs must have shape (batch,time,embed_dim).")
        if query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]:
            raise ValueError("Attention batch sizes and key/value lengths must match.")
        if query.shape[1] == 0 or key.shape[1] == 0:
            raise ValueError("Attention sequences must not be empty.")
        if self.rope is not None and not self_attention:
            raise ValueError("RoPE in this module is supported for self-attention only.")
        if cache is not None or use_cache:
            if not self_attention or self.training or _core._grad_enabled.get():
                raise RuntimeError("Caching needs self-attention, eval(), and no_grad().")
        batch = query.shape[0]
        q = self.q_proj(query).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        offset = 0
        if cache is not None:
            if not isinstance(cache, (tuple, list)) or len(cache) != 2:
                raise ValueError("Cache must be a pair of key/value tensors.")
            ck, cv = cache
            if (
                not isinstance(ck, Tensor)
                or not isinstance(cv, Tensor)
                or ck.ndim != 4
                or ck.shape != cv.shape
                or ck.shape[:2] != (batch, self.num_kv_heads)
                or ck.shape[-1] != self.head_dim
                or ck.requires_grad
                or cv.requires_grad
            ):
                raise ValueError("Cache shape or gradient state is invalid.")
            offset = ck.shape[-2]
        if self.rope is not None:
            q, k = self.rope(q, offset), self.rope(k, offset)
        if cache is not None:
            k, v = cat((cache[0], k), axis=2), cat((cache[1], v), axis=2)
        new_cache = (k.detach(), v.detach()) if use_cache else None
        total = k.shape[2]
        repeats = self.num_heads // self.num_kv_heads
        if repeats != 1:
            k = (
                k.unsqueeze(2)
                .expand(batch, self.num_kv_heads, repeats, total, self.head_dim)
                .reshape(batch, self.num_heads, total, self.head_dim)
            )
            v = (
                v.unsqueeze(2)
                .expand(batch, self.num_kv_heads, repeats, total, self.head_dim)
                .reshape(batch, self.num_heads, total, self.head_dim)
            )
        if key_padding_mask is not None:
            padding = (
                key_padding_mask.numpy()
                if isinstance(key_padding_mask, Tensor)
                else np.asarray(key_padding_mask)
            )
            if padding.dtype.kind != "b" or padding.shape != (batch, total):
                raise ValueError("key_padding_mask must be boolean with shape (batch,key_length).")
            padding = padding[:, None, None, :]
            if mask is None:
                mask = padding
            else:
                mask = mask.numpy() if isinstance(mask, Tensor) else np.asarray(mask)
                if mask.dtype.kind != "b":
                    raise TypeError("Attention masks must be boolean.")
                mask = mask | padding
        result = scaled_dot_product_attention(
            q, k, v, mask, is_causal, self.dropout.p, self.training, offset
        )
        result = result.transpose(1, 2).reshape(batch, query.shape[1], self.embed_dim)
        result = self.out_proj(result)
        return (result, new_cache) if use_cache else result


class FeedForward(Module):
    """GELU or SwiGLU feature transformation, followed by dropout."""

    def __init__(self, features, hidden_size, dropout=0.0, activation="gelu", bias=True):
        super().__init__()
        if activation not in ("gelu", "swiglu"):
            raise ValueError("activation must be 'gelu' or 'swiglu'.")
        self.activation = activation
        self.up = Linear(features, hidden_size, bias=bias)
        self.gate = Linear(features, hidden_size, bias=bias) if activation == "swiglu" else None
        self.down = Linear(hidden_size, features, bias=bias)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        hidden = self.up(x)
        hidden = hidden.gelu() if self.gate is None else self.gate(x).silu() * hidden
        return self.dropout(self.down(hidden))


class TransformerEncoderLayer(Module):
    """Pre-normalized self-attention and feed-forward residual blocks."""

    def __init__(self, features, num_heads, hidden_size=None, dropout=0.0):
        super().__init__()
        hidden_size = 4 * features if hidden_size is None else hidden_size
        self.norm1, self.norm2 = LayerNorm(features), LayerNorm(features)
        self.attention = MultiheadAttention(features, num_heads, dropout)
        self.feed_forward = FeedForward(features, hidden_size, dropout)
        self.dropout = Dropout(dropout)

    def forward(self, x, mask=None, padding_mask=None):
        x = x + self.dropout(
            self.attention(self.norm1(x), mask=mask, key_padding_mask=padding_mask)
        )
        return x + self.feed_forward(self.norm2(x))


class TransformerEncoder(Module):
    def __init__(self, features, num_heads, num_layers=1, hidden_size=None, dropout=0.0):
        super().__init__()
        _positive_int(num_layers, "num_layers")
        self.layers = ModuleList(
            TransformerEncoderLayer(features, num_heads, hidden_size, dropout)
            for _ in range(num_layers)
        )
        self.norm = LayerNorm(features)

    def forward(self, x, mask=None, padding_mask=None):
        for layer in self.layers:
            x = layer(x, mask, padding_mask)
        return self.norm(x)


class TransformerDecoderLayer(Module):
    """Causal self-attention, cross-attention, and feed-forward residual blocks."""

    def __init__(self, features, num_heads, hidden_size=None, dropout=0.0):
        super().__init__()
        hidden_size = 4 * features if hidden_size is None else hidden_size
        self.norm1, self.norm2, self.norm3 = [LayerNorm(features) for _ in range(3)]
        self.self_attention = MultiheadAttention(features, num_heads, dropout)
        self.cross_attention = MultiheadAttention(features, num_heads, dropout)
        self.feed_forward = FeedForward(features, hidden_size, dropout)
        self.dropout = Dropout(dropout)

    def forward(self, x, memory, target_padding_mask=None, memory_padding_mask=None):
        x = x + self.dropout(
            self.self_attention(self.norm1(x), is_causal=True, key_padding_mask=target_padding_mask)
        )
        x = x + self.dropout(
            self.cross_attention(
                self.norm2(x), memory, memory, key_padding_mask=memory_padding_mask
            )
        )
        return x + self.feed_forward(self.norm3(x))


class TransformerDecoder(Module):
    def __init__(self, features, num_heads, num_layers=1, hidden_size=None, dropout=0.0):
        super().__init__()
        _positive_int(num_layers, "num_layers")
        self.layers = ModuleList(
            TransformerDecoderLayer(features, num_heads, hidden_size, dropout)
            for _ in range(num_layers)
        )
        self.norm = LayerNorm(features)

    def forward(self, x, memory, target_padding_mask=None, memory_padding_mask=None):
        for layer in self.layers:
            x = layer(x, memory, target_padding_mask, memory_padding_mask)
        return self.norm(x)


class Transformer(Module):
    """Encoder-decoder over feature tensors; embeddings are supplied by callers."""

    def __init__(
        self, features, num_heads, encoder_layers=1, decoder_layers=1, hidden_size=None, dropout=0.0
    ):
        super().__init__()
        self.encoder = TransformerEncoder(features, num_heads, encoder_layers, hidden_size, dropout)
        self.decoder = TransformerDecoder(features, num_heads, decoder_layers, hidden_size, dropout)

    def forward(self, source, target, source_padding_mask=None, target_padding_mask=None):
        memory = self.encoder(source, padding_mask=source_padding_mask)
        return self.decoder(target, memory, target_padding_mask, source_padding_mask)


class CausalBlock(Module):
    """A decoder-only block, optionally using the modern RMSNorm/SwiGLU recipe."""

    def __init__(
        self, features, num_heads, hidden_size, dropout=0.0, modern=False, num_kv_heads=None
    ):
        super().__init__()
        norm = RMSNorm if modern else LayerNorm
        self.norm1, self.norm2 = norm(features), norm(features)
        self.attention = MultiheadAttention(
            features, num_heads, dropout, num_kv_heads, rope=modern, bias=not modern
        )
        self.feed_forward = FeedForward(
            features, hidden_size, dropout, "swiglu" if modern else "gelu", bias=not modern
        )
        self.dropout = Dropout(dropout)

    def forward(self, x, padding_mask=None, cache=None, use_cache=False):
        attention = self.attention(
            self.norm1(x),
            is_causal=True,
            key_padding_mask=padding_mask,
            cache=cache,
            use_cache=use_cache,
        )
        if use_cache:
            attention, cache = attention
        x = x + self.dropout(attention)
        x = x + self.feed_forward(self.norm2(x))
        return (x, cache) if use_cache else x
