"""Small configurable causal language models and text generation."""

import importlib
from dataclasses import asdict, dataclass

import numpy as np

from .nn.modules import Dropout, Embedding, LayerNorm, Module, ModuleList, RMSNorm, _positive_int
from .nn.transformer import CausalBlock, LearnedPositionEncoding
from .tensor import Tensor, cat, no_grad

_core = importlib.import_module("tensorforge.tensor")


@dataclass(frozen=True)
class LanguageModelConfig:
    vocab_size: int
    features: int = 64
    num_heads: int = 4
    num_layers: int = 2
    hidden_size: int = 128
    max_length: int = 256
    dropout: float = 0.0
    variant: str = "gpt"
    num_kv_heads: int | None = None

    def __post_init__(self):
        for name in (
            "vocab_size",
            "features",
            "num_heads",
            "num_layers",
            "hidden_size",
            "max_length",
        ):
            _positive_int(getattr(self, name), name)
        if self.variant not in ("gpt", "modern"):
            raise ValueError("variant must be 'gpt' or 'modern'.")
        heads = self.num_heads if self.num_kv_heads is None else self.num_kv_heads
        _positive_int(heads, "num_kv_heads")
        if self.features % self.num_heads or self.num_heads % heads:
            raise ValueError("features must divide by heads; heads must divide by kv heads.")
        if self.variant == "modern" and (self.features // self.num_heads) % 2:
            raise ValueError("Modern rotary attention needs an even head width.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")

    def to_dict(self):
        return asdict(self)


class CausalLanguageModel(Module):
    """GPT-style or modern decoder with tied input/output token embeddings.

    gpt: learned positions, LayerNorm, GELU.
    modern: rotary positions, RMSNorm, SwiGLU, optional grouped-query heads.
    These are local architectures, not pretrained-weight compatibility layers.
    """

    def __init__(self, config):
        super().__init__()
        if not isinstance(config, LanguageModelConfig):
            raise TypeError("config must be a LanguageModelConfig.")
        self.config = config
        modern = config.variant == "modern"
        self.embedding = Embedding(config.vocab_size, config.features)
        self.position = (
            None if modern else LearnedPositionEncoding(config.features, config.max_length)
        )
        self.dropout = Dropout(config.dropout)
        self.layers = ModuleList(
            CausalBlock(
                config.features,
                config.num_heads,
                config.hidden_size,
                config.dropout,
                modern,
                config.num_kv_heads,
            )
            for _ in range(config.num_layers)
        )
        self.norm = RMSNorm(config.features) if modern else LayerNorm(config.features)

    def forward(self, tokens, padding_mask=None, cache=None, use_cache=False):
        if not isinstance(tokens, Tensor):
            raw = np.asarray(tokens)
            if raw.dtype.kind not in "iu":
                raise TypeError("Token IDs must be integers.")
            tokens = Tensor(raw, dtype=np.int64)
        if tokens.ndim != 2 or tokens.dtype.kind not in "iu" or tokens.shape[1] == 0:
            raise ValueError("tokens must be a nonempty (batch,time) integer tensor.")
        offset = 0
        if cache is not None:
            if not isinstance(cache, (tuple, list)) or len(cache) != len(self.layers):
                raise ValueError("One key/value cache pair is required per layer.")
            for pair in cache:
                if (
                    not isinstance(pair, (tuple, list))
                    or len(pair) != 2
                    or not all(isinstance(t, Tensor) and t.ndim == 4 for t in pair)
                ):
                    raise ValueError("Invalid layer cache.")
            lengths = [pair[0].shape[2] for pair in cache]
            if len(set(lengths)) != 1:
                raise ValueError("All layer caches must have equal sequence lengths.")
            offset = lengths[0]
        if offset + tokens.shape[1] > self.config.max_length:
            raise ValueError("Input and cached tokens exceed max_length.")
        x = self.embedding(tokens)
        if self.position is not None:
            x = self.position(x, offset)
        x = self.dropout(x)
        updated = []
        for index, layer in enumerate(self.layers):
            previous = None if cache is None else cache[index]
            result = layer(x, padding_mask, previous, use_cache)
            if use_cache:
                x, pair = result
                updated.append(pair)
            else:
                x = result
        logits = self.norm(x) @ self.embedding.weight.T
        return (logits, updated) if use_cache else logits

    def generate(
        self,
        tokens,
        max_new_tokens=20,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        eos_token_id=None,
        use_cache=True,
        seed=None,
    ):
        """Generate a fixed batch of unpadded prompts. Temperature zero is greedy.

        All requested positions must fit max_length. Existing per-module modes
        are restored after generation. Returned token IDs include the prompt.
        """
        raw = tokens.numpy() if isinstance(tokens, Tensor) else np.asarray(tokens)
        if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0 or raw.dtype.kind not in "iu":
            raise ValueError("Generation needs nonempty (batch,time) integer prompts.")
        if np.any(raw < 0) or np.any(raw >= self.config.vocab_size):
            raise ValueError("Prompt token is outside the vocabulary.")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer.")
        if raw.shape[1] + max_new_tokens > self.config.max_length:
            raise ValueError("Requested generation exceeds max_length.")
        if not np.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("temperature must be nonnegative and top_p must lie in (0,1].")
        if top_k is not None and (
            type(top_k) is not int or not 1 <= top_k <= self.config.vocab_size
        ):
            raise ValueError("top_k must be between 1 and vocab_size.")
        if eos_token_id is not None and (
            type(eos_token_id) is not int or not 0 <= eos_token_id < self.config.vocab_size
        ):
            raise ValueError("eos_token_id must be a vocabulary index.")
        rng = _core._rng if seed is None else np.random.default_rng(seed)
        modes = [(m, m.training) for _, m in self._walk() if isinstance(m, Module)]
        generated = Tensor(raw, dtype=np.int64)
        finished = np.zeros(raw.shape[0], dtype=bool)
        cache = None
        self.eval()
        try:
            with no_grad():
                for _ in range(max_new_tokens):
                    if use_cache:
                        current = generated if cache is None else generated[:, -1:]
                        logits, cache = self(current, cache=cache, use_cache=True)
                    else:
                        logits = self(generated)
                    scores = logits._data[:, -1, :].copy()
                    if not np.isfinite(scores).all():
                        raise ValueError("Generation encountered non-finite logits.")
                    if temperature == 0:
                        next_ids = scores.argmax(axis=-1)
                    else:
                        # Shift before scaling so a tiny temperature cannot
                        # overflow a positive maximum into infinity.
                        scores = (scores - scores.max(axis=-1, keepdims=True)) / temperature
                        next_ids = []
                        for row in scores:
                            indices = np.argsort(-row, kind="stable")
                            if top_k is not None:
                                indices = indices[:top_k]
                            probabilities = np.exp(row[indices] - row[indices].max())
                            probabilities /= probabilities.sum()
                            keep = np.cumsum(probabilities) - probabilities < top_p
                            indices, probabilities = indices[keep], probabilities[keep]
                            probabilities /= probabilities.sum()
                            next_ids.append(rng.choice(indices, p=probabilities))
                        next_ids = np.asarray(next_ids, dtype=np.int64)
                    if eos_token_id is not None:
                        next_ids = np.where(finished, eos_token_id, next_ids)
                        finished |= next_ids == eos_token_id
                    generated = cat((generated, Tensor(next_ids[:, None], dtype=np.int64)), axis=1)
                    if eos_token_id is not None and finished.all():
                        break
        finally:
            for module, mode in modes:
                module.training = mode
        return generated
