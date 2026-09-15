"""Masked-token, image, multimodal, and sparse-expert model components."""

import numpy as np

from ..tensor import Tensor, cat
from .modules import Embedding, LayerNorm, Linear, Module, ModuleList, Parameter
from .transformer import FeedForward, LearnedPositionEncoding, TransformerEncoder


class MaskedLanguageModel(Module):
    """Bidirectional encoder trained to recover selected input tokens.

    BERT-style objective support, not pretrained BERT architecture compatibility.
    """

    def __init__(self, vocab_size, features=32, num_heads=4, num_layers=2, max_length=128):
        super().__init__()
        self.embedding = Embedding(vocab_size, features)
        self.segment = Embedding(2, features)
        self.position = LearnedPositionEncoding(features, max_length)
        self.input_norm = LayerNorm(features)
        self.encoder = TransformerEncoder(features, num_heads, num_layers)
        self.transform = Linear(features, features)
        self.output_norm = LayerNorm(features)
        self.bias = Parameter(np.zeros(vocab_size))

    def forward(self, tokens, padding_mask=None, token_types=None):
        if not isinstance(tokens, Tensor):
            raw = np.asarray(tokens)
            if raw.dtype.kind not in "iu":
                raise TypeError("Token IDs must be integers.")
            tokens = Tensor(raw, dtype=np.int64)
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch,time).")
        if token_types is None:
            token_types = np.zeros(tokens.shape, dtype=np.int64)
        x = self.position(self.embedding(tokens) + self.segment(token_types))
        x = self.encoder(self.input_norm(x), padding_mask=padding_mask)
        return self.output_norm(self.transform(x).gelu()) @ self.embedding.weight.T + self.bias


class PatchEmbedding(Module):
    def __init__(self, channels, patch_size, features):
        super().__init__()
        if type(patch_size) is not int or patch_size < 1:
            raise ValueError("patch_size must be a positive integer.")
        self.channels, self.patch_size = channels, patch_size
        self.projection = Linear(channels * patch_size * patch_size, features)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != self.channels:
            raise ValueError("Images must have shape (batch,channels,height,width).")
        b, c, h, w = images.shape
        p = self.patch_size
        if h == 0 or w == 0 or h % p or w % p:
            raise ValueError("Image dimensions must be positive multiples of patch_size.")
        patches = images.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
        return self.projection(patches.reshape(b, (h // p) * (w // p), c * p * p))


class VisionTransformer(Module):
    def __init__(
        self, image_size, patch_size, channels, classes, features=32, num_heads=4, num_layers=1
    ):
        super().__init__()
        if (
            type(image_size) is not int
            or type(patch_size) is not int
            or patch_size <= 0
            or image_size <= 0
            or image_size % patch_size
        ):
            raise ValueError("image_size must be a positive multiple of patch_size.")
        self.image_size = image_size
        self.patches = PatchEmbedding(channels, patch_size, features)
        self.cls = Parameter(np.zeros((1, 1, features)))
        self.position = LearnedPositionEncoding(features, (image_size // patch_size) ** 2 + 1)
        self.encoder = TransformerEncoder(features, num_heads, num_layers)
        self.head = Linear(features, classes)

    def forward(self, images):
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("Image dimensions do not match this model.")
        patches = self.patches(images)
        x = cat((self.cls.expand(len(images), 1, self.cls.shape[-1]), patches), axis=1)
        return self.head(self.encoder(self.position(x))[:, 0])


class MultimodalClassifier(Module):
    """Joint image-patch and text encoder for paired-input classification."""

    def __init__(
        self, vocab_size, channels, patch_size, classes, features=32, num_heads=4, max_length=128
    ):
        super().__init__()
        self.patches = PatchEmbedding(channels, patch_size, features)
        self.tokens = Embedding(vocab_size, features)
        self.modality = Embedding(2, features)
        self.cls = Parameter(np.zeros((1, 1, features)))
        self.position = LearnedPositionEncoding(features, max_length)
        self.encoder = TransformerEncoder(features, num_heads)
        self.head = Linear(features, classes)

    def forward(self, images, tokens, text_padding_mask=None):
        patch = self.patches(images) + self.modality(np.array(0, dtype=np.int64))
        text = self.tokens(tokens) + self.modality(np.array(1, dtype=np.int64))
        if text.ndim != 3 or len(text) != len(images):
            raise ValueError("Text and image batch sizes must match.")
        x = cat((self.cls.expand(len(images), 1, self.cls.shape[-1]), patch, text), axis=1)
        mask = None
        if text_padding_mask is not None:
            m = (
                text_padding_mask.numpy()
                if isinstance(text_padding_mask, Tensor)
                else np.asarray(text_padding_mask)
            )
            if m.dtype.kind != "b" or m.shape != text.shape[:2]:
                raise ValueError("Invalid text padding mask.")
            mask = np.concatenate(
                (np.zeros((len(images), 1 + patch.shape[1]), dtype=bool), m), axis=1
            )
        return self.head(self.encoder(self.position(x), padding_mask=mask)[:, 0])


class MixtureOfExperts(Module):
    """Token routing to selected feed-forward experts, with a balancing loss.

    No expert capacity limit or distributed expert placement. Top-one uses
    its original gate probability; multiple selected weights are renormalized.
    """

    def __init__(self, features, hidden_size, num_experts=4, top_k=2):
        super().__init__()
        if type(num_experts) is not int or type(top_k) is not int or not 1 <= top_k <= num_experts:
            raise ValueError("Require 1 <= top_k <= num_experts.")
        self.features, self.num_experts, self.top_k = features, num_experts, top_k
        self.router = Linear(features, num_experts, bias=False)
        self.experts = ModuleList(
            FeedForward(features, hidden_size, activation="swiglu") for _ in range(num_experts)
        )

    def forward(self, x, return_aux_loss=False):
        if x.shape[-1] != self.features or x.size == 0:
            raise ValueError("Nonempty expert inputs must match features.")
        flat = x.reshape(-1, self.features)
        probabilities = self.router(flat).softmax()
        chosen = np.argsort(-probabilities._data, axis=-1, kind="stable")[:, : self.top_k]
        routing = np.zeros(probabilities.shape)
        np.put_along_axis(routing, chosen, 1, axis=1)
        weights = probabilities * routing
        if self.top_k > 1:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        result = Tensor(np.zeros(flat.shape))
        for index, expert in enumerate(self.experts):
            rows = np.flatnonzero(routing[:, index])
            if not rows.size:
                continue
            values = expert(flat[rows]) * weights[rows, index].unsqueeze(-1)
            scattered = np.zeros(flat.shape)
            scattered[rows] = values._data

            def backward(g, rows=rows):
                return (g[rows],)

            result = result + Tensor._op(scattered, (values,), backward)
        result = result.reshape(x.shape)
        frequency = Tensor(routing.mean(axis=0) / self.top_k)
        auxiliary = self.num_experts * (probabilities.mean(axis=0) * frequency).sum()
        return (result, auxiliary) if return_aux_loss else result
