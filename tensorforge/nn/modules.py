"""Composable neural-network layers; each forward pass uses our Tensor engine."""

import importlib

import numpy as np

from ..tensor import Tensor
from . import functional as F

# Import the module rather than a snapshot of its random generator.
_core = importlib.import_module("tensorforge.tensor")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


class Parameter(Tensor):
    """A trainable Tensor discovered automatically by Module.parameters()."""

    def __init__(self, data, dtype=None):
        super().__init__(data, requires_grad=True, dtype=dtype)


class _Buffer(Tensor):
    """Persistent non-trainable state, such as a running variance."""


class Module:
    """Base class: subclass it and implement forward(self, ...)."""

    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        # model(x) invokes model.forward(x).
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward().")

    def _walk(self):
        # Walk modules, lists, tuples, and dictionaries, avoiding shared nodes
        # and cycles. Names become stable checkpoint keys such as layers.0.weight.
        seen = set()

        def visit(value, name):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, (Module, Parameter, _Buffer)):
                yield name, value
            if isinstance(value, Module):
                children = vars(value).items()
            elif isinstance(value, dict):
                children = value.items()
            elif isinstance(value, (list, tuple)):
                children = enumerate(value)
            else:
                return
            for key, child in children:
                path = f"{name}.{key}" if name else str(key)
                yield from visit(child, path)

        yield from visit(self, "")

    def named_parameters(self):
        return [(name, value) for name, value in self._walk() if isinstance(value, Parameter)]

    def parameters(self):
        return [value for _, value in self.named_parameters()]

    def register_buffer(self, name, tensor):
        if not isinstance(name, str) or not name.isidentifier() or hasattr(self, name):
            raise ValueError("A buffer needs a new valid attribute name.")
        if not isinstance(tensor, Tensor) or tensor.requires_grad:
            raise TypeError("Buffers must be tensors without gradients.")
        setattr(self, name, _Buffer(tensor._data, dtype=tensor.dtype))

    def named_buffers(self):
        return [(name, value) for name, value in self._walk() if isinstance(value, _Buffer)]

    def buffers(self):
        return [value for _, value in self.named_buffers()]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self, mode=True):
        for _, value in self._walk():
            if isinstance(value, Module):
                value.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        """Independent copies of named parameter and buffer arrays."""
        return {
            name: value.numpy() for name, value in self.named_parameters() + self.named_buffers()
        }

    def load_state_dict(self, state):
        """Strict load: validate every key and shape before updating anything."""
        parameters = dict(self.named_parameters() + self.named_buffers())
        if parameters.keys() != state.keys():
            raise ValueError("Checkpoint keys do not match model parameters.")
        validated = {}
        for name, parameter in parameters.items():
            data = np.asarray(state[name])
            if data.shape != parameter.shape:
                raise ValueError(f"Shape mismatch for parameter {name}.")
            if data.dtype.kind != parameter.dtype.kind or not np.isfinite(data).all():
                raise ValueError(f"State {name} must contain finite values of the correct kind.")
            converted = data.astype(parameter.dtype)
            if not np.isfinite(converted).all():
                raise ValueError(f"State {name} overflows the destination dtype.")
            validated[name] = converted
        for name, parameter in parameters.items():
            parameter._assign(validated[name])
            parameter.zero_grad()
        return self

    def __repr__(self):
        count = sum(p.size for p in self.parameters())
        return f"{type(self).__name__}(parameters={count}, training={self.training})"


class Linear(Module):
    """Affine map y = x @ weight.T + bias, preserving leading batch axes."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        _positive_int(in_features, "in_features")
        _positive_int(out_features, "out_features")
        # Small symmetric weights break symmetry without huge initial outputs.
        bound = 1 / np.sqrt(in_features)
        self.weight = Parameter(_core._rng.uniform(-bound, bound, (out_features, in_features)))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        result = x @ self.weight.T
        return result if self.bias is None else result + self.bias


class Sequential(Module):
    """Pass each layer's output to the next layer."""

    def __init__(self, *layers):
        super().__init__()
        if not all(isinstance(layer, Module) for layer in layers):
            raise TypeError("Sequential accepts Module instances only.")
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, index):
        return self.layers[index]


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Flatten(Module):
    """Preserve leading axes, combine start_dim and all following axes."""

    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        if not isinstance(self.start_dim, int) or not 0 <= self.start_dim < x.ndim:
            raise ValueError("start_dim must identify an existing nonnegative axis.")
        size = int(np.prod(x.shape[self.start_dim :]))
        return x.reshape(*x.shape[: self.start_dim], size)


class Dropout(Module):
    """Randomly drop entries during training; use identity in evaluation."""

    def __init__(self, p=0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout p must satisfy 0 <= p < 1.")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # Dividing by keep probability preserves the expected activation.
        mask = (_core._rng.random(x.shape) >= self.p) / (1 - self.p)
        return x * Tensor(mask, dtype=x.dtype)


class LayerNorm(Module):
    """Normalize the final feature axis, then learn a scale and shift."""

    def __init__(self, features, eps=1e-5):
        super().__init__()
        _positive_int(features, "features")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        self.features, self.eps = features, eps
        self.weight = Parameter(np.ones(features))
        self.bias = Parameter(np.zeros(features))

    def forward(self, x):
        if x.ndim == 0 or x.shape[-1] != self.features:
            raise ValueError("The final input axis must match LayerNorm features.")
        centered = x - x.mean(axis=-1, keepdims=True)
        variance = (centered**2).mean(axis=-1, keepdims=True)
        return centered / ((variance + self.eps) ** 0.5) * self.weight + self.bias


class Embedding(Module):
    """Look up trainable rows using integer token IDs."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        _positive_int(num_embeddings, "num_embeddings")
        _positive_int(embedding_dim, "embedding_dim")
        self.weight = Parameter(_core._rng.standard_normal((num_embeddings, embedding_dim)) * 0.02)

    def forward(self, indices):
        indices = indices.numpy() if isinstance(indices, Tensor) else np.asarray(indices)
        if indices.dtype.kind not in "iu":
            raise TypeError("Embedding indices must be integers.")
        if np.any(indices < 0) or np.any(indices >= self.weight.shape[0]):
            raise ValueError("Embedding index out of range.")
        return self.weight[indices]


class MSELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction, target):
        return F.mse_loss(prediction, target, self.reduction)


class CrossEntropyLoss(Module):
    def __init__(self, reduction="mean", ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        return F.cross_entropy(
            logits, target, self.reduction, self.ignore_index, self.label_smoothing
        )


class BCEWithLogitsLoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits, target, self.reduction)


class ModuleList(Module):
    """An iterable, indexed collection of registered modules."""

    def __init__(self, modules=()):
        super().__init__()
        self.items = []
        for module in modules:
            self.append(module)

    def append(self, module):
        if not isinstance(module, Module):
            raise TypeError("ModuleList accepts Module instances only.")
        self.items.append(module)
        return self

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index):
        return self.items[index]


class ModuleDict(Module):
    """A named collection of registered modules."""

    def __init__(self, modules=None):
        super().__init__()
        self.items = {}
        for name, module in (modules or {}).items():
            self[name] = module

    def __setitem__(self, name, module):
        if not isinstance(name, str) or not name.isidentifier() or not isinstance(module, Module):
            raise TypeError("ModuleDict needs valid string names and Module values.")
        self.items[name] = module

    def __getitem__(self, name):
        return self.items[name]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class Identity(Module):
    def forward(self, x):
        return x


class GELU(Module):
    def forward(self, x):
        return x.gelu()


class SiLU(Module):
    def forward(self, x):
        return x.silu()


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        if not np.isfinite(negative_slope) or negative_slope < 0:
            raise ValueError("negative_slope must be finite and nonnegative.")
        self.negative_slope = negative_slope

    def forward(self, x):
        from ..tensor import where

        return where(x._data > 0, x, x * self.negative_slope)


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return x.softmax(self.axis)


class RMSNorm(Module):
    """Normalize by the root mean square of the final feature axis."""

    def __init__(self, features, eps=1e-6):
        super().__init__()
        _positive_int(features, "features")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        self.features, self.eps = features, eps
        self.weight = Parameter(np.ones(features))

    def forward(self, x):
        if x.ndim == 0 or x.shape[-1] != self.features:
            raise ValueError("Input features do not match RMSNorm.")
        return x * (((x * x).mean(axis=-1, keepdims=True) + self.eps) ** -0.5) * self.weight


class BatchNorm1d(Module):
    """Normalize NC or NCL tensors over batch and optional length axes."""

    def __init__(self, features, eps=1e-5, momentum=0.1):
        super().__init__()
        _positive_int(features, "features")
        if not np.isfinite(eps) or eps <= 0 or not 0 <= momentum <= 1:
            raise ValueError("eps must be positive and momentum must lie in [0, 1].")
        self.features, self.eps, self.momentum = features, eps, momentum
        self.weight, self.bias = Parameter(np.ones(features)), Parameter(np.zeros(features))
        self.register_buffer("running_mean", Tensor(np.zeros(features)))
        self.register_buffer("running_var", Tensor(np.ones(features)))
        self.register_buffer("num_batches_tracked", Tensor(0, dtype=np.int64))

    def forward(self, x):
        if x.ndim not in (2, 3) or x.shape[1] != self.features:
            raise ValueError("BatchNorm1d expects (N,C) or (N,C,L) with matching C.")
        axes = (0,) if x.ndim == 2 else (0, 2)
        shape = (1, self.features) + (1,) * (x.ndim - 2)
        count = int(np.prod([x.shape[a] for a in axes]))
        if self.training:
            if count <= 1:
                raise ValueError("Training BatchNorm requires at least two values per channel.")
            mean = x.mean(axis=axes, keepdims=True)
            centered = x - mean
            variance = (centered * centered).mean(axis=axes, keepdims=True)
            rate = self.momentum
            self.running_mean._assign(
                (1 - rate) * self.running_mean._data + rate * mean._data.reshape(-1)
            )
            # Normalize using population variance; save unbiased running variance.
            self.running_var._assign(
                (1 - rate) * self.running_var._data
                + rate * variance._data.reshape(-1) * count / (count - 1)
            )
            self.num_batches_tracked._assign(self.num_batches_tracked._data + 1)
        else:
            centered = x - self.running_mean.reshape(shape)
            variance = self.running_var.reshape(shape)
        return centered * ((variance + self.eps) ** -0.5) * self.weight.reshape(
            shape
        ) + self.bias.reshape(shape)


class L1Loss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction, target):
        return F.l1_loss(prediction, target, self.reduction)


class BatchNorm2d(BatchNorm1d):
    """Normalize NCHW images over batch and spatial positions per channel."""

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("BatchNorm2d expects NCHW input.")
        flattened = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        return super().forward(flattened).reshape(x.shape)
