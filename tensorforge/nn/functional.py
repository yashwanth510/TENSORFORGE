"""Loss functions composed from differentiable Tensor operations."""

import numpy as np

from ..tensor import Tensor, _as_tensor


def _reduce(loss, reduction):
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'.")


def mse_loss(prediction, target, reduction="mean"):
    """Squared error with equal shapes required to catch accidental broadcasts."""
    target = _as_tensor(target)
    if prediction.shape != target.shape:
        raise ValueError("MSE prediction and target shapes must match.")
    return _reduce((prediction - target) ** 2, reduction)


def cross_entropy(logits, target, reduction="mean", ignore_index=-100, label_smoothing=0.0):
    """Cross-entropy for (..., classes) logits and matching integer targets.

    Ignored targets contribute zero; a fully ignored batch returns zero.
    Label smoothing mixes the target with a uniform class distribution.
    """
    target = target.numpy() if isinstance(target, Tensor) else np.asarray(target)
    if logits.ndim < 2 or logits.size == 0:
        raise ValueError("logits need nonempty sample axes and a final class axis.")
    if target.shape != logits.shape[:-1] or target.dtype.kind not in "iu":
        raise ValueError("target must contain one integer class index per sample.")
    if not 0 <= label_smoothing <= 1:
        raise ValueError("label_smoothing must lie in [0, 1].")
    classes = logits.shape[-1]
    flat_target = target.reshape(-1)
    valid = flat_target != ignore_index
    if np.any((flat_target[valid] < 0) | (flat_target[valid] >= classes)):
        raise ValueError("A target class index is out of range.")
    safe_target = np.where(valid, flat_target, 0)
    log_probs = logits.reshape(-1, classes).log_softmax(axis=-1)
    selected = -log_probs[np.arange(len(flat_target)), safe_target]
    smooth = -log_probs.mean(axis=-1)
    loss = ((1 - label_smoothing) * selected + label_smoothing * smooth) * valid
    loss = loss.reshape(target.shape)
    if reduction == "mean":
        return loss.sum() / max(int(valid.sum()), 1)
    return _reduce(loss, reduction)


def l1_loss(prediction, target, reduction="mean"):
    target = _as_tensor(target)
    if prediction.shape != target.shape:
        raise ValueError("L1 prediction and target shapes must match.")
    return _reduce((prediction - target).abs(), reduction)


def binary_cross_entropy_with_logits(logits, target, reduction="mean"):
    """Stable binary loss: max(x,0) - x*y + log(1 + exp(-abs(x)))."""
    target = _as_tensor(target)
    if logits.shape != target.shape:
        raise ValueError("Binary logits and target shapes must match.")
    if np.any((target._data < 0) | (target._data > 1)):
        raise ValueError("Binary targets must lie in [0, 1].")
    # logaddexp(0,x) is a stable log(1+exp(x)), including at extreme x.
    data = np.logaddexp(0, logits._data) - logits._data * target._data
    probability = logits.sigmoid()._data
    loss = Tensor._op(
        data, (logits, target), lambda g: (g * (probability - target._data), -g * logits._data)
    )
    return _reduce(loss, reduction)


def relu(x):
    return x.relu()


def sigmoid(x):
    return x.sigmoid()


def tanh(x):
    return x.tanh()


def softmax(x, axis=-1):
    return x.softmax(axis=axis)


def log_softmax(x, axis=-1):
    return x.log_softmax(axis=axis)


def scaled_dot_product_attention(
    query, key, value, mask=None, is_causal=False, dropout_p=0.0, training=False, query_offset=0
):
    """Attention for (..., query_length, width) tensors.

    Boolean masks use True to BLOCK an entry throughout TensorForge. A mask
    must broadcast to (..., query_length, key_length). Fully masked rows
    produce zeros. query_offset aligns causal positions during cached decoding.
    """
    import importlib

    core = importlib.import_module("tensorforge.tensor")
    if min(query.ndim, key.ndim, value.ndim) < 2:
        raise ValueError("Attention inputs need sequence and feature axes.")
    if query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
        raise ValueError("Attention key/query widths and key/value lengths must match.")
    if query.shape[-1] == 0 or key.shape[-2] == 0:
        raise ValueError("Attention width and key length must be nonzero.")
    if not 0 <= dropout_p < 1 or type(query_offset) is not int or query_offset < 0:
        raise ValueError("Invalid dropout probability or query offset.")
    scores = (query @ key.transpose(-2, -1)) / np.sqrt(query.shape[-1])
    blocked = np.zeros(scores.shape, dtype=bool)
    if mask is not None:
        mask = mask.numpy() if isinstance(mask, Tensor) else np.asarray(mask)
        if mask.dtype.kind != "b":
            raise TypeError("Attention mask must be boolean; True blocks attention.")
        blocked |= np.broadcast_to(mask, scores.shape)
    if is_causal:
        positions = np.arange(query.shape[-2]) + query_offset
        blocked |= np.arange(key.shape[-2])[None, :] > positions[:, None]
    masked = scores.masked_fill(blocked, -np.inf)
    shift = masked._data.max(axis=-1, keepdims=True)
    shift = np.where(np.isfinite(shift), shift, 0)
    exponentials = (masked - Tensor(shift)).exp()
    total = exponentials.sum(axis=-1, keepdims=True)
    weights = exponentials / (total + Tensor(total._data == 0))
    if training and dropout_p:
        keep = core._rng.random(weights.shape) >= dropout_p
        weights = weights * Tensor(keep / (1 - dropout_p))
    return weights @ value
