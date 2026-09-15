"""Held-out loss, perplexity, and simple classification evaluation."""

import math
from contextlib import contextmanager

import numpy as np

from .nn.functional import cross_entropy
from .nn.modules import Module
from .tensor import Tensor, no_grad


@contextmanager
def evaluating(model):
    modes = [(m, m.training) for _, m in model._walk() if isinstance(m, Module)]
    model.eval()
    try:
        with no_grad():
            yield model
    finally:
        for module, mode in modes:
            module.training = mode


def evaluate_language_model(model, batches, ignore_index=-100):
    """Token-weighted loss and perplexity; batches yield (inputs, targets)."""
    total_loss, total_tokens = 0.0, 0
    with evaluating(model):
        for inputs, targets in batches:
            ids = targets.numpy() if isinstance(targets, Tensor) else np.asarray(targets)
            count = int((ids != ignore_index).sum())
            logits = model(inputs)
            total_loss += cross_entropy(
                logits, targets, reduction="sum", ignore_index=ignore_index
            ).item()
            total_tokens += count
    if total_tokens == 0:
        raise ValueError("Evaluation needs at least one non-ignored target.")
    loss = total_loss / total_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 709 else math.inf,
        "tokens": total_tokens,
    }


def classification_accuracy(logits, targets):
    logits = logits.numpy() if isinstance(logits, Tensor) else np.asarray(logits)
    targets = targets.numpy() if isinstance(targets, Tensor) else np.asarray(targets)
    if logits.ndim < 2 or targets.shape != logits.shape[:-1] or targets.size == 0:
        raise ValueError("Provide nonempty logits and matching targets.")
    return float(np.mean(logits.argmax(axis=-1) == targets))
