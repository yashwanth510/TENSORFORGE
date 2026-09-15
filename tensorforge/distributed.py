"""Synchronous local CPU data parallelism using Python worker processes.

Models and loss functions must be importable/picklable, and batches must have
matching non-batch shapes. This is not multi-node or GPU distributed training.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .tensor import Tensor, manual_seed


def _worker(model, arrays, loss_fn, seed):
    manual_seed(seed)
    before = model.state_dict()
    model.zero_grad()
    x, y = (Tensor(a, dtype=a.dtype) for a in arrays)
    loss = loss_fn(model(x), y)
    loss.backward()
    for name, buffer in model.named_buffers():
        if not np.array_equal(before[name], buffer._data):
            raise ValueError("State-changing buffers cannot be synchronized by this helper.")
    gradients = {name: p.grad for name, p in model.named_parameters()}
    return loss.item(), gradients


def data_parallel_backward(model, batches, loss_fn, workers=2, seed=0):
    """Accumulate sample-weighted mean gradients from local process replicas.

    loss_fn must return a mean loss with equal per-sample normalization. This
    helper does not handle unequal token counts in masked sequence losses.
    Call optimizer.step() in the parent process after this function returns.
    """
    batches = list(batches)
    if type(workers) is not int or workers < 1 or not batches:
        raise ValueError("Need positive workers and at least one batch.")
    arrays = []
    for x, y in batches:
        if (
            not isinstance(x, Tensor)
            or not isinstance(y, Tensor)
            or x.ndim == 0
            or y.ndim == 0
            or len(x) == 0
            or len(x) != len(y)
        ):
            raise ValueError("Each batch needs matching nonempty tensor sample axes.")
        arrays.append((x.numpy(), y.numpy()))
    if any(
        a.shape[1:] != arrays[0][0].shape[1:] or b.shape[1:] != arrays[0][1].shape[1:]
        for a, b in arrays
    ):
        raise ValueError("All non-batch shapes must match.")
    versions = {name: p._version for name, p in model.named_parameters()}
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(_worker, model, items, loss_fn, seed + i) for i, items in enumerate(arrays)
        ]
        results = [future.result() for future in futures]
    total = sum(len(a) for a, _ in arrays)
    parameters = dict(model.named_parameters())
    if any(p._version != versions[name] for name, p in parameters.items()):
        raise RuntimeError("Model changed during parallel computation.")
    gradients = {name: None for name in parameters}
    loss = 0.0
    for (value, worker_gradients), (x, _) in zip(results, arrays):
        weight = len(x) / total
        loss += value * weight
        for name, gradient in worker_gradients.items():
            if gradient is not None:
                gradients[name] = (
                    gradient * weight
                    if gradients[name] is None
                    else gradients[name] + gradient * weight
                )
    for name, p in parameters.items():
        if p.requires_grad and gradients[name] is not None:
            p.grad = gradients[name] if p.grad is None else p.grad + gradients[name]
    return loss
