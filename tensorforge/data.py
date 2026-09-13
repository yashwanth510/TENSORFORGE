"""In-memory datasets and deterministic single-process mini-batching."""

import numpy as np

from .tensor import Tensor


class Dataset:
    """Override __len__ and __getitem__ to define a custom indexed dataset."""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        if not tensors or not all(isinstance(t, Tensor) and t.ndim > 0 for t in tensors):
            raise ValueError("TensorDataset needs tensors with a sample axis.")
        if any(len(t) != len(tensors[0]) for t in tensors):
            raise ValueError("Dataset tensors must have equal sample counts.")
        if any(t.requires_grad for t in tensors):
            raise ValueError("Dataset storage must not require gradients.")
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)


class DataLoader:
    """Single-process batches with optional custom collation and resumable order.

    Iteration resumes an unfinished epoch; after exhaustion a new iteration
    starts a new epoch. Concurrent iterators over one loader are unsupported.
    """

    def __init__(
        self, dataset, batch_size=1, shuffle=False, drop_last=False, seed=None, collate_fn=None
    ):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if collate_fn is not None and not callable(collate_fn):
            raise TypeError("collate_fn must be callable.")
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.drop_last = bool(shuffle), bool(drop_last)
        self.collate_fn = collate_fn
        self._rng = np.random.default_rng(seed)
        self._order, self._cursor = None, 0

    def __len__(self):
        size = len(self.dataset)
        return (
            size // self.batch_size
            if self.drop_last
            else (size + self.batch_size - 1) // self.batch_size
        )

    def _limit(self):
        return len(self) * self.batch_size if self.drop_last else len(self.dataset)

    def __iter__(self):
        if self._order is None or self._cursor >= self._limit():
            self._order = np.arange(len(self.dataset))
            if self.shuffle:
                self._rng.shuffle(self._order)
            self._cursor = 0
        return self

    def __next__(self):
        if self._order is None:
            self.__iter__()
        if self._cursor >= self._limit():
            raise StopIteration
        stop = min(self._cursor + self.batch_size, self._limit())
        indices = self._order[self._cursor : stop]
        samples = [self.dataset[int(i)] for i in indices]
        batch = self.collate_fn(samples) if self.collate_fn is not None else _collate(samples)
        # Advance only after successfully constructing the batch.
        self._cursor = stop
        return batch

    def state_dict(self):
        import copy

        return {
            "size": len(self.dataset),
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "order": None if self._order is None else self._order.copy(),
            "cursor": self._cursor,
            "rng": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        import copy

        for name, current in (
            ("size", len(self.dataset)),
            ("batch_size", self.batch_size),
            ("shuffle", self.shuffle),
            ("drop_last", self.drop_last),
        ):
            if state[name] != current:
                raise ValueError(f"Loader {name} does not match checkpoint.")
        order, cursor = state["order"], state["cursor"]
        if type(cursor) is not int or not 0 <= cursor <= self._limit():
            raise ValueError("Invalid loader cursor.")
        if cursor != self._limit() and cursor % self.batch_size:
            raise ValueError("Loader cursor must be at a batch boundary.")
        if order is None:
            if cursor != 0:
                raise ValueError("An unstarted loader must have cursor zero.")
        else:
            order = np.asarray(order)
            if (
                order.dtype.kind not in "iu"
                or order.shape != (len(self.dataset),)
                or not np.array_equal(np.sort(order), np.arange(len(self.dataset)))
            ):
                raise ValueError("Invalid loader sample permutation.")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["rng"])
        self._order = None if order is None else order.copy()
        self._cursor, self._rng = cursor, rng


def _collate(samples):
    if isinstance(samples[0], Tensor):
        samples = [(sample,) for sample in samples]
    if not all(isinstance(s, tuple) and len(s) == len(samples[0]) for s in samples):
        raise TypeError("Dataset samples must have consistent tuple structure.")
    batch = []
    for column in zip(*samples):
        if not all(isinstance(t, Tensor) and not t.requires_grad for t in column):
            raise TypeError("Dataset items must be tensors without gradients.")
        array = np.stack([t._data for t in column])
        batch.append(Tensor(array, dtype=array.dtype))
    return tuple(batch)


def pad_sequence(sequences, padding_value=0):
    """Right-pad time-first tensors into a batch-first (B,T,...) tensor."""
    sequences = list(sequences)
    if not sequences or not all(isinstance(s, Tensor) and s.ndim > 0 for s in sequences):
        raise ValueError("pad_sequence needs a nonempty collection of sequence tensors.")
    shape, dtype = sequences[0].shape[1:], sequences[0].dtype
    if any(s.shape[1:] != shape or s.dtype != dtype or s.requires_grad for s in sequences):
        raise ValueError("Sequences must have matching features/dtypes and no gradients.")
    result = np.full(
        (len(sequences), max(len(s) for s in sequences)) + shape, padding_value, dtype=dtype
    )
    for index, sequence in enumerate(sequences):
        result[index, : len(sequence)] = sequence._data
    return Tensor(result, dtype=dtype)


def padding_mask(lengths, max_length=None):
    """Return a (batch,time) boolean tensor; True marks padding to block."""
    lengths = lengths.numpy() if isinstance(lengths, Tensor) else np.asarray(lengths)
    if lengths.ndim != 1 or lengths.dtype.kind not in "iu" or np.any(lengths < 0):
        raise ValueError("lengths must be a vector of nonnegative integers.")
    needed = int(lengths.max()) if lengths.size else 0
    max_length = needed if max_length is None else max_length
    if type(max_length) is not int or max_length < needed:
        raise ValueError("max_length must cover every sequence.")
    return Tensor(np.arange(max_length)[None, :] >= lengths[:, None], dtype=bool)
