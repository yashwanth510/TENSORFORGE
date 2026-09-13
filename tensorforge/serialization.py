"""Portable checkpoints containing JSON and numeric NumPy arrays, never pickle."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def save(state, path):
    """Atomically save nested dict/list/scalar/array state to an NPZ archive.

    Typical use: save(model.state_dict(), 'weights.npz'). For training state,
    save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path).
    The caller stores architecture, epoch, and sampler state when needed.
    """
    arrays = {}

    def encode(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf":
                raise TypeError("Only real numeric arrays can be saved.")
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"kind": "array", "key": key}
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("Checkpoint dictionary keys must be strings.")
            return {"kind": "dict", "items": [[k, encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"kind": "list", "items": [encode(v) for v in value]}
        if value is None or type(value) in (bool, int, float, str):
            return {"kind": "scalar", "value": value}
        raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")

    metadata = {"format": "tensorforge", "version": 1, "state": encode(state)}
    arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
    path = Path(path)
    # Write in the destination directory so os.replace stays on one filesystem.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
            temporary = file.name
            np.savez_compressed(file, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load(path):
    """Read TensorForge state without allowing executable pickle objects."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].item())
        if metadata.get("format") != "tensorforge" or metadata.get("version") != 1:
            raise ValueError("Unsupported TensorForge checkpoint format.")

        def decode(node):
            kind = node["kind"]
            if kind == "array":
                array = archive[node["key"]]
                if array.dtype.kind not in "biuf":
                    raise ValueError("Checkpoint contains a nonnumeric array.")
                return array.copy()
            if kind == "dict":
                return {key: decode(value) for key, value in node["items"]}
            if kind == "list":
                return [decode(value) for value in node["items"]]
            if kind == "scalar":
                return node["value"]
            raise ValueError("Invalid checkpoint node.")

        return decode(metadata["state"])
