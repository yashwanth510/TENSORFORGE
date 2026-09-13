"""A deterministic character tokenizer and next-token text dataset."""

import numpy as np

from .data import Dataset
from .tensor import Tensor


class CharacterTokenizer:
    """Four special IDs followed by sorted characters; no external tokenizer."""

    special_tokens = ("<pad>", "<bos>", "<eos>", "<unk>")
    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3

    def __init__(self, text):
        if not isinstance(text, str):
            raise TypeError("Tokenizer training text must be a string.")
        self.tokens = list(self.special_tokens) + sorted(set(text))
        self._ids = {token: index for index, token in enumerate(self.tokens)}

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, text, add_bos=False, add_eos=False):
        if not isinstance(text, str):
            raise TypeError("encode expects a string.")
        ids = [self._ids.get(character, self.unk_id) for character in text]
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids, skip_special=True):
        ids = ids.numpy() if isinstance(ids, Tensor) else np.asarray(ids)
        if ids.size == 0:
            return ""
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= self.vocab_size)
        ):
            raise ValueError("decode expects a vector of valid integer token IDs.")
        return "".join(self.tokens[int(i)] for i in ids if not skip_special or i >= 4)

    def state_dict(self):
        return {"type": "character", "tokens": list(self.tokens)}

    @classmethod
    def from_state_dict(cls, state):
        tokens = state.get("tokens", [])
        if (
            state.get("type") != "character"
            or not isinstance(tokens, list)
            or tokens[:4] != list(cls.special_tokens)
            or not all(isinstance(t, str) for t in tokens)
            or len(set(tokens)) != len(tokens)
            or any(len(t) != 1 for t in tokens[4:])
        ):
            raise ValueError("Invalid character tokenizer state.")
        result = cls("")
        result.tokens = list(tokens)
        result._ids = {token: index for index, token in enumerate(tokens)}
        return result


class TextDataset(Dataset):
    """Fixed-length next-token windows over a vector of token IDs."""

    def __init__(self, token_ids, sequence_length, stride=None):
        ids = token_ids.numpy() if isinstance(token_ids, Tensor) else np.asarray(token_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise ValueError("token_ids must be a vector of nonnegative integers.")
        if type(sequence_length) is not int or sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer.")
        stride = sequence_length if stride is None else stride
        if type(stride) is not int or stride <= 0 or len(ids) <= sequence_length:
            raise ValueError("Need a positive stride and at least sequence_length+1 tokens.")
        self.ids, self.sequence_length, self.stride = ids.copy(), sequence_length, stride

    def __len__(self):
        return (len(self.ids) - self.sequence_length - 1) // self.stride + 1

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("TextDataset index out of range.")
        start = index * self.stride
        x = self.ids[start : start + self.sequence_length]
        y = self.ids[start + 1 : start + self.sequence_length + 1]
        return Tensor(x, dtype=np.int64), Tensor(y, dtype=np.int64)
