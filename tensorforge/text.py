"""Character and byte-pair tokenizers with next-token text datasets."""

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


class BytePairTokenizer:
    """UTF-8 byte-pair encoding with lossless byte fallback and learned merges.

    IDs 0..3 are pad/bos/eos/unknown; IDs 4..259 represent raw bytes. This is
    a local tokenizer format, not an import of GPT-2 or another model's BPE.
    """

    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3

    def __init__(self, text="", vocab_size=300, min_frequency=2):
        from collections import Counter

        if not isinstance(text, str) or type(vocab_size) is not int or vocab_size < 260:
            raise ValueError("Provide text and a vocabulary size of at least 260.")
        if type(min_frequency) is not int or min_frequency < 1:
            raise ValueError("min_frequency must be positive.")
        self.merges = []
        self._pieces = [bytes([i]) for i in range(256)]
        ids = [byte + 4 for byte in text.encode("utf-8")]
        while self.vocab_size < vocab_size:
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            if counts[pair] < min_frequency:
                break
            new_id = self.vocab_size
            self.merges.append(pair)
            self._pieces.append(self._pieces[pair[0] - 4] + self._pieces[pair[1] - 4])
            ids = self._merge(ids, pair, new_id)

    @property
    def vocab_size(self):
        return 4 + len(self._pieces)

    @staticmethod
    def _merge(ids, pair, new_id):
        result, index = [], 0
        while index < len(ids):
            if index + 1 < len(ids) and (ids[index], ids[index + 1]) == pair:
                result.append(new_id)
                index += 2
            else:
                result.append(ids[index])
                index += 1
        return result

    def encode(self, text, add_bos=False, add_eos=False):
        if not isinstance(text, str):
            raise TypeError("encode expects text.")
        ids = [byte + 4 for byte in text.encode("utf-8")]
        for index, pair in enumerate(self.merges):
            ids = self._merge(ids, pair, 260 + index)
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
            raise ValueError("decode expects valid integer token IDs.")
        special = (b"<pad>", b"<bos>", b"<eos>", b"<unk>")
        pieces = [
            self._pieces[int(i) - 4] if i >= 4 else (b"" if skip_special else special[int(i)])
            for i in ids
        ]
        return b"".join(pieces).decode("utf-8", errors="replace")

    def state_dict(self):
        return {"type": "byte_pair", "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_state_dict(cls, state):
        if state.get("type") != "byte_pair" or not isinstance(state.get("merges"), list):
            raise ValueError("Invalid byte-pair tokenizer state.")
        result = cls()
        seen = set()
        for pair in state["merges"]:
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or any(type(i) is not int or not 4 <= i < result.vocab_size for i in pair)
                or tuple(pair) in seen
            ):
                raise ValueError("Invalid merge table.")
            pair = tuple(pair)
            seen.add(pair)
            result.merges.append(pair)
            result._pieces.append(result._pieces[pair[0] - 4] + result._pieces[pair[1] - 4])
        return result
