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


def mask_tokens(tokens, mask_id, vocab_size, probability=0.15, special_ids=(), seed=None):
    """BERT-style 80% mask / 10% random / 10% unchanged corruption.

    Targets are -100 at unselected positions. Sampling may select no tokens;
    callers should not force hidden labels into model inputs.
    """
    ids = tokens.numpy() if isinstance(tokens, Tensor) else np.asarray(tokens)
    if ids.ndim != 2 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= vocab_size):
        raise ValueError("tokens must be a batch of valid vocabulary IDs.")
    if (
        type(vocab_size) is not int
        or type(mask_id) is not int
        or not 0 <= mask_id < vocab_size
        or not 0 <= probability <= 1
    ):
        raise ValueError("Invalid vocabulary, mask ID, or probability.")
    rng = np.random.default_rng(seed)
    eligible = ~np.isin(ids, list(special_ids))
    selected = eligible & (rng.random(ids.shape) < probability)
    targets = np.where(selected, ids, -100)
    corrupted = ids.copy()
    draw = rng.random(ids.shape)
    corrupted[selected & (draw < 0.8)] = mask_id
    random_positions = selected & (draw >= 0.8) & (draw < 0.9)
    choices = np.array([i for i in range(vocab_size) if i not in set(special_ids)], dtype=np.int64)
    if random_positions.any():
        if not choices.size:
            raise ValueError("Random replacement needs at least one non-special token.")
        corrupted[random_positions] = rng.choice(choices, size=int(random_positions.sum()))
    return Tensor(corrupted, dtype=np.int64), Tensor(targets, dtype=np.int64)


def instruction_sample(tokenizer, prompt, response, max_length=None):
    """Next-token inputs with prompt targets ignored for supervised fine-tuning."""
    prefix = tokenizer.encode(prompt, add_bos=True)
    answer = tokenizer.encode(response, add_eos=True)
    ids = prefix + answer
    if max_length is not None:
        if type(max_length) is not int or max_length < 2:
            raise ValueError("max_length must be at least two.")
        ids = ids[: max_length + 1]
    x, y = np.array(ids[:-1]), np.array(ids[1:])
    y[: min(len(prefix) - 1, len(y))] = -100
    return Tensor(x, dtype=np.int64), Tensor(y, dtype=np.int64)


class PairedTextDataset(Dataset):
    """Local source/target text pairs for translation or other sequence tasks."""

    def __init__(self, pairs, tokenizer, target_tokenizer=None):
        self.pairs = list(pairs)
        self.source_tokenizer = tokenizer
        self.target_tokenizer = tokenizer if target_tokenizer is None else target_tokenizer
        if not self.pairs or any(
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or not all(isinstance(s, str) for s in pair)
            for pair in self.pairs
        ):
            raise ValueError("Provide nonempty (source_text,target_text) pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target = self.pairs[index]
        source = self.source_tokenizer.encode(source, add_eos=True)
        target = self.target_tokenizer.encode(target, add_bos=True, add_eos=True)
        return tuple(Tensor(ids, dtype=np.int64) for ids in (source, target[:-1], target[1:]))


def corrupt_spans(tokens, sentinel_ids, noise_density=0.15, mean_span_length=3.0, seed=None):
    """Replace random spans with sentinels; return source and reconstruction target.

    Sentinels must be reserved IDs absent from the input. This operates on one
    sequence and does not insert beginning/end tokens automatically.
    """
    ids = tokens.numpy() if isinstance(tokens, Tensor) else np.asarray(tokens)
    sentinels = list(sentinel_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or ids.size == 0:
        raise ValueError("tokens must be a nonempty integer vector.")
    if not 0 <= noise_density <= 1 or not np.isfinite(mean_span_length) or mean_span_length < 1:
        raise ValueError("Invalid noise density or mean span length.")
    if (
        not sentinels
        or len(set(sentinels)) != len(sentinels)
        or any(type(i) is not int or i < 0 or i in ids for i in sentinels)
    ):
        raise ValueError("Sentinels must be distinct reserved nonnegative IDs.")
    rng = np.random.default_rng(seed)
    count = int(round(len(ids) * noise_density))
    selected = np.zeros(len(ids), dtype=bool)
    remaining = count
    while remaining:
        start = int(rng.choice(np.flatnonzero(~selected)))
        length = min(int(rng.geometric(1 / mean_span_length)), remaining)
        for index in range(start, min(start + length, len(ids))):
            if selected[index]:
                break
            selected[index] = True
            remaining -= 1
    source, target = [], []
    spans = 0
    for index, token in enumerate(ids):
        if selected[index]:
            if index == 0 or not selected[index - 1]:
                if spans == len(sentinels):
                    raise ValueError("Not enough sentinel IDs for the sampled spans.")
                source.append(sentinels[spans])
                target.append(sentinels[spans])
                spans += 1
            target.append(int(token))
        else:
            source.append(int(token))
    return Tensor(source, dtype=np.int64), Tensor(target, dtype=np.int64)
