"""Reference beam search and exact greedy speculative decoding."""

import numpy as np

from .evaluation import evaluating
from .tensor import Tensor


def _prompt(model, tokens, max_new_tokens):
    raw = tokens.numpy() if isinstance(tokens, Tensor) else np.asarray(tokens)
    if raw.ndim != 2 or raw.shape[0] != 1 or raw.shape[1] == 0 or raw.dtype.kind not in "iu":
        raise ValueError("This search API accepts one nonempty integer prompt at a time.")
    if (
        type(max_new_tokens) is not int
        or max_new_tokens < 0
        or raw.shape[1] + max_new_tokens > model.config.max_length
    ):
        raise ValueError("Requested length is invalid or exceeds the model context.")
    if np.any(raw < 0) or np.any(raw >= model.config.vocab_size):
        raise ValueError("Prompt IDs are out of range.")
    return raw[0].tolist()


def beam_search(
    model,
    tokens,
    max_new_tokens=20,
    num_beams=4,
    length_penalty=1.0,
    repetition_penalty=1.0,
    eos_token_id=None,
    allowed_tokens_fn=None,
):
    """Reference full-prefix beam search with optional token constraints.

    Scores use log probabilities divided by generated_length**length_penalty.
    Constraints receive the full prefix and return allowed IDs. One prompt only.
    """
    prompt = _prompt(model, tokens, max_new_tokens)
    if type(num_beams) is not int or not 1 <= num_beams <= model.config.vocab_size:
        raise ValueError("num_beams must lie between one and vocabulary size.")
    if (
        not np.isfinite(length_penalty)
        or length_penalty < 0
        or not np.isfinite(repetition_penalty)
        or repetition_penalty < 1
    ):
        raise ValueError("Invalid length or repetition penalty.")
    if eos_token_id is not None and (
        type(eos_token_id) is not int or not 0 <= eos_token_id < model.config.vocab_size
    ):
        raise ValueError("Invalid end token.")
    if allowed_tokens_fn is not None and not callable(allowed_tokens_fn):
        raise TypeError("allowed_tokens_fn must be callable.")
    beams = [(prompt, 0.0, False)]

    def score(beam):
        return beam[1] / max(len(beam[0]) - len(prompt), 1) ** length_penalty

    with evaluating(model):
        for _ in range(max_new_tokens):
            candidates = []
            for prefix, total, done in beams:
                if done:
                    candidates.append((prefix, total, done))
                    continue
                logits = model(Tensor([prefix], dtype=np.int64)).numpy()[0, -1]
                if not np.isfinite(logits).all():
                    raise ValueError("Non-finite generation logits.")
                used = np.array(sorted(set(prefix)))
                logits[used] = np.where(
                    logits[used] >= 0,
                    logits[used] / repetition_penalty,
                    logits[used] * repetition_penalty,
                )
                if allowed_tokens_fn is not None:
                    allowed = list(allowed_tokens_fn(tuple(prefix)))
                    if not allowed or any(
                        type(i) is not int or not 0 <= i < len(logits) for i in allowed
                    ):
                        raise ValueError("Constraints must return nonempty valid integer IDs.")
                    blocked = np.ones(len(logits), dtype=bool)
                    blocked[allowed] = False
                    logits[blocked] = -np.inf
                log_probs = logits - logits.max()
                log_probs -= np.log(np.exp(log_probs).sum())
                order = np.argsort(-log_probs, kind="stable")[:num_beams]
                for token in order:
                    if np.isfinite(log_probs[token]):
                        candidates.append(
                            (
                                prefix + [int(token)],
                                total + float(log_probs[token]),
                                token == eos_token_id,
                            )
                        )
            beams = sorted(candidates, key=score, reverse=True)[:num_beams]
            if all(beam[2] for beam in beams):
                break
    return Tensor([max(beams, key=score)[0]], dtype=np.int64)


def speculative_greedy(target, draft, tokens, max_new_tokens=20, draft_steps=4, eos_token_id=None):
    """Verify a draft block with one target forward call; match greedy target output.

    This is exact greedy verification, not stochastic speculative sampling.
    Draft and target must use the same vocabulary and token meanings.
    """
    prefix = _prompt(target, tokens, max_new_tokens)
    _prompt(draft, tokens, max_new_tokens)
    if (
        target.config.vocab_size != draft.config.vocab_size
        or type(draft_steps) is not int
        or draft_steps < 1
    ):
        raise ValueError("Models need matching vocabularies and positive draft_steps.")
    if eos_token_id is not None and (
        type(eos_token_id) is not int or not 0 <= eos_token_id < target.config.vocab_size
    ):
        raise ValueError("Invalid end token.")
    end = len(prefix) + max_new_tokens
    with evaluating(target), evaluating(draft):
        while len(prefix) < end:
            start = len(prefix)
            proposal = list(prefix)
            for _ in range(min(draft_steps, end - start)):
                token = int(draft(Tensor([proposal], dtype=np.int64)).numpy()[0, -1].argmax())
                proposal.append(token)
            verified = target(Tensor([proposal], dtype=np.int64)).numpy()[0]
            for index in range(start, len(proposal)):
                actual = int(verified[index - 1].argmax())
                prefix.append(actual)
                if actual == eos_token_id or len(prefix) == end:
                    return Tensor([prefix], dtype=np.int64)
                if actual != proposal[index]:
                    break
    return Tensor([prefix], dtype=np.int64)
