"""Parameter-efficient layers and preference-training objectives."""

import numpy as np

from .nn.functional import binary_cross_entropy_with_logits
from .nn.modules import Linear, Module
from .tensor import Tensor


def freeze(module):
    """Freeze parameters before building a new computation graph."""
    for parameter in module.parameters():
        parameter.requires_grad = False
        parameter.zero_grad()
        parameter._version += 1
    return module


def trainable_parameters(module):
    return [p for p in module.parameters() if p.requires_grad]


class LoRALinear(Module):
    """Frozen linear map plus a trainable low-rank update, scaled by alpha/rank."""

    def __init__(self, base, rank=4, alpha=4.0):
        super().__init__()
        if not isinstance(base, Linear) or type(rank) is not int or rank < 1:
            raise ValueError("LoRALinear needs a Linear layer and positive rank.")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive.")
        self.base = freeze(base)
        self.scale = alpha / rank
        self.a = Linear(base.weight.shape[1], rank, bias=False)
        self.b = Linear(rank, base.weight.shape[0], bias=False)
        self.b.weight._assign(np.zeros(self.b.weight.shape))

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale

    def merged(self):
        """Return an independent frozen Linear layer for inference."""
        out = Linear(
            self.base.weight.shape[1], self.base.weight.shape[0], bias=self.base.bias is not None
        )
        out.weight._assign(
            self.base.weight._data + self.scale * (self.b.weight._data @ self.a.weight._data)
        )
        if out.bias is not None:
            out.bias._assign(self.base.bias._data)
        return freeze(out)


class Adapter(Module):
    """A residual bottleneck adapter initialized as the identity."""

    def __init__(self, features, bottleneck=4):
        super().__init__()
        self.down = Linear(features, bottleneck)
        self.up = Linear(bottleneck, features)
        self.up.weight._assign(np.zeros(self.up.weight.shape))
        self.up.bias._assign(np.zeros(self.up.bias.shape))

    def forward(self, x):
        return x + self.up(self.down(x).gelu())


def sequence_log_probs(logits, targets, ignore_index=-100, average=False):
    """Sum (or average) selected token log probabilities for each batch item."""
    ids = targets.numpy() if isinstance(targets, Tensor) else np.asarray(targets)
    if logits.ndim != 3 or ids.shape != logits.shape[:2] or ids.dtype.kind not in "iu":
        raise ValueError("Expected (batch,time,classes) logits and integer (batch,time) targets.")
    valid = ids != ignore_index
    if np.any(ids[valid] < 0) or np.any(ids[valid] >= logits.shape[-1]):
        raise ValueError("Target outside the vocabulary.")
    safe = np.where(valid, ids, 0)
    b, t = ids.shape
    selected = logits.log_softmax()[np.arange(b)[:, None], np.arange(t)[None, :], safe]
    result = (selected * valid).sum(axis=1)
    return result / Tensor(np.maximum(valid.sum(axis=1), 1)) if average else result


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.1):
    """DPO loss on sequence log probabilities; reference scores are detached."""
    if not np.isfinite(beta) or beta <= 0 or chosen.shape != rejected.shape or chosen.ndim != 1:
        raise ValueError("Provide matching score vectors and a positive beta.")
    refs = [
        r.detach() if isinstance(r, Tensor) else Tensor(r)
        for r in (reference_chosen, reference_rejected)
    ]
    if any(r.shape != chosen.shape for r in refs):
        raise ValueError("Reference score shapes must match the policy scores.")
    margin = beta * ((chosen - rejected) - (refs[0] - refs[1]))
    return binary_cross_entropy_with_logits(margin, Tensor(np.ones(margin.shape)))


def reward_ranking_loss(chosen_rewards, rejected_rewards):
    if chosen_rewards.shape != rejected_rewards.shape:
        raise ValueError("Reward shapes must match.")
    margin = chosen_rewards - rejected_rewards
    return binary_cross_entropy_with_logits(margin, Tensor(np.ones(margin.shape)))


def policy_gradient_loss(log_probs, advantages):
    """REINFORCE objective with fixed advantages; not a complete PPO trainer."""
    advantages = advantages.detach() if isinstance(advantages, Tensor) else Tensor(advantages)
    if log_probs.shape != advantages.shape:
        raise ValueError("Log probabilities and advantages must have matching shapes.")
    return -(log_probs * advantages).mean()


class QuantizedLinear(Module):
    """Per-output-channel int8 weight storage, dequantized for CPU computation.

    This reduces stored weight size; it is not an accelerated integer kernel.
    """

    def __init__(self, source):
        super().__init__()
        if not isinstance(source, Linear):
            raise TypeError("QuantizedLinear needs a Linear layer.")
        weight = source.weight.numpy()
        scale = np.maximum(
            np.abs(weight).max(axis=1, keepdims=True) / 127, np.finfo(np.float64).tiny
        )
        self.register_buffer(
            "weight_int8", Tensor(np.clip(np.rint(weight / scale), -127, 127), dtype=np.int8)
        )
        self.register_buffer("scale", Tensor(scale))
        self.register_buffer(
            "bias",
            Tensor(np.zeros(weight.shape[0]) if source.bias is None else source.bias.numpy()),
        )

    def forward(self, x):
        weight = Tensor(self.weight_int8._data.astype(np.float64) * self.scale._data)
        return x @ weight.T + self.bias


def apply_lora(model, rank=4, alpha=4.0, target_names=("q_proj", "v_proj")):
    """Freeze a model and replace selected Linear attributes with LoRA layers."""
    selected = []
    for path, owner in model._walk():
        if isinstance(owner, Module):
            for name, value in vars(owner).items():
                if name in target_names and isinstance(value, Linear):
                    selected.append((path, owner, name, value))
    if not selected:
        raise ValueError("No Linear attributes match target_names.")
    if type(rank) is not int or rank < 1 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid LoRA rank or alpha.")
    freeze(model)
    shared = {}
    for _, owner, name, base in selected:
        if id(base) not in shared:
            shared[id(base)] = LoRALinear(base, rank, alpha)
        setattr(owner, name, shared[id(base)])
    return [f"{path}.{name}" if path else name for path, _, name, _ in selected]


def merge_lora(model):
    """Replace LoRA wrappers with independent merged frozen Linear layers."""
    replacements = []
    for _, owner in model._walk():
        if isinstance(owner, Module):
            for name, value in vars(owner).items():
                if isinstance(value, LoRALinear):
                    replacements.append((owner, name, value))
    shared = {}
    for owner, name, layer in replacements:
        if id(layer) not in shared:
            shared[id(layer)] = layer.merged()
        setattr(owner, name, shared[id(layer)])
    return model
