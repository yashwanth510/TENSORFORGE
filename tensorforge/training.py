"""Gradient clipping, learning-rate schedules, and resumable training state."""

import copy

import numpy as np

from .nn.modules import Module
from .serialization import load, save
from .tensor import get_rng_state, set_rng_state


def clip_grad_norm_(parameters, max_norm):
    """Scale all gradients to a combined L2 norm at most max_norm; return prior norm."""
    if not np.isfinite(max_norm) or max_norm < 0:
        raise ValueError("max_norm must be finite and nonnegative.")
    parameters = list(parameters)
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Duplicate parameters are not allowed.")
    gradients = [p.grad for p in parameters if p.grad is not None]
    if any(not np.isfinite(g).all() for g in gradients):
        raise ValueError("Cannot clip non-finite gradients.")
    # Normalize before squaring to reduce overflow for large finite gradients.
    largest = max((float(np.abs(g).max()) for g in gradients if g.size), default=0.0)
    scaled_norm = (
        0.0 if largest == 0 else np.sqrt(sum(float(np.sum((g / largest) ** 2)) for g in gradients))
    )
    norm = float(largest) * float(scaled_norm)
    if norm > max_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad = (p.grad / largest) * (max_norm / scaled_norm)
    return float(norm)


class LRScheduler:
    """Schedule the learning rate used by successive optimizer updates.

    Construction sets the rate for update 0. Call scheduler.step() AFTER each
    optimizer.step(). Warmup uses (update+1)/warmup_steps for early updates.
    """

    def __init__(
        self,
        optimizer,
        schedule="constant",
        warmup_steps=0,
        total_steps=100,
        step_size=10,
        gamma=0.1,
        min_lr=0.0,
    ):
        if schedule not in ("constant", "step", "cosine"):
            raise ValueError("schedule must be constant, step, or cosine.")
        if (
            type(warmup_steps) is not int
            or warmup_steps < 0
            or type(total_steps) is not int
            or total_steps <= warmup_steps
            or type(step_size) is not int
            or step_size <= 0
        ):
            raise ValueError("Invalid schedule step counts.")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1 or not 0 <= min_lr <= optimizer.lr:
            raise ValueError("gamma must lie in [0,1]; min_lr must lie between zero and base lr.")
        self.optimizer, self.base_lr = optimizer, optimizer.lr
        self.schedule, self.warmup_steps, self.total_steps = schedule, warmup_steps, total_steps
        self.step_size, self.gamma, self.min_lr = step_size, gamma, min_lr
        self.step_count = 0
        self.optimizer.lr = self._rate()

    def _rate(self):
        if self.step_count < self.warmup_steps:
            return self.base_lr * (self.step_count + 1) / self.warmup_steps
        elapsed = self.step_count - self.warmup_steps
        if self.schedule == "step":
            return max(self.min_lr, self.base_lr * self.gamma ** (elapsed // self.step_size))
        if self.schedule == "cosine":
            progress = min(elapsed / (self.total_steps - self.warmup_steps), 1.0)
            return float(
                self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
            )
        return self.base_lr

    def step(self):
        self.step_count += 1
        self.optimizer.lr = self._rate()
        return self.optimizer.lr

    def state_dict(self):
        names = (
            "base_lr",
            "schedule",
            "warmup_steps",
            "total_steps",
            "step_size",
            "gamma",
            "min_lr",
            "step_count",
        )
        return {name: getattr(self, name) for name in names}

    def load_state_dict(self, state):
        expected = set(self.state_dict())
        if (
            set(state) != expected
            or type(state["step_count"]) is not int
            or state["step_count"] < 0
        ):
            raise ValueError("Invalid scheduler state.")

        # Validate on a tiny proxy so validation cannot alter the real optimizer.
        class Proxy:
            lr = state["base_lr"]

        if not np.isfinite(Proxy.lr) or Proxy.lr < 0:
            raise ValueError("Invalid base learning rate.")
        candidate = LRScheduler(
            Proxy(),
            state["schedule"],
            state["warmup_steps"],
            state["total_steps"],
            state["step_size"],
            state["gamma"],
            state["min_lr"],
        )
        candidate.step_count = state["step_count"]
        for name in expected:
            setattr(self, name, getattr(candidate, name))
        self.optimizer.lr = self._rate()


def save_checkpoint(path, model, optimizer=None, scheduler=None, loader=None, step=0, extra=None):
    """Save at an optimizer-step boundary, after all accumulated gradients are used.

    Captures TensorForge RNG, module modes, optional loader order/cursor, and
    optimizer/scheduler states. Architecture and arbitrary user RNGs are not
    serialized; store model config and tokenizer in extra when appropriate.
    """
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer.")
    if scheduler is not None and (optimizer is None or scheduler.optimizer is not optimizer):
        raise ValueError("scheduler must belong to the supplied optimizer.")
    save(
        {
            "model": model.state_dict(),
            "modes": {name: m.training for name, m in model._walk() if isinstance(m, Module)},
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "loader": None if loader is None else loader.state_dict(),
            "rng": get_rng_state(),
            "step": step,
            "extra": extra,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, loader=None):
    """Restore state and return {'step': ..., 'extra': ...}."""
    state = load(path)
    if scheduler is not None and (optimizer is None or scheduler.optimizer is not optimizer):
        raise ValueError("scheduler must belong to the supplied optimizer.")
    modules = {name: m for name, m in model._walk() if isinstance(m, Module)}
    if set(state["modes"]) != set(modules) or any(
        type(v) is not bool for v in state["modes"].values()
    ):
        raise ValueError("Checkpoint module structure does not match.")
    # Validate against isolated copies before committing any changes.
    model_copy = copy.deepcopy(model)
    model_copy.load_state_dict(state["model"])
    for name, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("loader", loader)):
        if obj is not None:
            if state[name] is None:
                raise ValueError(f"Checkpoint has no {name} state.")
            copy.deepcopy(obj).load_state_dict(state[name])
    candidate_rng = np.random.default_rng()
    candidate_rng.bit_generator.state = copy.deepcopy(state["rng"])
    if type(state["step"]) is not int or state["step"] < 0:
        raise ValueError("Invalid checkpoint step.")
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if loader is not None:
        loader.load_state_dict(state["loader"])
    for name, module in modules.items():
        module.training = state["modes"][name]
    set_rng_state(state["rng"])
    return {"step": state["step"], "extra": state["extra"]}
