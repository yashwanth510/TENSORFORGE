"""Gradient-based parameter updates; optimizer state uses plain arrays."""

import numpy as np

from .tensor import Tensor


def _nonnegative(value, name):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")


class Optimizer:
    def __init__(self, parameters, lr):
        _nonnegative(lr, "lr")
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("An optimizer needs at least one parameter.")
        if not all(
            isinstance(p, Tensor) and p.requires_grad and not p._parents for p in self.parameters
        ):
            raise TypeError("Optimizer parameters must be trainable leaf tensors.")
        if len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError("Duplicate optimizer parameters are not allowed.")
        self.lr = float(lr)

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

    def _gradients(self):
        # Validate all gradients before modifying any parameter.
        ready = []
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            gradient = np.asarray(parameter.grad, dtype=parameter.dtype)
            if gradient.shape != parameter.shape or not np.isfinite(gradient).all():
                raise ValueError("Gradients must be finite and match parameter shapes.")
            ready.append((index, parameter, gradient))
        return ready

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and L2 penalty."""

    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters, lr)
        if not 0 <= momentum < 1:
            raise ValueError("momentum must satisfy 0 <= momentum < 1.")
        _nonnegative(weight_decay, "weight_decay")
        self.momentum, self.weight_decay = momentum, weight_decay
        self._velocity = [np.zeros_like(p._data) for p in self.parameters]

    def step(self):
        for index, parameter, gradient in self._gradients():
            gradient = gradient + self.weight_decay * parameter._data
            # v_t = momentum*v_(t-1) + g_t; theta_t = theta_(t-1) - lr*v_t.
            self._velocity[index] = self.momentum * self._velocity[index] + gradient
            parameter._assign(parameter._data - self.lr * self._velocity[index])

    def state_dict(self):
        return {
            "type": "SGD",
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "velocity": [v.copy() for v in self._velocity],
        }

    def load_state_dict(self, state):
        if state.get("type") != "SGD":
            raise ValueError("Expected an SGD checkpoint.")
        candidate = SGD(self.parameters, state["lr"], state["momentum"], state["weight_decay"])
        candidate._velocity = _validated_arrays(state["velocity"], self.parameters)
        self.lr, self.momentum = candidate.lr, candidate.momentum
        self.weight_decay, self._velocity = candidate.weight_decay, candidate._velocity


class Adam(Optimizer):
    """Adaptive moment estimation, with bias correction and optional L2 penalty."""

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        if len(betas) != 2 or not all(0 <= b < 1 for b in betas):
            raise ValueError("betas must contain two values in [0, 1).")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        _nonnegative(weight_decay, "weight_decay")
        self.betas, self.eps, self.weight_decay = tuple(betas), eps, weight_decay
        self._m = [np.zeros_like(p._data) for p in self.parameters]
        self._v = [np.zeros_like(p._data) for p in self.parameters]
        # Each parameter counts its own updates; missing gradients skip updates.
        self._steps = [0 for _ in self.parameters]

    def step(self):
        b1, b2 = self.betas
        for index, parameter, gradient in self._gradients():
            gradient = gradient + self.weight_decay * parameter._data
            self._steps[index] += 1
            self._m[index] = b1 * self._m[index] + (1 - b1) * gradient
            self._v[index] = b2 * self._v[index] + (1 - b2) * gradient**2
            # Correct the initial bias caused by starting both averages at zero.
            m_hat = self._m[index] / (1 - b1 ** self._steps[index])
            v_hat = self._v[index] / (1 - b2 ** self._steps[index])
            parameter._assign(parameter._data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))

    def state_dict(self):
        return {
            "type": "Adam",
            "lr": self.lr,
            "betas": list(self.betas),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "m": [v.copy() for v in self._m],
            "v": [v.copy() for v in self._v],
            "steps": list(self._steps),
        }

    def load_state_dict(self, state):
        if state.get("type") != "Adam":
            raise ValueError("Expected an Adam checkpoint.")
        candidate = Adam(
            self.parameters, state["lr"], state["betas"], state["eps"], state["weight_decay"]
        )
        candidate._m = _validated_arrays(state["m"], self.parameters)
        candidate._v = _validated_arrays(state["v"], self.parameters)
        if any(np.any(v < 0) for v in candidate._v):
            raise ValueError("Adam second moments cannot be negative.")
        steps = state["steps"]
        if len(steps) != len(self.parameters) or any(type(s) is not int or s < 0 for s in steps):
            raise ValueError("Invalid Adam step counters.")
        self.lr, self.betas, self.eps = candidate.lr, candidate.betas, candidate.eps
        self.weight_decay = candidate.weight_decay
        self._m, self._v, self._steps = candidate._m, candidate._v, list(steps)


def _validated_arrays(arrays, parameters):
    if len(arrays) != len(parameters):
        raise ValueError("Optimizer parameter count does not match.")
    result = []
    for array, parameter in zip(arrays, parameters):
        array = np.asarray(array, dtype=parameter.dtype)
        if array.shape != parameter.shape or not np.isfinite(array).all():
            raise ValueError("Invalid optimizer state array.")
        result.append(array.copy())
    return result


class AdamW(Adam):
    """Adam with decoupled weight decay applied directly to parameters."""

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr, betas, eps, weight_decay)

    def step(self):
        b1, b2 = self.betas
        for index, parameter, gradient in self._gradients():
            self._steps[index] += 1
            self._m[index] = b1 * self._m[index] + (1 - b1) * gradient
            self._v[index] = b2 * self._v[index] + (1 - b2) * gradient**2
            m = self._m[index] / (1 - b1 ** self._steps[index])
            v = self._v[index] / (1 - b2 ** self._steps[index])
            parameter._assign(
                parameter._data * (1 - self.lr * self.weight_decay)
                - self.lr * m / (np.sqrt(v) + self.eps)
            )

    def state_dict(self):
        state = super().state_dict()
        state["type"] = "AdamW"
        return state

    def load_state_dict(self, state):
        if state.get("type") != "AdamW":
            raise ValueError("Expected an AdamW checkpoint.")
        state = dict(state)
        state["type"] = "Adam"
        super().load_state_dict(state)
