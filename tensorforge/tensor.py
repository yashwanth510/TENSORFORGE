"""CPU tensors and reverse-mode automatic differentiation.

NumPy performs array arithmetic; TensorForge constructs the graph
and implements every derivative. Internal arrays are never mutated in place.
"""

import copy
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

# ContextVar isolates this setting across asynchronous contexts.
_grad_enabled = ContextVar("tensorforge_grad_enabled", default=True)
_rng = np.random.default_rng()


def manual_seed(seed):
    """Reset the generator used by factories, initialization, and dropout."""
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng_state():
    """Copy the random generator state for reproducible continuation."""
    return copy.deepcopy(_rng.bit_generator.state)


def set_rng_state(state):
    """Restore state produced by get_rng_state()."""
    candidate = np.random.default_rng()
    candidate.bit_generator.state = copy.deepcopy(state)
    _rng.bit_generator.state = candidate.bit_generator.state


@contextmanager
def no_grad():
    """Temporarily stop recording operations; nesting restores prior state."""
    token = _grad_enabled.set(False)
    try:
        yield
    finally:
        _grad_enabled.reset(token)


def _unbroadcast(gradient, shape):
    """Reverse broadcasting by summing the dimensions expanded forward."""
    # Extra leading dimensions did not exist in the original operand.
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    # A size-one dimension contributed its value to multiple output entries.
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


def _as_tensor(value):
    return value if isinstance(value, Tensor) else Tensor(value)


class Tensor:
    """An array and the graph needed to differentiate it.

    Numeric inputs default to float64 for beginner-friendly gradient checks.
    Explicit integer and boolean dtypes are supported without gradients.
    ``grad`` is a NumPy array or None, rather than another Tensor.
    """

    __array_priority__ = 1000

    def __init__(self, data, requires_grad=False, dtype=None):
        if isinstance(data, Tensor):
            data = data._data
        array = np.array(data, dtype=np.float64 if dtype is None else dtype, copy=True)
        if array.dtype.kind not in "biuf":
            raise TypeError("Tensor data must be real numeric or boolean values.")
        if requires_grad and array.dtype.kind != "f":
            raise TypeError("Only floating-point tensors can require gradients.")
        self._data = array
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._parents = ()
        self._versions = ()
        self._vjp = None
        self._version = 0

    @property
    def data(self):
        """A read-only copy; use optimizers to update parameters."""
        result = self._data.copy()
        result.flags.writeable = False
        return result

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def size(self):
        return self._data.size

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"Tensor({self._data!r}, requires_grad={self.requires_grad})"

    def numpy(self):
        """Return an independent, writable NumPy copy."""
        return self._data.copy()

    def item(self):
        """Extract the Python number from a one-element tensor."""
        return self._data.item()

    def detach(self):
        """Copy values without their computation history."""
        return Tensor(self._data, dtype=self.dtype)

    def zero_grad(self):
        self.grad = None

    def clone(self):
        return self._op(self._data, (self,), lambda g: (g,))

    def squeeze(self, axis=None):
        data = np.squeeze(self._data, axis=axis)
        return self._op(data, (self,), lambda g: (g.reshape(self.shape),))

    def unsqueeze(self, axis):
        data = np.expand_dims(self._data, axis)
        return self._op(data, (self,), lambda g: (g.reshape(self.shape),))

    def flatten(self, start_dim=0):
        if not isinstance(start_dim, int) or not 0 <= start_dim < self.ndim:
            raise ValueError("start_dim must identify an existing nonnegative axis.")
        return self.reshape(*self.shape[:start_dim], int(np.prod(self.shape[start_dim:])))

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        data = np.broadcast_to(self._data, shape)
        return self._op(data, (self,), lambda g: (_unbroadcast(g, self.shape),))

    def abs(self):
        return self._op(np.abs(self._data), (self,), lambda g: (g * np.sign(self._data),))

    def __abs__(self):
        return self.abs()

    def sqrt(self):
        return self**0.5

    def sin(self):
        return self._op(np.sin(self._data), (self,), lambda g: (g * np.cos(self._data),))

    def cos(self):
        return self._op(np.cos(self._data), (self,), lambda g: (-g * np.sin(self._data),))

    def amax(self, axis=None, keepdims=False):
        maximum = self._data.max(axis=axis, keepdims=True)
        mask = self._data == maximum
        count = mask.sum(axis=axis, keepdims=True)
        data = self._data.max(axis=axis, keepdims=keepdims)

        def vjp(g):
            # Equal maxima share the derivative equally.
            return (g.reshape(maximum.shape) * mask / count,)

        return self._op(data, (self,), vjp)

    def amin(self, axis=None, keepdims=False):
        return -((-self).amax(axis=axis, keepdims=keepdims))

    def argmax(self, axis=None):
        return Tensor(self._data.argmax(axis=axis), dtype=np.int64)

    def masked_fill(self, mask, value):
        """Replace positions where a boolean mask is True with a constant."""
        return where(mask, value, self)

    def gelu(self):
        """GELU using the standard tanh approximation."""
        return 0.5 * self * (1 + (np.sqrt(2 / np.pi) * (self + 0.044715 * self**3)).tanh())

    def silu(self):
        return self * self.sigmoid()

    def _assign(self, data):
        """Internal update hook; version counters detect stale graphs."""
        data = np.asarray(data, dtype=self.dtype)
        if data.shape != self.shape:
            raise ValueError("Updates must preserve the tensor shape.")
        self._data = data.copy()
        self._version += 1

    @staticmethod
    def _op(data, parents, vjp):
        # A vector-Jacobian product (VJP) maps an output gradient to inputs.
        requires_grad = _grad_enabled.get() and any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=requires_grad, dtype=np.asarray(data).dtype)
        if requires_grad:
            out._parents = tuple(parents)
            out._versions = tuple(p._version for p in parents)
            out._vjp = vjp
        return out

    def backward(self, gradient=None):
        """Differentiate this output, accumulating gradients across calls.

        Multi-element outputs need an explicit gradient of the same shape.
        Call zero_grad() between independent optimization steps.
        """
        if not self.requires_grad:
            raise RuntimeError("This tensor does not require gradients.")
        if gradient is None:
            if self.size != 1:
                raise ValueError("A non-scalar output needs an explicit gradient.")
            gradient = np.ones_like(self._data)
        else:
            if isinstance(gradient, Tensor):
                gradient = gradient._data
            gradient = np.asarray(gradient, dtype=self.dtype)
            if gradient.shape != self.shape:
                raise ValueError("Gradient shape must match the output shape exactly.")

        # Iterative depth-first traversal avoids Python's recursion limit.
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                stack.extend((p, False) for p in node._parents)

        # Validate the whole graph before changing any stored gradients.
        for node in order:
            for parent, version in zip(node._parents, node._versions):
                if parent._version != version:
                    raise RuntimeError("A tensor changed after forward; recompute the output.")

        # Fresh per-call gradients prevent stale intermediate gradients from
        # being propagated on a second call to backward().
        pending = {self: gradient}
        for node in reversed(order):
            if node not in pending:
                continue
            incoming = pending[node]
            if node.requires_grad:
                node.grad = incoming.copy() if node.grad is None else node.grad + incoming
            if node._vjp is not None:
                for parent, contribution in zip(node._parents, node._vjp(incoming)):
                    if parent.requires_grad:
                        contribution = np.asarray(contribution, dtype=parent.dtype)
                        pending[parent] = pending.get(parent, 0) + contribution

    def __add__(self, other):
        other = _as_tensor(other)
        return self._op(
            self._data + other._data,
            (self, other),
            lambda g: (_unbroadcast(g, self.shape), _unbroadcast(g, other.shape)),
        )

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-_as_tensor(other))

    def __rsub__(self, other):
        return _as_tensor(other) - self

    def __mul__(self, other):
        other = _as_tensor(other)
        # For z=a*b, dz/da=b and dz/db=a; undo any broadcasting.
        return self._op(
            self._data * other._data,
            (self, other),
            lambda g: (
                _unbroadcast(g * other._data, self.shape),
                _unbroadcast(g * self._data, other.shape),
            ),
        )

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)) or not np.isfinite(exponent):
            raise TypeError("The exponent must be a finite Python int or float.")
        if np.any(self._data < 0) and not float(exponent).is_integer():
            raise ValueError("Fractional powers of negative values are unsupported.")
        if np.any(self._data == 0) and exponent < 0:
            raise ZeroDivisionError("Zero cannot have a negative power.")
        if np.any(self._data == 0) and 0 < exponent < 1:
            raise ValueError("This power has no finite derivative at zero.")

        def vjp(g):
            if exponent == 0:
                return (np.zeros_like(self._data),)
            return (g * exponent * self._data ** (exponent - 1),)

        return self._op(self._data**exponent, (self,), vjp)

    def __truediv__(self, other):
        return self * (_as_tensor(other) ** -1)

    def __rtruediv__(self, other):
        return _as_tensor(other) / self

    def __matmul__(self, other):
        other = _as_tensor(other)
        # NumPy matmul covers vector, matrix, and broadcasted batch cases.
        data = np.matmul(self._data, other._data)

        def vjp(g):
            a, b = self._data, other._data
            a_vector, b_vector = a.ndim == 1, b.ndim == 1
            aa = a[np.newaxis, :] if a_vector else a
            bb = b[:, np.newaxis] if b_vector else b
            gg = g
            if b_vector:
                gg = np.expand_dims(gg, -1)
            if a_vector:
                gg = np.expand_dims(gg, -2)
            ga = np.matmul(gg, np.swapaxes(bb, -1, -2))
            gb = np.matmul(np.swapaxes(aa, -1, -2), gg)
            if a_vector:
                ga = np.squeeze(ga, -2)
            if b_vector:
                gb = np.squeeze(gb, -1)
            return _unbroadcast(ga, self.shape), _unbroadcast(gb, other.shape)

        return self._op(data, (self, other), vjp)

    def __rmatmul__(self, other):
        return _as_tensor(other) @ self

    def matmul(self, other):
        return self @ other

    def sum(self, axis=None, keepdims=False):
        # Normalize axes so the backward pass knows which dimensions vanished.
        if axis is None:
            axes = tuple(range(self.ndim))
        else:
            raw_axes = (axis,) if isinstance(axis, int) else tuple(axis)
            if any(not isinstance(a, int) or a < -self.ndim or a >= self.ndim for a in raw_axes):
                raise ValueError("Reduction axis is out of range.")
            axes = tuple(a % self.ndim for a in raw_axes)
            if len(set(axes)) != len(axes):
                raise ValueError("Reduction axes must be unique.")

        def vjp(g):
            if not keepdims:
                for a in sorted(axes):
                    g = np.expand_dims(g, a)
            return (np.broadcast_to(g, self.shape),)

        return self._op(self._data.sum(axis=axes, keepdims=keepdims), (self,), vjp)

    def mean(self, axis=None, keepdims=False):
        summed = self.sum(axis=axis, keepdims=keepdims)
        # Use shapes to infer the number of entries combined per output.
        count = self.size // max(summed.size, 1)
        if count == 0:
            raise ValueError("The mean of an empty reduction is undefined.")
        return summed / count

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return self._op(self._data.reshape(shape), (self,), lambda g: (g.reshape(self.shape),))

    def transpose(self, dim0=-2, dim1=-1):
        # Like torch.transpose, swap two axes; use permute for all axes.
        return self._op(
            np.swapaxes(self._data, dim0, dim1), (self,), lambda g: (np.swapaxes(g, dim0, dim1),)
        )

    def permute(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        data = np.transpose(self._data, axes)
        normalized = tuple(a % self.ndim for a in axes)
        inverse = np.argsort(normalized)
        return self._op(data, (self,), lambda g: (np.transpose(g, inverse),))

    @property
    def T(self):
        """Reverse all axes, matching NumPy's .T convention."""
        return self.permute(*reversed(range(self.ndim)))

    def __getitem__(self, index):
        # Copy mutable indices so later user edits cannot change backward.
        def freeze(part):
            if isinstance(part, Tensor):
                part = part._data
            if isinstance(part, (np.ndarray, list)):
                return np.array(part, copy=True)
            if isinstance(part, tuple):
                return tuple(freeze(x) for x in part)
            return part

        index = freeze(index)
        data = self._data[index]

        def vjp(g):
            result = np.zeros_like(self._data)
            # add.at accumulates repeated indices instead of overwriting them.
            np.add.at(result, index, g)
            return (result,)

        return self._op(data, (self,), vjp)

    def exp(self):
        result = np.exp(self._data)
        return self._op(result, (self,), lambda g: (g * result,))

    def log(self):
        if np.any(self._data <= 0):
            raise ValueError("log requires strictly positive inputs.")
        return self._op(np.log(self._data), (self,), lambda g: (g / self._data,))

    def tanh(self):
        result = np.tanh(self._data)
        return self._op(result, (self,), lambda g: (g * (1 - result**2),))

    def sigmoid(self):
        # exp(-abs(x)) avoids overflow for large negative inputs.
        e = np.exp(-np.abs(self._data))
        result = np.where(self._data >= 0, 1 / (1 + e), e / (1 + e))
        return self._op(result, (self,), lambda g: (g * result * (1 - result),))

    def relu(self):
        return self._op(np.maximum(self._data, 0), (self,), lambda g: (g * (self._data > 0),))

    def log_softmax(self, axis=-1):
        # Subtracting a constant shift leaves softmax unchanged and prevents
        # exp(large logits) from overflowing. No gradient through the shift
        # is necessary: its contributions cancel algebraically.
        shift = Tensor(self._data.max(axis=axis, keepdims=True), dtype=self.dtype)
        shifted = self - shift
        return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()

    def softmax(self, axis=-1):
        return self.log_softmax(axis=axis).exp()


def tensor(data, requires_grad=False, dtype=None):
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(*shape, requires_grad=False, dtype=np.float64):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad, dtype)


def ones(*shape, requires_grad=False, dtype=np.float64):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(np.ones(shape, dtype=dtype), requires_grad, dtype)


def randn(*shape, requires_grad=False, dtype=np.float64):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_rng.standard_normal(shape), requires_grad, dtype)


def arange(*args, dtype=np.float64):
    return Tensor(np.arange(*args, dtype=dtype), dtype=dtype)


def cat(tensors, axis=0):
    """Concatenate tensors along an existing axis."""
    tensors = tuple(_as_tensor(t) for t in tensors)
    if not tensors:
        raise ValueError("cat needs at least one tensor.")
    data = np.concatenate([t._data for t in tensors], axis=axis)
    boundaries = np.cumsum([t.shape[axis] for t in tensors])[:-1]
    return Tensor._op(data, tensors, lambda g: tuple(np.split(g, boundaries, axis=axis)))


def stack(tensors, axis=0):
    """Stack equally shaped tensors along a new axis."""
    tensors = tuple(_as_tensor(t) for t in tensors)
    if not tensors or any(t.shape != tensors[0].shape for t in tensors):
        raise ValueError("stack needs nonempty inputs with identical shapes.")
    return cat([t.unsqueeze(axis) for t in tensors], axis=axis)


def where(condition, x, y):
    """Select x where the boolean condition is True, otherwise y."""
    condition = condition._data if isinstance(condition, Tensor) else np.asarray(condition)
    if condition.dtype.kind != "b":
        raise TypeError("where requires a boolean condition.")
    condition = condition.copy()
    x, y = _as_tensor(x), _as_tensor(y)
    data = np.where(condition, x._data, y._data)
    return Tensor._op(
        data,
        (x, y),
        lambda g: (
            _unbroadcast(np.where(condition, g, 0), x.shape),
            _unbroadcast(np.where(condition, 0, g), y.shape),
        ),
    )
