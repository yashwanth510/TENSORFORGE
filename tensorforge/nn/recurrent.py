"""Single-layer batch-first recurrent networks built from Tensor operations."""

from ..tensor import stack, zeros
from .modules import Linear, Module, _positive_int


class _Recurrent(Module):
    gates = 1

    def __init__(self, input_size, hidden_size):
        super().__init__()
        _positive_int(input_size, "input_size")
        _positive_int(hidden_size, "hidden_size")
        self.input_size, self.hidden_size = input_size, hidden_size
        self.input = Linear(input_size, self.gates * hidden_size)
        self.hidden = Linear(hidden_size, self.gates * hidden_size)

    def _validate(self, x, h):
        if x.ndim != 3 or x.shape[-1] != self.input_size or x.shape[1] == 0:
            raise ValueError("Recurrent input must have shape (batch, nonempty_time, input_size).")
        if h is None:
            h = zeros(x.shape[0], self.hidden_size, dtype=x.dtype)
        if h.shape != (x.shape[0], self.hidden_size):
            raise ValueError("Hidden state must have shape (batch, hidden_size).")
        return h


class RNN(_Recurrent):
    """Tanh recurrence h_t = tanh(Wx_t + Uh_(t-1) + b)."""

    def forward(self, x, h=None):
        h = self._validate(x, h)
        outputs = []
        for time in range(x.shape[1]):
            h = (self.input(x[:, time]) + self.hidden(h)).tanh()
            outputs.append(h)
        return stack(outputs, axis=1), h


class LSTM(_Recurrent):
    """Input, forget, candidate, and output gates with a separate cell state."""

    gates = 4

    def forward(self, x, state=None):
        h, c = (None, None) if state is None else state
        h = self._validate(x, h)
        c = zeros(*h.shape, dtype=x.dtype) if c is None else c
        if c.shape != h.shape:
            raise ValueError("Cell and hidden state shapes must match.")
        outputs, width = [], self.hidden_size
        for time in range(x.shape[1]):
            gates = self.input(x[:, time]) + self.hidden(h)
            i = gates[:, :width].sigmoid()
            f = gates[:, width : 2 * width].sigmoid()
            g = gates[:, 2 * width : 3 * width].tanh()
            o = gates[:, 3 * width :].sigmoid()
            c = f * c + i * g
            h = o * c.tanh()
            outputs.append(h)
        return stack(outputs, axis=1), (h, c)


class GRU(_Recurrent):
    """Reset and update gates; gate order is reset, update, candidate."""

    gates = 3

    def forward(self, x, h=None):
        h = self._validate(x, h)
        outputs, width = [], self.hidden_size
        for time in range(x.shape[1]):
            a, b = self.input(x[:, time]), self.hidden(h)
            reset = (a[:, :width] + b[:, :width]).sigmoid()
            update = (a[:, width : 2 * width] + b[:, width : 2 * width]).sigmoid()
            candidate = (a[:, 2 * width :] + reset * b[:, 2 * width :]).tanh()
            h = (1 - update) * candidate + update * h
            outputs.append(h)
        return stack(outputs, axis=1), h
