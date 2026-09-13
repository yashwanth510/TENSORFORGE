"""A single-number autograd engine to learn before reading tensor.py.

This independent implementation uses Python numbers only. Unlike Tensor,
Value.backward() resets gradients on every call.
"""

import math


class Value:
    """A number, its derivative, and its immediate computation history."""

    def __init__(self, data, _children=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    @staticmethod
    def _wrap(other):
        # Constants participate in arithmetic just like user-created Values.
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other))

        def backward():
            # d(a+b)/da = d(a+b)/db = 1. Multiply by the incoming gradient.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other))
        # Save forward values so changing .data cannot alter this derivative.
        a, b = self.data, other.data

        def backward():
            self.grad += b * out.grad
            other.grad += a * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) - self

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)) or not math.isfinite(exponent):
            raise TypeError("The exponent must be a finite Python int or float.")
        if self.data < 0 and not float(exponent).is_integer():
            raise ValueError("A negative base requires an integer exponent.")
        if self.data == 0 and exponent < 0:
            raise ZeroDivisionError("Zero cannot have a negative power.")
        if self.data == 0 and 0 < exponent < 1:
            raise ValueError("This power has no finite derivative at zero.")
        out = Value(self.data**exponent, (self,))
        local = 0.0 if exponent == 0 else exponent * self.data ** (exponent - 1)

        def backward():
            self.grad += local * out.grad

        out._backward = backward
        return out

    def __truediv__(self, other):
        return self * (self._wrap(other) ** -1)

    def __rtruediv__(self, other):
        return self._wrap(other) / self

    def relu(self):
        out = Value(max(0.0, self.data), (self,))
        local = float(self.data > 0)

        def backward():
            # At zero, choose gradient zero: a convention, not a derivative.
            self.grad += local * out.grad

        out._backward = backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,))
        local = out.data

        def backward():
            self.grad += local * out.grad

        out._backward = backward
        return out

    def log(self):
        if self.data <= 0:
            raise ValueError("log requires a positive input.")
        out = Value(math.log(self.data), (self,))
        local = 1 / self.data

        def backward():
            self.grad += local * out.grad

        out._backward = backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,))
        local = 1 - out.data**2

        def backward():
            self.grad += local * out.grad

        out._backward = backward
        return out

    def backward(self):
        # Postorder puts every operation after its inputs. An explicit stack
        # performs depth-first search without a recursion-depth limitation.
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent in node._prev)
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
